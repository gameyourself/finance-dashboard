
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"

ASSET_STRUCTURE = {
    "流动性资产": ["现金", "货币基金/活期", "其他流动性存款"],
    "投资性资产": ["定期存款", "外币存款", "股票投资", "债券投资", "基金投资", "投资性房地产", "其他投资性资产"],
    "自用性资产": ["自用房产", "自用汽车", "其他自用性资产"],
    "其他资产": ["其他资产"],
}

LIABILITY_STRUCTURE = {
    "流动性负债": ["信用卡负债", "小额消费信贷", "其他流动性负债"],
    "投资性负债": ["金融投资借款", "实业投资借款", "投资性房地产贷款", "其他投资性负债"],
    "自用性负债": ["自住房按揭贷款", "自用车按揭贷款", "其他自用性负债"],
    "其他负债": ["其他负债"],
}

STRUCTURE_ORDER = [
    "流动性资产", "流动性负债",
    "投资性资产", "投资性负债",
    "自用性资产", "自用性负债",
    "其他资产", "其他负债",
]

MONTHS = [f"{i}月" for i in range(1, 13)]


def load_local_config():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_deepseek_api_key():
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    config = load_local_config()
    file_key = str(config.get("deepseek_api_key", "")).strip()
    if file_key:
        return file_key
    raise ValueError("未检测到 DeepSeek API Key，请在环境变量或 config.json 中配置。")


def call_deepseek(prompt: str, system_prompt: str = "你是一个擅长家庭财务建模与建议输出的助手。", timeout: int = 90):
    api_key = get_deepseek_api_key()
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        raise ValueError(f"DeepSeek 连接失败：{e}")

    return r.json()["choices"][0]["message"]["content"]


def load_users():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {"users": []}


def save_users(data):
    USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def empty_month_record(month):
    assets = {k: 0.0 for group in ASSET_STRUCTURE.values() for k in group}
    liabilities = {k: 0.0 for group in LIABILITY_STRUCTURE.values() for k in group}
    return {"month": month, "note": "", "assets": assets, "liabilities": liabilities}


def ensure_months(user):
    existing = {m["month"]: m for m in user["monthly_data"]}
    user["monthly_data"] = [existing.get(m, empty_month_record(m)) for m in MONTHS]


def compute_structure_totals(record):
    totals = {}
    for group, items in ASSET_STRUCTURE.items():
        totals[group] = sum(float(record["assets"].get(i, 0) or 0) for i in items)
    for group, items in LIABILITY_STRUCTURE.items():
        totals[group] = sum(float(record["liabilities"].get(i, 0) or 0) for i in items)
    totals["总资产"] = totals["流动性资产"] + totals["投资性资产"] + totals["自用性资产"] + totals["其他资产"]
    totals["总负债"] = totals["流动性负债"] + totals["投资性负债"] + totals["自用性负债"] + totals["其他负债"]
    totals["净资产"] = totals["总资产"] - totals["总负债"]
    return totals


def build_users_index(users_data):
    return {u["id"]: u for u in users_data.get("users", [])}


def create_user(users_data, name):
    uid = f"user_{int(datetime.now().timestamp())}"
    user = {
        "id": uid,
        "name": name,
        "profile": {
            "name": name,
            "age": "",
            "job": "",
            "city_level": "",
            "marital_status": "",
            "background": "",
            "basic_goal": "",
        },
        "monthly_data": [empty_month_record(m) for m in MONTHS],
    }
    users_data["users"].append(user)
    save_users(users_data)
    return uid


def tolerant_json_loads(text: str):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("输入框为空，请粘贴完整 JSON。")
    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            clipped = raw[start:end + 1]
            return json.loads(clipped)
    raise ValueError("不是有效 JSON，请确认最外层是完整对象。")


def import_ai_data(user, ai_json):
    if not isinstance(ai_json, dict):
        raise ValueError("导入内容不是 JSON 对象。")
    if "profile" in ai_json and isinstance(ai_json["profile"], dict):
        user["profile"].update(ai_json["profile"])

    month_map = {m["month"]: m for m in user["monthly_data"]}
    monthly_data = ai_json.get("monthly_data", [])
    if isinstance(monthly_data, dict):
        monthly_data = [monthly_data]

    for m in monthly_data:
        month = m.get("month")
        if month in month_map:
            month_map[month]["note"] = m.get("note", month_map[month].get("note", ""))
            for k, v in m.get("assets", {}).items():
                if k in month_map[month]["assets"]:
                    month_map[month]["assets"][k] = float(v or 0)
            for k, v in m.get("liabilities", {}).items():
                if k in month_map[month]["liabilities"]:
                    month_map[month]["liabilities"][k] = float(v or 0)
    user["monthly_data"] = [month_map[m] for m in MONTHS]


def month_selector(user):
    month = st.radio("选择月份", MONTHS, horizontal=True)
    current = next(m for m in user["monthly_data"] if m["month"] == month)
    idx = MONTHS.index(month)
    prev1 = user["monthly_data"][idx - 1] if idx - 1 >= 0 else None
    prev2 = user["monthly_data"][idx - 2] if idx - 2 >= 0 else None
    return month, current, prev1, prev2



def inject_table_styles():
    st.markdown(
        """
        <style>
        .pro-balance-wrap{
            border:1px solid #d9deea;
            border-radius:16px;
            overflow:hidden;
            background:#ffffff;
            box-shadow:0 2px 10px rgba(31,41,55,0.04);
            margin-bottom: 12px;
        }
        .pro-balance-header{
            display:grid;
            grid-template-columns:1fr 1fr;
            background:linear-gradient(90deg,#4f7cff 0%, #7c4dff 100%);
            color:#fff;
            font-weight:700;
            font-size:20px;
        }
        .pro-balance-header div{
            padding:14px 18px;
            text-align:center;
        }
        .pro-balance-grid{
            display:grid;
            grid-template-columns:1fr 1fr;
        }
        .pro-balance-col{
            border-right:1px solid #e7ebf3;
        }
        .pro-balance-col:last-child{
            border-right:none;
        }
        .pro-section{
            border-bottom:1px solid #eef2f7;
        }
        .pro-section-title{
            display:flex;
            justify-content:space-between;
            align-items:center;
            background:#f5f7fb;
            padding:12px 16px;
            font-weight:700;
            color:#111827;
            border-bottom:1px solid #e7ebf3;
        }
        .pro-section-total{
            color:#4b5563;
            font-weight:700;
        }
        .pro-row{
            display:flex;
            justify-content:space-between;
            gap:16px;
            padding:11px 16px;
            border-bottom:1px solid #f1f5f9;
            font-size:15px;
        }
        .pro-row:last-child{
            border-bottom:none;
        }
        .pro-row-name{
            color:#374151;
            flex:1;
        }
        .pro-row-value{
            color:#111827;
            min-width:92px;
            text-align:right;
            font-variant-numeric: tabular-nums;
        }
        .pro-summary{
            display:grid;
            grid-template-columns:1fr 1fr;
            background:linear-gradient(90deg,#4f7cff 0%, #7c4dff 100%);
            color:white;
            font-weight:700;
        }
        .pro-summary div{
            display:flex;
            justify-content:space-between;
            padding:14px 18px;
            font-size:18px;
        }
        .pro-muted{
            color:#9ca3af;
        }
        .compare-note-box{
            padding:10px 12px;
            border-left:4px solid #6366f1;
            background:#f8faff;
            border-radius:8px;
            margin-bottom:10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_money(v):
    try:
        return f"{float(v):,.2f}" if float(v) != 0 else "--"
    except Exception:
        return "--"


def build_professional_balance_html(record):
    totals = compute_structure_totals(record)

    asset_groups = [
        ("流动性资产", ASSET_STRUCTURE["流动性资产"]),
        ("投资性资产", ASSET_STRUCTURE["投资性资产"]),
        ("自用性资产", ASSET_STRUCTURE["自用性资产"]),
        ("其他资产", ASSET_STRUCTURE["其他资产"]),
    ]
    liability_groups = [
        ("流动性负债", LIABILITY_STRUCTURE["流动性负债"]),
        ("投资性负债", LIABILITY_STRUCTURE["投资性负债"]),
        ("自用性负债", LIABILITY_STRUCTURE["自用性负债"]),
        ("其他负债", LIABILITY_STRUCTURE["其他负债"]),
    ]

    def render_side(groups, bucket):
        html = ""
        for group_name, items in groups:
            section_total = totals.get(group_name, 0)
            html += f'<div class="pro-section">'
            html += f'<div class="pro-section-title"><span>▼ {group_name}</span><span class="pro-section-total">{format_money(section_total)}</span></div>'
            for item in items:
                value = record[bucket].get(item, 0)
                value_cls = "pro-row-value" if float(value or 0) != 0 else "pro-row-value pro-muted"
                html += f'<div class="pro-row"><div class="pro-row-name">{item}</div><div class="{value_cls}">{format_money(value)}</div></div>'
            html += '</div>'
        return html

    left_html = render_side(asset_groups, "assets")
    right_html = render_side(liability_groups, "liabilities")

    summary_right = totals["总负债"] + totals["净资产"]

    html = f"""
    <div class="pro-balance-wrap">
        <div class="pro-balance-header">
            <div>资产</div>
            <div>负债及净资产</div>
        </div>
        <div class="pro-balance-grid">
            <div class="pro-balance-col">
                {left_html}
            </div>
            <div class="pro-balance-col">
                {right_html}
                <div class="pro-section">
                    <div class="pro-section-title"><span>净资产</span><span class="pro-section-total">{format_money(totals['净资产'])}</span></div>
                </div>
            </div>
        </div>
        <div class="pro-summary">
            <div><span>资产</span><span>{format_money(totals['总资产'])}</span></div>
            <div><span>负债及净资产</span><span>{format_money(summary_right)}</span></div>
        </div>
    </div>
    """
    return html



def render_current_balance_table(record):
    st.subheader(f"{record['month']}资产负债表")
    inject_table_styles()
    totals = compute_structure_totals(record)

    c1, c2, c3 = st.columns(3)
    c1.metric("总资产", f"{totals['总资产']:,.2f}")
    c2.metric("总负债", f"{totals['总负债']:,.2f}")
    c3.metric("净资产", f"{totals['净资产']:,.2f}")

    st.markdown(build_professional_balance_html(record), unsafe_allow_html=True)


def diff_from_previous(current, previous):
    rows = []
    for group, items in ASSET_STRUCTURE.items():
        for item in items:
            cur = float(current["assets"].get(item, 0) or 0)
            pre = float(previous["assets"].get(item, 0) or 0)
            rows.append({"类型": group, "项目": item, "本月": cur, "上月": pre, "差额": cur - pre, "方向": "上升" if cur > pre else ("下降" if cur < pre else "持平")})
    for group, items in LIABILITY_STRUCTURE.items():
        for item in items:
            cur = float(current["liabilities"].get(item, 0) or 0)
            pre = float(previous["liabilities"].get(item, 0) or 0)
            rows.append({"类型": group, "项目": item, "本月": cur, "上月": pre, "差额": cur - pre, "方向": "上升" if cur > pre else ("下降" if cur < pre else "持平")})
    return pd.DataFrame(rows)


def render_compare_table(current, previous):
    current_month = current["month"]
    prev_month = previous["month"]
    st.subheader(f"{current_month} vs {prev_month} 对比表")
    st.markdown(
        f'<div class="compare-note-box">当前列“<b>{current_month}</b>”代表本月，列“<b>{prev_month}</b>”代表上个月；建议优先关注“差额”和“方向”。</div>',
        unsafe_allow_html=True
    )
    df = diff_from_previous(current, previous).rename(columns={"本月": current_month, "上月": prev_month})
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_structure_bars(user):
    st.subheader("结构性条形图（按月份）")
    data = []
    for record in user["monthly_data"]:
        totals = compute_structure_totals(record)
        row = {"month": record["month"]}
        row.update({k: totals[k] for k in STRUCTURE_ORDER})
        data.append(row)
    df = pd.DataFrame(data)

    chart_mode = st.selectbox("查看模式", ["资产结构", "负债结构"])
    if chart_mode == "资产结构":
        groups = ["流动性资产", "投资性资产", "自用性资产", "其他资产"]
    else:
        groups = ["流动性负债", "投资性负债", "自用性负债", "其他负债"]

    fig = go.Figure()
    for g in groups:
        fig.add_bar(x=df["month"], y=df[g], name=g)
    fig.update_layout(barmode="stack", xaxis_title="月份", yaxis_title="金额")
    st.plotly_chart(fig, use_container_width=True)


def render_structure_lines(user):
    st.subheader("结构性趋势折线图")
    rows = []
    for record in user["monthly_data"]:
        totals = compute_structure_totals(record)
        row = {"month": record["month"]}
        row.update({k: totals[k] for k in STRUCTURE_ORDER})
        rows.append(row)
    df = pd.DataFrame(rows)

    selected_groups = st.multiselect(
        "勾选想看的结构维度",
        options=STRUCTURE_ORDER,
        default=["流动性资产", "流动性负债", "投资性资产", "自用性负债"]
    )
    if not selected_groups:
        st.info("请至少勾选一个维度。")
        return

    fig = go.Figure()
    for group in selected_groups:
        fig.add_trace(go.Scatter(
            x=df["month"], y=df[group], mode="lines+markers+text", name=group,
            text=[f"{v:,.0f}" for v in df[group]], textposition="top center"
        ))
    fig.update_layout(xaxis_title="月份", yaxis_title="金额")
    st.plotly_chart(fig, use_container_width=True)

    delta = df.copy()
    for group in selected_groups:
        delta[f"{group}_较上月差额"] = delta[group].diff().fillna(0)
    cols = ["month"] + selected_groups + [f"{g}_较上月差额" for g in selected_groups]
    st.dataframe(delta[cols], use_container_width=True, hide_index=True)


def render_profile_editor(user):
    st.subheader("人物背景与基础目标")
    profile = user["profile"]
    c1, c2 = st.columns(2)
    profile["name"] = c1.text_input("姓名/代号", value=str(profile.get("name", "")))
    profile["age"] = c2.text_input("年龄", value=str(profile.get("age", "")))
    c3, c4 = st.columns(2)
    profile["job"] = c3.text_input("职业", value=str(profile.get("job", "")))
    profile["city_level"] = c4.text_input("城市级别", value=str(profile.get("city_level", "")))
    c5, c6 = st.columns(2)
    profile["marital_status"] = c5.text_input("婚姻状态", value=str(profile.get("marital_status", "")))
    profile["basic_goal"] = c6.text_input("基础理财目标", value=str(profile.get("basic_goal", "")))
    profile["background"] = st.text_area("背景描述", value=str(profile.get("background", "")), height=120)


def render_month_editor(record):
    st.subheader(f"{record['month']}手动编辑")
    st.caption("只填核心字段即可，不必全部填满。")
    with st.expander("资产项", expanded=True):
        for group, items in ASSET_STRUCTURE.items():
            st.markdown(f"**{group}**")
            cols = st.columns(3)
            for idx, item in enumerate(items):
                record["assets"][item] = cols[idx % 3].number_input(
                    item, min_value=0.0, value=float(record["assets"].get(item, 0) or 0), step=1000.0, key=f"{record['month']}_{item}"
                )
    with st.expander("负债项", expanded=True):
        for group, items in LIABILITY_STRUCTURE.items():
            st.markdown(f"**{group}**")
            cols = st.columns(3)
            for idx, item in enumerate(items):
                record["liabilities"][item] = cols[idx % 3].number_input(
                    item, min_value=0.0, value=float(record["liabilities"].get(item, 0) or 0), step=1000.0, key=f"{record['month']}_{item}_debt"
                )
    record["note"] = st.text_area("本月说明", value=record.get("note", ""), key=f"{record['month']}_note")


BACKGROUND_TEMPLATE = """示例模板：
30岁，二线城市上班族，已婚，有一套自住房并承担房贷；
月收入比较稳定，消费偏日常型，没有特别激进的投资习惯；
希望先把现金流稳定下来，再慢慢建立应急资金，逐步降低负债压力。"""


def build_profile_prompt(background_text, selected_goals):
    goals = selected_goals if selected_goals else ["先建立 3-6 个月应急资金", "优先稳定月度现金流", "逐步压降负债规模"]
    return f"""
请先生成一个普通人的人物背景 profile，输出只能是 JSON。
要求：
1. 背景要与下面文字保持一致或补充完善；
2. basic_goal 要尽量贴近以下目标偏好；
3. 输出字段只包含：name, age, job, city_level, marital_status, background, basic_goal

人物背景：
{background_text.strip() if background_text.strip() else "普通人画像，自行生成"}

理财目标偏好：
{chr(10).join("- " + g for g in goals)}

输出格式：
{{
  "name": "用户A",
  "age": 30,
  "job": "......",
  "city_level": "......",
  "marital_status": "......",
  "background": "......",
  "basic_goal": "......"
}}
"""


def build_single_month_prompt(month, prev_month_data, background_text, selected_goals, include_optional_fields):
    optional_items = []
    if "货币基金/活期" in include_optional_fields:
        optional_items.append("货币基金/活期")
    if "股票投资/基金投资" in include_optional_fields:
        optional_items.extend(["股票投资", "基金投资"])
    if "信用卡负债/小额消费信贷" in include_optional_fields:
        optional_items.extend(["信用卡负债", "小额消费信贷"])
    if "其他资产/其他负债" in include_optional_fields:
        optional_items.extend(["其他资产", "其他负债"])

    optional_text = "、".join(optional_items) if optional_items else "无额外可选字段"

    if prev_month_data is None:
        context_text = """
这是第一个月份，请围绕以下初始参考数据生成 1月：
- 现金：82000
- 定期存款：10000
- 自住房按揭贷款：360000
- 总资产：9.2万元
- 总负债：36万元
- 净资产：-26.8万元
"""
    else:
        context_text = f"请基于上个月数据继续生成{month}，必须与上个月保持连续性：\n{json.dumps(prev_month_data, ensure_ascii=False)}"

    return f"""
请只生成 {month} 的单月家庭财务数据，输出必须是标准 JSON，不要输出解释文字。

人物背景：
{background_text.strip() if background_text.strip() else "普通人画像"}

理财目标偏好：
{chr(10).join("- " + g for g in selected_goals)}

重点覆盖字段：
- 现金
- 定期存款
- 自住房按揭贷款
- 可选字段：{optional_text}

生成原则：
- 数值要符合普通人的财务轨迹
- 每月变化要有原因，比如工资入账、日常支出、节假日消费、奖金、还贷、理财转移等
- 不必把所有项目都填满，保留空白是允许的
- 必须让年度前后连贯
- 房贷余额整体缓慢下降
- 现金围绕收入和支出进行合理波动
- 若出现投资项目，规模不要夸张

上下文：
{context_text}

输出格式：
{{
  "month": "{month}",
  "note": "......",
  "assets": {{
    "现金": 0,
    "货币基金/活期": 0,
    "其他流动性存款": 0,
    "定期存款": 0,
    "外币存款": 0,
    "股票投资": 0,
    "债券投资": 0,
    "基金投资": 0,
    "投资性房地产": 0,
    "其他投资性资产": 0,
    "自用房产": 0,
    "自用汽车": 0,
    "其他自用性资产": 0,
    "其他资产": 0
  }},
  "liabilities": {{
    "信用卡负债": 0,
    "小额消费信贷": 0,
    "其他流动性负债": 0,
    "金融投资借款": 0,
    "实业投资借款": 0,
    "投资性房地产贷款": 0,
    "其他投资性负债": 0,
    "自住房按揭贷款": 0,
    "自用车按揭贷款": 0,
    "其他自用性负债": 0,
    "其他负债": 0
  }}
}}
"""


def generate_year_data_with_progress(background_text, selected_goals, include_optional_fields):
    progress_bar = st.progress(0)
    status = st.empty()
    month_preview = st.empty()

    # profile
    status.info("正在生成用户背景...")
    profile_raw = call_deepseek(
        build_profile_prompt(background_text, selected_goals),
        system_prompt="你是一名擅长家庭财务人物画像构建的助手，只输出 JSON。"
    )
    profile = tolerant_json_loads(profile_raw)

    monthly_data = []
    prev = None

    for idx, month in enumerate(MONTHS, start=1):
        status.info(f"正在生成 {month} 数据...")
        raw = call_deepseek(
            build_single_month_prompt(month, prev, background_text or profile.get("background", ""), selected_goals, include_optional_fields),
            system_prompt="你是一名擅长家庭财务月度建模的助手，只输出 JSON。"
        )
        month_data = tolerant_json_loads(raw)
        monthly_data.append(month_data)
        prev = month_data

        progress = idx / 12
        progress_bar.progress(progress)
        month_preview.success(f"{month} 已生成完成（{idx}/12）")
        status.info(f"当前进度：{idx}/12，已完成到 {month}")

    status.success("12个月数据已全部生成完成。")
    return {"profile": profile, "monthly_data": monthly_data}


def build_month_suggestion_prompt(user, current, prev1, prev2):
    current_totals = compute_structure_totals(current)
    prev1_totals = compute_structure_totals(prev1) if prev1 else {}
    prev2_totals = compute_structure_totals(prev2) if prev2 else {}

    return f"""
你是一名家庭财务顾问，请基于用户背景与本月、上月、上上月数据，输出结构化建议。
不要输出 JSON，请直接输出分段中文，格式如下：

【当月概览】
2-3句

【流动性资产建议】
2-3句

【流动性负债建议】
2-3句

【投资性资产建议】
2-3句

【投资性负债建议】
2-3句

【自用性资产与负债建议】
2-3句

【其他项提醒】
1-2句

【当月优先动作】
1句

用户背景：
{json.dumps(user["profile"], ensure_ascii=False)}

本月数据：
{json.dumps({"month": current["month"], "note": current.get("note", ""), "totals": current_totals, "assets": current["assets"], "liabilities": current["liabilities"]}, ensure_ascii=False)}

上月数据：
{json.dumps({"month": prev1["month"], "note": prev1.get("note", ""), "totals": prev1_totals} if prev1 else {}, ensure_ascii=False)}

上上月数据：
{json.dumps({"month": prev2["month"], "note": prev2.get("note", ""), "totals": prev2_totals} if prev2 else {}, ensure_ascii=False)}
"""


st.set_page_config(page_title="本地家庭财务云", layout="wide")
st.title("本地家庭财务云")
st.caption("支持多人物背景、12个月资产负债表、结构化图表、本地 JSON 保存、DeepSeek 月度建议。")

users_data = load_users()
user_index = build_users_index(users_data)

with st.sidebar:
    st.header("用户管理")
    user_names = {u["id"]: u["name"] for u in users_data.get("users", [])}
    selected_id = None
    if user_names:
        selected_id = st.selectbox("选择用户", options=list(user_names.keys()), format_func=lambda x: user_names[x])

    new_name = st.text_input("新增用户名称", value="")
    if st.button("创建新用户"):
        if new_name.strip():
            create_user(users_data, new_name.strip())
            st.success(f"已创建用户：{new_name}")
            st.rerun()
        else:
            st.warning("请先输入用户名称")

if user_names and not selected_id:
    selected_id = list(user_names.keys())[0]

if not selected_id:
    st.info("先在左侧创建一个用户，再开始录入和分析。")
    st.stop()

user = user_index[selected_id]
ensure_months(user)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["版本1：月份访问", "版本2：结构分析", "人物背景", "AI导入", "数据导出"])

with tab1:
    month, current, prev1, prev2 = month_selector(user)
    left, right = st.columns([1.2, 1])
    with left:
        render_current_balance_table(current)
        if prev1:
            render_compare_table(current, prev1)
    with right:
        render_month_editor(current)
        if st.button("保存当前月份修改"):
            save_users(users_data)
            st.success("已保存到本地 JSON。")

    render_structure_lines(user)

    st.subheader("当月建议（DeepSeek）")
    st.caption("这里输出的是结构化文字建议，不再显示 JSON。")
    if st.button("生成当月建议"):
        try:
            result = call_deepseek(
                build_month_suggestion_prompt(user, current, prev1, prev2),
                system_prompt="你是一名擅长家庭财务分析与中文结构化表达的顾问。"
            )
            st.markdown(result)
        except Exception as e:
            st.error(f"生成失败：{e}")

with tab2:
    render_structure_bars(user)
    st.subheader("整体性分析（DeepSeek）")
    if st.button("生成年度整体分析"):
        yearly = []
        for record in user["monthly_data"]:
            row = {"month": record["month"]}
            row.update(compute_structure_totals(record))
            yearly.append(row)
        prompt = f"""
请基于这位用户12个月的结构性财务数据，直接输出结构化中文分析，不要输出JSON。
格式：
【整体趋势概览】
【资产端问题】
【负债端问题】
【流动性建议】
【投资配置建议】
【下一阶段最重要动作】

数据：
{json.dumps({'profile': user['profile'], 'yearly': yearly}, ensure_ascii=False)}
"""
        try:
            result = call_deepseek(prompt, system_prompt="你是一名擅长家庭财务年度分析的顾问。")
            st.markdown(result)
        except Exception as e:
            st.error(f"生成失败：{e}")

with tab3:
    render_profile_editor(user)
    if st.button("保存人物背景"):
        save_users(users_data)
        st.success("人物背景已保存")

with tab4:
    st.subheader("AI生成12个月数据 + JSON导入")
    st.write("这里可以先输入人物背景，再勾选目标和字段偏好，然后按月份逐步生成 12 个月数据。")

    st.markdown("**人物背景模板参考**")
    st.code(BACKGROUND_TEMPLATE, language="text")

    background_text = st.text_area("单独输入人物背景", value=user["profile"].get("background", ""), height=140)

    goal_options = [
        "先建立 3-6 个月应急资金",
        "优先稳定月度现金流",
        "逐步压降负债规模",
        "控制消费贷比例",
        "保留一定流动资金后再做小额理财",
    ]
    selected_goals = st.multiselect(
        "理财目标偏好（自选）",
        options=goal_options,
        default=["先建立 3-6 个月应急资金", "优先稳定月度现金流", "逐步压降负债规模"]
    )

    include_optional_fields = st.multiselect(
        "可选覆盖字段（自选）",
        options=["货币基金/活期", "股票投资/基金投资", "信用卡负债/小额消费信贷", "其他资产/其他负债"],
        default=["货币基金/活期", "股票投资/基金投资", "信用卡负债/小额消费信贷", "其他资产/其他负债"]
    )

    st.text_area(
        "给 DeepSeek 的生成逻辑说明（预览）",
        value=(
            "系统会先生成人物 profile，然后按 1月 到 12月 逐月生成数据；\n"
            "每生成完一个月份，都会更新进度条与状态提示；\n"
            "后一个月份会参考前一个月份数据，保证年度连续性。"
        ),
        height=100
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("调用 DeepSeek 按月份生成12个月数据"):
            try:
                generated = generate_year_data_with_progress(background_text, selected_goals, include_optional_fields)
                raw = json.dumps(generated, ensure_ascii=False, indent=2)
                st.session_state["generated_json_text"] = raw
                st.success("12个月数据已生成完成，下面可以直接导入。")
            except Exception as e:
                st.error(f"生成失败：{e}")

    with c2:
        if st.button("导入下方 JSON 到当前用户"):
            try:
                ai_json = tolerant_json_loads(st.session_state.get("generated_json_text", "") or st.session_state.get("manual_json_text", ""))
                import_ai_data(user, ai_json)
                if background_text.strip():
                    user["profile"]["background"] = background_text.strip()
                if selected_goals:
                    user["profile"]["basic_goal"] = "；".join(selected_goals)
                save_users(users_data)
                st.success("导入成功，当前用户的 12 个月数据已更新。")
            except Exception as e:
                st.error(f"导入失败：{e}")

    st.text_area(
        "AI返回的 JSON（可修改后再导入）",
        value=st.session_state.get("generated_json_text", ""),
        height=340,
        key="manual_json_text"
    )

with tab5:
    st.subheader("导出与备份")
    export_text = json.dumps(user, ensure_ascii=False, indent=2)
    st.download_button("下载当前用户数据(JSON)", export_text, file_name=f"{user['name']}_finance.json")
    st.download_button("下载全部用户数据(JSON)", json.dumps(users_data, ensure_ascii=False, indent=2), file_name="all_users_finance.json")
