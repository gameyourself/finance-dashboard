
import os
import json
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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

DEFAULT_PROMPT_TEMPLATE = """你是一名家庭资产配置顾问，请你基于以下初始数据，生成一个“普通人、合理但不夸张”的年度财务样本。

【人物背景要求】
1. 先生成一个简洁的人物背景，包含：年龄、职业、所在城市级别、婚姻状态、是否有房贷、收入特点、消费特点。
2. 风格要贴近现实生活，不要过度理想化。

【初始参考数据】
- 总资产：9.2万元
- 总负债：36万元
- 净资产：-26.8万元
- 已知核心项：
  - 现金：82000
  - 定期存款：10000
  - 自住房按揭贷款：360000

【输出任务】
请生成 12 个月（1月到12月）的财务情况，按月给出即可，不必把所有字段都填满，但要优先填核心字段，保持连续性和合理波动。
至少覆盖以下字段中的核心部分：
- 现金
- 货币基金/活期（可选）
- 定期存款
- 股票投资 / 基金投资（可选）
- 自住房按揭贷款
- 信用卡负债 / 小额消费信贷（如需要可少量设置）
- 其他资产 / 其他负债（可选）

【生成原则】
1. 数值要符合普通人的财务轨迹；
2. 每月变化要有原因，比如工资入账、日常支出、节假日消费、奖金、还贷、理财转移等；
3. 不要把所有项目都填满，保留空白是允许的；
4. 必须让年度数据前后连贯；
5. 房贷余额整体缓慢下降；
6. 现金可能围绕收入和支出波动；
7. 若出现投资项目，规模不要夸张。

【额外输出】
1. 给出一个“最基础的理财目标”，要简单务实，比如“先建立3-6个月应急资金”或“控制消费贷比例”等；
2. 再给出每个月一句简短说明，解释该月财务变化原因；
3. 输出格式必须是 JSON。

【JSON格式要求】
{
  "profile": {
    "name": "用户A",
    "age": 30,
    "job": "......",
    "city_level": "......",
    "marital_status": "......",
    "background": "......",
    "basic_goal": "......"
  },
  "monthly_data": [
    {
      "month": "1月",
      "note": "......",
      "assets": {
        "现金": 82000,
        "货币基金/活期": 0,
        "其他流动性存款": 0,
        "定期存款": 10000,
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
      },
      "liabilities": {
        "信用卡负债": 0,
        "小额消费信贷": 0,
        "其他流动性负债": 0,
        "金融投资借款": 0,
        "实业投资借款": 0,
        "投资性房地产贷款": 0,
        "其他投资性负债": 0,
        "自住房按揭贷款": 360000,
        "自用车按揭贷款": 0,
        "其他自用性负债": 0,
        "其他负债": 0
      }
    }
  ]
}
"""

def load_users():
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    return {"users": []}

def save_users(data):
    USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def empty_month_record(month):
    assets = {k: 0 for group in ASSET_STRUCTURE.values() for k in group}
    liabilities = {k: 0 for group in LIABILITY_STRUCTURE.values() for k in group}
    return {"month": month, "note": "", "assets": assets, "liabilities": liabilities}

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

def ensure_months(user):
    existing = {m["month"]: m for m in user["monthly_data"]}
    user["monthly_data"] = [existing.get(m, empty_month_record(m)) for m in MONTHS]

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

def load_local_config():
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
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

def call_deepseek(prompt: str):
    api_key = get_deepseek_api_key()

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个擅长家庭财务建模、输出标准 JSON 的助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
        raise ValueError("DeepSeek 返回内容不是可解析 JSON，请调整提示词。")

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

def import_ai_data(user, ai_json):
    if "profile" in ai_json:
        user["profile"].update(ai_json["profile"])
    month_map = {m["month"]: m for m in user["monthly_data"]}
    for m in ai_json.get("monthly_data", []):
        if m["month"] in month_map:
            month_map[m["month"]]["note"] = m.get("note", "")
            for k, v in m.get("assets", {}).items():
                if k in month_map[m["month"]]["assets"]:
                    month_map[m["month"]]["assets"][k] = float(v or 0)
            for k, v in m.get("liabilities", {}).items():
                if k in month_map[m["month"]]["liabilities"]:
                    month_map[m["month"]]["liabilities"][k] = float(v or 0)
    user["monthly_data"] = [month_map[m] for m in MONTHS]

def month_selector(user):
    month = st.radio("选择月份", MONTHS, horizontal=True)
    current = next(m for m in user["monthly_data"] if m["month"] == month)
    idx = MONTHS.index(month)
    prev1 = user["monthly_data"][idx - 1] if idx - 1 >= 0 else None
    prev2 = user["monthly_data"][idx - 2] if idx - 2 >= 0 else None
    return month, current, prev1, prev2

def render_current_balance_table(record):
    st.subheader(f"{record['month']}资产负债表")
    asset_rows = []
    for group, items in ASSET_STRUCTURE.items():
        for item in items:
            asset_rows.append([group, item, float(record["assets"].get(item, 0) or 0)])
    liability_rows = []
    for group, items in LIABILITY_STRUCTURE.items():
        for item in items:
            liability_rows.append([group, item, float(record["liabilities"].get(item, 0) or 0)])

    df_a = pd.DataFrame(asset_rows, columns=["资产分类", "资产项目", "金额"])
    df_l = pd.DataFrame(liability_rows, columns=["负债分类", "负债项目", "金额"])
    totals = compute_structure_totals(record)
    c1, c2, c3 = st.columns(3)
    c1.metric("总资产", f"{totals['总资产']:,.0f}")
    c2.metric("总负债", f"{totals['总负债']:,.0f}")
    c3.metric("净资产", f"{totals['净资产']:,.0f}")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_a, use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(df_l, use_container_width=True, hide_index=True)

def render_compare_table(current, previous):
    st.subheader("本月 vs 上月 对比表")
    df = diff_from_previous(current, previous)

    def style_row(row):
        color = ""
        if row["差额"] > 0:
            color = "color: blue;"
        elif row["差额"] < 0:
            color = "color: red;"
        return ["", "", color, color, color, ""]  # not perfect but enough

    st.dataframe(df, use_container_width=True, hide_index=True)
    styled = df.style.apply(style_row, axis=1)
    st.write("颜色说明：高于上月可理解为蓝色，低于上月可理解为红色。")
    st.dataframe(styled, use_container_width=True)

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

def render_delta_lines(user, month_index):
    st.subheader("与上月 / 上上月差额折线图")
    points1, points2 = [], []
    for i, record in enumerate(user["monthly_data"]):
        totals = compute_structure_totals(record)
        if i >= 1:
            pre1 = compute_structure_totals(user["monthly_data"][i-1])
            points1.append({"month": record["month"], "净资产差额(较上月)": totals["净资产"] - pre1["净资产"]})
        else:
            points1.append({"month": record["month"], "净资产差额(较上月)": 0})
        if i >= 2:
            pre2 = compute_structure_totals(user["monthly_data"][i-2])
            points2.append({"month": record["month"], "净资产差额(较上上月)": totals["净资产"] - pre2["净资产"]})
        else:
            points2.append({"month": record["month"], "净资产差额(较上上月)": 0})
    df1 = pd.DataFrame(points1)
    df2 = pd.DataFrame(points2)

    fig1 = go.Figure()
    fig1.add_scatter(x=df1["month"], y=df1["净资产差额(较上月)"], mode="lines+markers", name="较上月")
    fig1.update_layout(xaxis_title="月份", yaxis_title="净资产差额")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=df2["month"], y=df2["净资产差额(较上上月)"], mode="lines+markers", name="较上上月")
    fig2.update_layout(xaxis_title="月份", yaxis_title="净资产差额")
    st.plotly_chart(fig2, use_container_width=True)

def get_month_analysis(user, current, prev1=None, prev2=None):
    totals = compute_structure_totals(current)
    summary = {
        "profile": user["profile"],
        "month": current["month"],
        "current_totals": totals,
        "current_record": current,
        "prev1_note": prev1["note"] if prev1 else "",
        "prev2_note": prev2["note"] if prev2 else "",
    }
    prompt = f"""
你是一名务实型家庭财务顾问，请根据下面的用户背景、理财目标和当月资产负债情况，输出模块化建议。
要求：
1. 从流动性、投资性、自用性、其他四个模块分别给建议；
2. 每个模块 2-3 句话；
3. 语气务实，不要空泛；
4. 结合本月数据情况；
5. 最后补一句“当月优先事项”。

数据如下：
{json.dumps(summary, ensure_ascii=False)}
"""
    return call_deepseek(prompt)

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
            new_id = create_user(users_data, new_name.strip())
            st.success(f"已创建用户：{new_name}")
            st.rerun()
        else:
            st.warning("请先输入用户名称")
    st.markdown("---")
    st.subheader("DeepSeek 提示词模板")
    prompt_text = st.text_area("可复制给 DeepSeek", value=DEFAULT_PROMPT_TEMPLATE, height=420)
    st.download_button("下载提示词模板", prompt_text, file_name="deepseek_prompt_template.txt")

if not selected_id and user_names:
    selected_id = list(user_names.keys())[0]

if not selected_id:
    st.info("先在左侧创建一个用户，再开始录入和分析。")
    st.stop()

user = user_index[selected_id]
ensure_months(user)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["版本1：月份访问", "版本2：结构分析", "人物背景", "AI 导入", "数据导出"])

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

    render_delta_lines(user, MONTHS.index(month))

    st.subheader("当月建议（DeepSeek）")
    if st.button("生成当月建议"):
        try:
            result = get_month_analysis(user, current, prev1, prev2)
            st.json(result)
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
你是一名家庭财务分析顾问，请对下面这位用户 12 个月的结构性数据进行整体性分析。
请输出：
1. 整体趋势概览；
2. 资产端问题；
3. 负债端问题；
4. 流动性建议；
5. 投资配置建议；
6. 下一阶段最重要的一个动作。
数据：
{json.dumps({'profile': user['profile'], 'yearly': yearly}, ensure_ascii=False)}
"""
        try:
            result = call_deepseek(prompt)
            st.json(result)
        except Exception as e:
            st.error(f"生成失败：{e}")

with tab3:
    render_profile_editor(user)
    if st.button("保存人物背景"):
        save_users(users_data)
        st.success("人物背景已保存")

with tab4:
    st.subheader("从 DeepSeek 结果导入")
    st.write("方式一：把左侧提示词复制给 DeepSeek，拿到 JSON 后粘贴到下面；方式二：自行修改提示词后再导入。")
    ai_json_text = st.text_area("粘贴 DeepSeek 返回的 JSON", height=380)
    if st.button("导入 JSON 到当前用户"):
        try:
            ai_json = json.loads(ai_json_text)
            import_ai_data(user, ai_json)
            save_users(users_data)
            st.success("导入成功，已保存到本地")
        except Exception as e:
            st.error(f"导入失败：{e}")

with tab5:
    st.subheader("导出与备份")
    export_text = json.dumps(user, ensure_ascii=False, indent=2)
    st.download_button("下载当前用户数据(JSON)", export_text, file_name=f"{user['name']}_finance.json")
    st.download_button("下载全部用户数据(JSON)", json.dumps(users_data, ensure_ascii=False, indent=2), file_name="all_users_finance.json")
