# 本地家庭财务云

## 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 2. 配置 DeepSeek API Key
**不要把密钥直接写进代码。**
请在终端设置环境变量：

### Windows PowerShell
```powershell
$env:DEEPSEEK_API_KEY="你的DeepSeek密钥"
streamlit run app.py
```

### Windows CMD
```cmd
set DEEPSEEK_API_KEY=你的DeepSeek密钥
streamlit run app.py
```

### macOS / Linux
```bash
export DEEPSEEK_API_KEY="你的DeepSeek密钥"
streamlit run app.py
```

## 3. 功能
- 多用户背景管理
- 12个月资产负债表
- 按月份查看
- 结构性堆叠条形图
- 与上月 / 上上月净资产差额折线图
- 本月 vs 上月对比表
- DeepSeek 月度建议、年度整体分析
- 本地 JSON 持久化保存

## 4. DeepSeek 提示词
应用左侧边栏内置了可复制模板，也可使用项目中的 `deepseek_prompt_template.txt`
