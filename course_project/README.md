# 网络空间安全实训：多智能体异构网络流量质量审计与保稀协同清洗

## 快速复现（3 步）

1. **检查环境与数据集**（在仓库根目录执行）：
   ```bash
   python course_project/setup_env.py
   ```
   如有缺失依赖，按提示 `pip install ...` 即可。

2. **直接查看已跑好的实验结果与图表**（无需 API Key）：
   - 核心报告：`course_project/report/report.md`
   - 5 分钟答辩提纲与 Q&A：`course_project/report/ppt_outline.md`、`course_project/report/qa_memo.md`
   - 对比与消融结果表：`course_project/results/main_table.csv`、`course_project/results/ablation.csv`
   - 可视化图表：`course_project/figures/`

3. **如需重新联网运行 LLM 多智能体实验**：
   先配置环境变量（勿将 Key 写入仓库）：
   ```powershell
   $env:LLM_API_KEY="你的sk-key"
   $env:LLM_BASE_URL="https://api.openai-next.com/v1"
   $env:LLM_MODEL="deepseek-v4-1-flash-261017"
   ```
   然后运行 `course_project/src/` 下的主实验入口即可。
