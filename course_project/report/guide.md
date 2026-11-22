# 后期精简、复现与交稿操作指南

> 本指南帮助你快速根据课程要求（字数/页数限制、答辩时长、源码打包规范）裁剪材料并确保安全合规交稿。

---

## 一、交付物目录地图（全部位于 `course_project/`）

| 路径 | 内容 | 交稿用途 |
|---|---|---|
| `course_project/report/report.md` | 完整课程技术报告（含摘要、7 章正文、附录） | 主报告底稿（可直接转 Word/PDF） |
| `course_project/report/highlights_transcript.md` | UNSW `seed 102` 与 CICIDS 真实协商高光实录 | 报告附录 / PPT 案例演示素材 |
| `course_project/report/ppt_outline.md` | 13 页答辩 PPT 逐页框架与演讲词 | 制作答辩幻灯片 |
| `course_project/report/qa_memo.md` | 7 个答辩高频尖锐 Q&A 标准口径 | 赛前/答辩前背诵演练 |
| `course_project/figures/fig1..fig5.png` | 5 张 220 DPI 高清插图（均已人工目检无遮挡） | 直接插入报告与 PPT |
| `course_project/results/` | `main_table.csv`、`ablation.csv`、`control_plane.csv`、`per_seed_metrics.csv`、`stats.json`、`stability.json`、`cicids_demo.json` | 原始汇总数据（老师查验真伪凭证） |
| `course_project/prompts/` | `node_agent.md`、`coordinator_agent.md`、`FROZEN_MANIFEST.json` | 冻结提示词与 SHA256 证据 |
| `course_project/src/` & `tests/` | 完整可运行源码与 9 项自动化单元测试 | 源码提交物 |
| `course_project/runs/` & `transcripts/` | 全部 13 个评测/干净场景 + 3 个开发场景 + 9 次稳定性重复的完整计划与脱敏 LLM JSONL 审计日志 | 实验全量留痕（体积较大，可按需压缩附交） |

---

## 二、按不同课程字数/页数要求的精简策略

1. **若要求 3,000–4,000 字紧凑版课程报告**：
   - 保留 `report.md` 的摘要 + 第 1 章（精简为 2 段）+ 第 2 章（保留对比表与“三句话”+ 负结果 1 段）+ 第 3 章（`fig1` + 工具/隔离简介）+ 第 5 章（主表 + 统计结论 + 控制面表 + `fig2/fig3/fig4`）+ 第 6 章（CICIDS 1 段 + `fig5`）+ 第 7 章（3 条局限精简版）。
   - 将第 4 章的详细特征列池与附录复现命令折叠为简表。
2. **若答辩时间压缩至 5 分钟**：
   - 使用 `ppt_outline.md` 中的 **P1、P3（负结果）、P4（架构图）、P7（主结果）、P8（控制面+悖论）、P11（CICIDS 亮点）、P12（诚实局限）** 共 7 页核心片，节奏稳准狠。
3. **转 Word / LaTeX / PDF 技巧**：
   - 用 Pandoc 或 Typora 直接导出 `course_project/report/report.md`，插图相对路径 `../figures/fig*.png` 已配置好，可直接渲染无需改动路径。

---

## 三、一键复现与自检命令

在项目根目录（`网安实训选题/`）下运行：

```powershell
# 1. 运行全部 9 项自动化单元测试（隔离 + 执行器 + 提示词防泄漏，无需 API 费用，约 11 秒）
python -m pytest -v course_project/tests/

# 2. 仅重新汇总已有结果并重新生成 5 张高清图表（无需 API 费用，约 3 秒）
python -u -m course_project.src.run_experiments --stage p6_aggregate --seeds 100-109 --clean-seeds 200-202
python -m course_project.src.make_figures

# 3. 验证提示词 SHA256 与冻结清单一致
Get-FileHash course_project/prompts/node_agent.md, course_project/prompts/coordinator_agent.md -Algorithm SHA256
```

---

## 四、交稿前安全检查清单（务必执行！）

- [ ] **绝不打包根目录的两个 API Key txt 文件**！仅打包 `course_project/` 目录提交。
- [ ] 提交前在 `course_project/` 下执行秘密扫描确认零泄露：
  ```powershell
  Select-String -Path (Get-ChildItem -Recurse -File course_project) -Pattern "sk-[A-Za-z0-9_\-]{8,}"
  ```
  （预期输出为空，0 命中）。
- [ ] 确认 `data/`、`legacy/`、`v2_research/` 未被改动（如不需要提交大体积原始数据集，只交 `course_project/` 并在 README 中注明数据集放置路径即可）。
