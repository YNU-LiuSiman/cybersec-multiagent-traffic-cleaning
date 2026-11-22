# 多智能体协商高光实录（真实 Transcript 精选）

> 本实录全部摘自冻结提示词后的真实运行落盘日志（`course_project/runs/` 与 `course_project/transcripts/`），所有数值与工具调用链均可逐字溯源复核。

---

## 高光案例一：UNSW-NB15 盲测场景 `seed 102` —— 四重并发故障下的跨节点三角印证与追问破案

### 场景盲测真值（仅评测脚本可见，Agent 全程不可读）
- **随机节点置换**：`node_1` = Role D（长尾 Rare-4 困难负例，零故障）；`node_2` = Role C（`31.09%` 标签噪声 + 单节点作弊列 `flow_audit_tag`）；`node_3` = Role A（健康基准）；`node_4` = Role B（**`1/8 = 0.125` 比特/字节量纲漂移**作用于 `sbytes, dbytes, sload, dload` + **3 个非正则语义别名** + 288 条脏行）。

### Round 0：四个 Node Agent 在严格隐私沙箱中独立取证（彼此不可见）
- **`node_1` Agent** 调用 6 个本地工具后如实上报：行健康零脏行，物理守恒比值 `1.000`（违背率 `0.0`），但本地存在大量极稀有类别（`Worms=44`, `Shellcode=378`, `Backdoor=583`, `Analysis=677`），列 `flow_audit_tag` 本地 `mi_norm = 0.0046`。
- **`node_2` Agent** 上报：行健康与物理守恒均正常，但发现 `flow_audit_tag` 具有异常高的本地互信息 `mi_norm = 1.0002`——严格遵循职责边界，仅标记“单节点无法判断这是强物理特征还是局部捷径，交由 Coordinator 跨节点比对”。
- **`node_4` Agent** 上报：发现 3 个未映射列（`handshake_roundtrip_val`、`http_reply_payload_sz`、`recent_dst_sport_fanout`）与 3 个缺失规范列（`tcprtt`、`response_body_len`、`ct_dst_sport_ltm`），`row_health` 检出 288 条脏行（144 负值 + 144 NaN），且两条主物理守恒比值均为 **`0.125`**（违背率 `100%`）。

### Round 1：Coordinator 跨节点比对 + 关键多轮追问（揭开被捷径掩盖的标签噪声！）
1. **跨节点 MI 狩猎（`compare_mi`）**：Coordinator 一眼锁定 `flow_audit_tag` 在 `node_2` 上 `mi_norm = 1.0002`，而在 `node_1 / node_3 / node_4` 上仅为 `0.0046 / 0.0013 / 0.0034`（`robust_z = 49.81`）——确认其为单节点标签捷径！
2. **全局稀缺直方图（`aggregate_histograms`）**：仅凭计数汇总识别全局稀缺类 `["Analysis", "Backdoor", "Shellcode", "Worms"]`（100% 集中于 `node_1`）。
3. **关键追问动作（`query_node` × 4）**：Coordinator 意识到“如果不先剔除作弊列，本地置信学习会被作弊列骗过”，立即并行向 4 个节点发出追问指令：
   ```json
   query_node(node="node_1..4", tool="confident_learning",
              args={"exclude_cols": ["flow_audit_tag"],
                    "exempt_classes": ["Analysis", "Backdoor", "Shellcode", "Worms"]})
   ```
   各节点在沙箱内执行并返回**脱敏统计摘要（零行索引出域）**：
   - `node_3`（健康基准）：`suspected_noise_rate = 0.0445`, `mean_self_confidence = 0.8190`, `cv_macro_f1 = 0.7189`
   - `node_4`（量纲漂移但标签干净）：`suspected_noise_rate = 0.0399`, `mean_self_confidence = 0.8166`, `cv_macro_f1 = 0.7121`
   - `node_1`（困难负例 Rare-4）：`suspected_noise_rate = 0.1086`, `mean_self_confidence = 0.6457`, `cv_macro_f1 = 0.5245`
   - **`node_2`（31.1% 真标签噪声暴露！）**：`suspected_noise_rate = 0.2258`, `high_conf_conflict_rate = 0.1133`, **`mean_self_confidence = 0.4082`**, `cv_macro_f1 = 0.4394`！

### 最终签发的结构化 Remediation Plan（节选真实 JSON 证据）
- **`drop_column("flow_audit_tag")`** —— 依据：`node_2 mi_norm=1.0002 vs peers 0.0013–0.0046 (robust_z=49.81)`。
- **`local_cl_clean("node_2", exempt_classes=["Analysis","Backdoor","Shellcode","Worms"])`** —— 依据：剔除捷径后噪声率 `0.2258`、自置信度骤降至 `0.4082`（同伴 `0.817–0.819`）。
- **`rename_columns("node_4", {"handshake_roundtrip_val": "tcprtt", "http_reply_payload_sz": "response_body_len", "recent_dst_sport_fanout": "ct_dst_sport_ltm"})`** —— 依据：语义 + Benign 锚点分位数精准匹配（如 `handshake_roundtrip_val` `p50=0.000706, zero_frac=0.2957` 精确对应 `tcprtt` 参考值 `0.000703, 0.2983`），且正确忽略跨节点共享的 `flow_audit_tag`。
- **`drop_rows("node_4", rule="nan_or_negative_physical")`** —— 依据：288 条脏行（144 负值 + 144 NaN）。
- **`rescale("node_4", cols=["sbytes","dbytes","sload","dload"], factor=8.0)`** —— 依据：主守恒比值精确为 `0.125 (1/8)`，Benign 锚点 `dbytes 44.25 vs 354 (0.125)`，判定为比特/字节换算误差，精准恢复 `1 / 0.125 = 8.0`。
- **`no_op("node_1")` & `no_op("node_3")`** —— 明确保护 `node_1`（全部 Rare-4 宿节点，主守恒 `1.0`，Benign 锚点误差 `< 0.03 log10`）。

**执行结果**：控制面全项 `1.0000`（Noise Sample F1=`0.7881`，Node D 零误处置），RF Macro-F1 从 `Raw-Merge` `0.4845` / `Static-Pipeline` `0.4168` 跃升至 **`0.5202`**（与真值上界 `Oracle-Clean` `0.5209` 仅差 `0.0007`）！

---

## 高光案例二：CICIDS-2017 零改动泛化 —— 精准放行 77.4% DDoS 突发与极不平衡分片 `slice_8`

在不改动冻结提示词一字的前提下，面对 9 个真实时序分片（`slice_1..8` 周五 DDoS + `slice_9` 跨天周二 Benign，仅 `slice_2` 注入 `1000×` 量纲漂移、`slice_5` 注入 `25%` 标签翻转）：

- **`Static-Pipeline` 惨遭连环误报**：把 `slice_1, 3, 7, 8, 9` 全部误标为故障（`FPR = 71.4%`），例如在 77.4% DDoS 高峰 `slice_3` 上因整表 `totlen_bwd_pkts` 偏移 `65.9×` 直接触发误缩放！
- **`Ours (Full)` 冻结提示词原声推理高光（摘自 `results/cicids_demo.json`）**：
  > *"`slice_2` shows a clean x1000 unit drift on `totlen_fwd_pkts` (benign p50=64000 vs canonical 64; conservation `median_ratio_lhs_over_rhs=1000.0`, `violation_rate=1.0`)... `slice_5` shows localized label noise (`suspected_noise_rate=0.1632`, `mean_self_confidence=0.619`, `cv_macro_f1=0.7029`) versus a peer baseline of `~0.0002` noise and `~0.999` confidence... **`slice_8` (17 DDOS rows, `cv_macro_f1=0.4999` but `0.0002` noise, `0.9997` confidence)** and `slice_9` (BENIGN-only) are legitimate Non-IID class-mixture effects, not faults."*

**亮点解析**：`slice_8` 仅有 17 条 DDoS 与 28,202 条正常流（`0.06%`），导致 `cv_macro_f1` 跌至 `0.4999`（任何静态 F1 阈值都会直接误杀）；但 Coordinator 联合 `suspected_noise_rate = 0.0002`、`mean_self_confidence = 0.9997` 与 `conservation = 1.0`，瞬间看穿这是极端类别不平衡而非标签污染，果断签发 `no_op`——最终取得 **`0.0% FPR (0/7)` 与 `100.0% TPR (2/2)`**！
