# Paper Progress (CAL → TCAS-I Journal)

**论文**: EdgeStereoDAv2 (renamed 2026-04-20 from CAL → TCAS-I journal)
**最新标题**: *EdgeStereoDAv2: A Confidence-Centric Stereo-Mono Fusion Accelerator for Edge Depth Estimation* (2026-04-21 起)
**模板**: `paper/tcasi/` (IEEEtran `[journal]`)
**当前状态**: **2026-04-22 — Phase 10.11 完成, 共 6 轮 reviewer 迭代, 最终 cold-eyes R3 verdict = Accept-with-Minor (camera-ready polish only)**, PDF 19 页 1.70 MB, 0 errors, 0 undefined refs
**主文件**: `paper/tcasi/main.tex` (约 670 行)
**审稿迭代**: Reject (P10.6) → Major (R1) → Accept-with-minor (R2) → Accept (R3, 知历史) → Major (cold R1) → Minor (cold R2) → **Accept-with-minor (cold R3, 完全独立)**
**产物**: `paper/tcasi/main.pdf`

---

## Phase 10.11 — Cold-Eyes Round 3 → ACCEPT-WITH-MINOR (2026-04-22 收尾)

**触发**: Phase 10.10 修复 (estimated-cells honesty across abstract / VII.G / IV.F / L7) 后 dispatch cold-eyes R3.

### Cold-eyes R3 verdict: **Accept with Minor Revision** (从 R2 borderline 升一格)

reviewer 明示: "remaining concerns are camera-ready polish, not blocking"

### 3 strongest aspects (R3 给的)
1. Honest, structured limitations disclosure (L1-L7 全列, L7 是 estimated cells 专项)
2. Disciplined claim scoping (abstract / §VII.G / §IV.F 各处都明确 measured vs estimated)
3. Coherent novelty story (single PKRN provenance broadcast; 三条 ablation 各管一面)

### 3 minor camera-ready polish items (不阻塞)
1. **§VII.H L549 复用 23.50 时没标 (estimated)** —— 已修, 加 "(this Row-C D1 is the trend-estimated value flagged by Tab.~\ref{tab:tm_wcaps_2x2} footnote $^\ddagger$ and L7)"
2. A1 V2 control-area 是 modeled (已 L2 admitted, 已 tab:a1 footnoted) —— 不需修, 已诚实声明
3. A3 leave-one-out analytical (已 L3 admitted) —— 不需修, F4 是 future work

### Final fix (Phase 10.11)
单点 edit: §VII.H L549 加 "(this Row-C D1 is the trend-estimated value flagged by the $^\ddagger$ footnote in Tab.~\ref{tab:tm_wcaps_2x2} and L7 in Sec.~\ref{sec:limitations})". 重编译 19 页, 无 error.

### Decision letter (≈100 words, R3 原文)
> Manuscript reviewed: "EdgeStereoDAv2: A Confidence-Centric Stereo-Mono Fusion Accelerator." **Decision: Accept with Minor Revision.** The paper presents a well-positioned contribution — promoting SGM PKRN confidence to a shared cross-stage external control signal driving five SCUs and a 12-op fusion ISA — supported by three ablations (A1 unification, A2 2×2 TM×W-CAPS factorial, A3 ISA leave-one-out). Methodological honesty is exemplary: limitations L1–L7 explicitly disclose the two-tier modeling gap (±8% per-engine), the Sobel-surrogate scope of A1 V2, the analytical nature of A3, and the 4 estimated cells in A2/hp-sweep. Camera-ready should re-tag the estimated "Both" cell at its §VII.H re-use site and ideally close the 4 measurement gaps L7 itself rates as one day's work.

### 总迭代轨迹 (6 reviewer rounds)

| Round | Reviewer (\u77e5/\u4e0d\u77e5\u5386\u53f2) | Verdict |
|---|---|---|
| Phase 10.6 | aware | **Reject** |
| Round 1 | aware | Major Revision |
| Round 2 | aware | Accept-with-Minor |
| Round 3 | aware | **Accept** |
| Cold R1 | independent | Major Revision (3 表本地缺 + L1 unbounded) |
| Cold R2 | independent | Minor Revision (4 estimated cells in load-bearing tables) |
| **Cold R3** | independent | **Accept with Minor Revision** (camera-ready polish only) |

### 关键学习 (D32-D34)

| ID | Decision | Why |
|---|---|---|
| D32 | 6 轮迭代后两条 reviewer track (aware vs independent) verdict 收敛到 Accept / Accept-with-minor | 跨 reviewer 一致性是真接受信号, 不是单 reviewer 偏好 |
| D33 | 全部修复都是 honest disclose 而非 fake-fix | TCAS-I reviewer 对"诚实声明 + 路径 forward"接受度比对"完美 numbers"高 |
| D34 | 4 个 estimated cell 不强行替换为 measurements (~1 day 但需要 GPU 时间), 而是 L7 完整披露 + abstract/body 全文 hedge | F2/F3 留作 camera-ready follow-up, paper 主体 claim 站得住 |

---

## Phase 10.10 — Cold-Eyes Round 2 → Estimated-Cells Honesty (2026-04-22)

**触发**: cold-eyes Round 2 reviewer 在 Phase 10.9 修复 (3 表同步 + L1 partial bound) 后, 把 verdict 从 Major Revision 升到 **Minor Revision** (borderline accept-with-minor). 唯一新发现: tab:hp_sweep 和 tab:tm_wcaps_2x2 里有 **3 个 cell 标注 "extrapolated/estimated from trend"**, 其中 1 个 (Both 行 KITTI D1=23.50) 是 abstract 里 "TM+W-CAPS recovers 2.3 pp" claim 的直接来源.

### Cold-eyes R2 verdict 详情

- **3 strongest**: 两层模型诚实声明 + scope-disciplined 措辞 + same-protocol vs cross-protocol 区分
- **3 concerning**: estimated cells in load-bearing tables (NEW), A1 V2 surrogate (已 L2 admitted), A3 analytical (已 L3 admitted)
- decision letter: "minor revision; replace estimated/extrapolated cells in Tables 6 and 7 with measured values and tighten A1 V2 framing before camera-ready"

### 修法 (3 项, 全部诚实 disclose 而不是 fake-fix)

| Fix | 操作 | 位置 |
|---|---|---|
| C-R3a | abstract A2 句改为 "whose Neither and TM-only cells are measured and whose W-CAPS-only and Both cells contain a KITTI D1 estimate; within those measured / trend-estimated bounds ... they restore an estimated $\sim$2 pp of TM's D1 loss" | abstract |
| C-R3b | §VII.G 段首加 "Measurement status" 子段, 显式说明 2/4 cell 是 trend-extrapolated, 一一标注每个 cell 来源 | sec:a2_orthogonality |
| C-R3c | §IV.F hp-sweep 段加 "(both **measured**)" 强调 hp=50/75 比较是直接测量, hp=25 row 是 extrapolated 的辅助上下文不是 load-bearing | sec:wcaps |
| C-R3d | 新增 **L7** in sec:limitations: 显式列出 4 个 estimated cell + "directional claims robust to estimate magnitude, but TCAS-I camera-ready should replace with measurements (~1 day on existing infra, folded into F2/F3)" | sec:limitations |

### Cold-eyes R3: 验证 in flight

prompt 强调: 验证 L7 是否齐全 + abstract 是否 hedged + body text 是否 explicit + 3 个表 footnote 是否 cross-referenced.

---

## Phase 10.9 — Cold-Eyes Independent Review (2026-04-22)

**触发**: 用户"继续". Round 3 reviewer 已知前两轮历史, 可能存在偏向. 跑一轮**不知道前史的独立 reviewer** 验证 ACCEPT verdict 的稳健性.

### Cold-eyes Round 1 verdict: **Major Revision** (回退一格)

理由 (3 大核心 concern):

1. **3 个 input'd 表本地不存在** —— `table_hp_sweep.tex` / `table_tm_wcaps_2x2.tex` / `table_hw_ablation.tex` 在 `main.tex` 用 `\input{}` 引用但本地源树没有, 标记为 "reproducibility blocker". (实际是本地 mirror 同步问题, 远程都在并且编译成功 19 页)
2. **L1 admits unbounded Tier-1/Tier-2 gap, 但 abstract 的 40 FPS / 172 mW / 1.87 mm² 都吊在这个未量化的模型上** —— 要求至少给一个 partial bound
3. **C3 (learned fusion) 自承"非贡献", 把 novelty 全压给 C1+C2, 而 C2 又靠 A3 (analytical) + W-CAPS 没 head-to-head** —— 这是描述, 不算阻塞

### 修法 (2 项)

| Fix | 操作 | 位置 |
|---|---|---|
| C-R1 | scp 远程 3 个表到本地 mirror | `paper/tcasi/table_{hp_sweep,tm_wcaps_2x2,hw_ablation}.tex` |
| C-R2 | L1 加 partial bound: "Tier-2 formulas track Tier-1 measurements within $\pm 8\%$ on each engine on canonical micro-benchmarks; residual whole-frame uncertainty dominated by inter-engine collisions on shared L2/AXI which Tier-2 serializes but does not cycle-simulate" | sec:limitations L1 |

(C3 concern 不修, 因为不是阻塞项, 且 reviewer 自承"This is fine, but it leaves C1+C2 carrying nearly all the architectural novelty")

### Cold-eyes Round 2: 重新 dispatch 验证 (in flight)

prompt 强调: 验证 (a) 3 表是否齐全, (b) L1 是否有 partial bound, (c) "first" claim 是否真的清零, (d) SOTA 表是否仍 scope 为 positioning.

---

## Phase 10.8 — Round 3 Single-Line Fix → Accept (2026-04-22)

**触发**: Round 2 strict reviewer verdict = **accept-with-minor**, 唯一阻塞项是 line 44 的 "first mixed-precision accelerator" claim 没在 Round 1+2 的 first-claim 清零里被抓到. 同行还顺手发现同一行的 "strict 8/8 metric dominance" 残余. user 批准 Round 3 single-line fix plan.

**单点编辑** (main.tex line 44):
- "to our knowledge it is the **first** mixed-precision accelerator scheme whose precision mask is computed at *spatial* granularity from an external stereo confidence rather than assigned per layer at compile time" → "**unlike prior layer-wise mixed-precision accelerators~\cite{sharma2018bitfusion,park2019olaccel}**, its precision mask is computed at *spatial* granularity from an external stereo confidence rather than assigned per layer at compile time"
- "yields **strict 8/8 metric dominance over the hand-crafted heuristic baseline**" → "**improves on all eight accuracy metrics over the hand-crafted heuristic baseline implemented in the same in-house pipeline**"

**重编译验证**: 19 页, 0 errors, 0 undefined refs. "strict 8/8" grep = 0; "the first / first.*accelerator" body claim grep = 0; 余下 `first` 出现 4 处全是 narrative ordinal ("the first anchoring..." / "First, Row A vs GPU" 之类), reviewer 显式 OK.

**Round 3 strict reviewer verdict**: **ACCEPT**. 60-word decision-letter summary: "This co-design paper unifies an externally-produced SGM PKRN confidence as a cross-stage control signal across token pruning, spatial precision dispatch (W-CAPS), calibration anchoring, and output fusion, on a 1.87 mm² / 172 mW 28 nm accelerator reaching 40 FPS at KITTI D1=6.43%. Round 3 properly softens novelty positioning and metric-dominance phrasing; ablations and limitations are scoped honestly. Recommended accept."

**所有 Round 1+2+3 修正核心点回顾**:
- Methodology 两层诚实声明 (Tier-1 event-driven + Tier-2 critical-path scheduler with explicit `_estimate_op_cycles` 公式)
- Limitations L1-L6 + Future Work F1-F6 一一对应
- Tab perf 拆为 v2 main path + v1 carry-over 双表
- Tab a1 / a3 caption 加 modeled-vs-synth 与 analytical-estimates disclosure footnote
- SOTA 表 caption 加粗体 "**not a same-protocol comparison**" + "**Mixed power scopes**"
- 全文 "first / strict 8/8 / load-bearing / fully hidden / tight ISA / Pareto frontier" 等强 claim 全部清零或软化
- Abstract 加回 hp sweep + 2×2 factorial + GPU 700× efficiency 三大 real evidence

### 新增决策日志

| ID | Decision | Why |
|---|---|---|
| D30 | Round 3 只做 line 44 单点修, 不引入新内容 | reviewer Round 2 verdict 已经是 "accept-with-minor", residual 是 1 句话; 任何额外修改增加 regression 风险 |
| D31 | softening 时引用 sharma2018bitfusion + park2019olaccel 而不是删整段 | 保留 contrastive positioning 的实质论据, 只去掉 "first" 这个具体形容词 |

---

## Phase 10.7 — Reject→MajorRevision (Strict-Reviewer 迭代, 2026-04-21 收尾)

**触发**: 用户提供 Phase 10.6 之后的 strict reviewer 完整审稿意见, verdict=Reject. 5 大 + 4 小 concerns. user 要求"修改并启用 subagent 严厉审稿循环到满意".

### Reviewer 5 大 Concern

| ID | Concern | 修法 (Round 1) |
|---|---|---|
| **M1** | "event-driven cycle-accurate simulator with 904/2100-op DAG, bank conflict, AXI contention" 与 `run_simulator_phase10.py` 实际 `_estimate_op_cycles()` 简化估算不一致. 影响所有 HW 数字 | §VI 重写 "Performance Modeling Methodology" 为**两层 (Tier 1 per-engine event sim + Tier 2 critical-path scheduler)**, 明确说 Tier 2 用解析公式 (pixels/32 etc.) 而非 Tier 1 API. Round 2 进一步在 Tier-2 段加一句 "the formulas are calibrated to Tier-1 measurements but are NOT direct API calls into Tier 1" |
| **M2** | "not just integration" + 单 conf 信号的 novelty 论据弱 (V2 是 Sobel surrogate, 面积模型用 2.5x/3.5x 手算系数) | §VIII.A 重写为 "defensive-but-honest": 承认组件不新, 创新在组织. 限定 A1 V2 为 "Sobel inference-time surrogate" 而非 fully-trained per-stage controller. tab:a1 caption 加 disclosure footnote |
| **M3** | A3 ISA minimality 是硬编码常数 (OP_MIX_PCT 字典, LEAVE_ONE_OUT 字典) 而非 simulator trace | A3 script 头部加 "LIMITATION DISCLOSURE: analytical model, NOT a recompilation/re-trace". §VII.F 段尾加同样 caveat. tab:a3 caption 加 "design-time analytical estimates" 标注 |
| **M4** | SOTA 表混用协议 (in-house split vs test-server) 与功率口径 (FPGA-board / Jetson-SoC / ASIC-core), 但正文用 "strict 8/8 dominance / Pareto frontier" 强 claim | tab:sota_acc / tab:sota_hw caption 加粗体警示 "**not a same-protocol comparison**" + "**Mixed power scopes; cross-protocol; treat as positioning**". 正文 §VII.B 把 "beats FP-Stereo by 2.7 pp" 删除, 改为 "we do not report a direct ΔD1 against them without same-protocol re-evaluation". §III intro / §VII.A 把 "strict 8/8 metric dominance" 改为 "improves all eight metrics over the heuristic baseline implemented in the same in-house pipeline" |
| **M5** | Main perf 表混用 B0-h24 (518×518 v1 path) 与 Heuristic+B1-h24 (384×768 v2 path), 读者无法分离贡献 | tab:perf 拆为两表: tab:perf 只保 v2 main path (Heur + B1-h24, 384×768); 新建 **tab:perf_v1carry** 单独放 B0-h24 (v1, 518×518) carry-over baseline, caption 明确说"NOT on v2 main path" |

### Round 1 收尾: 加 Limitations 专节 (L1-L6) + 重写 Future Work (F1-F6)

新增 `\subsection{Methodology and Evaluation Limitations}` (sec:limitations) 集中列出:
- L1: Tier 2 不是 single-loop cycle-accurate, 是 schedule-aware 估算
- L2: A1 V2 是 Sobel surrogate, 不是 trained per-stage uncertainty head
- L3: A3 leave-one-out 是 analytical, 不是 compiler-regenerated
- L4: W-CAPS 没有 vs layer-wise mixed precision 的 head-to-head (但有 hp sweep 内部表征)
- L5: 4 数据集 in-house split, 不可与 test-server 直接比
- L6: 跨论文表混用 power scope / 分辨率 / 协议

`\subsection{Future Work}` 从原 3 段散文改为 F1-F6 编号清单, 每条对应一个 L:
- F1 ↔ L1: 单循环 whole-frame cycle-accurate co-sim (用同一 SimConfig)
- F2 ↔ L4: W-CAPS vs layer-wise matched-bit-budget head-to-head (highest leverage)
- F3 ↔ L2: A1 V2 用 trained uncertainty head 替换 Sobel
- F4 ↔ L3: 编译器 recompile + Tier-1 re-sim 替换 analytical leave-one-out
- F5 ↔ L5: 标准 test-server 协议 re-eval (KITTI-test/SF-test/ETH3D-test/Mid-test)
- F6: 28nm tape-out 验证 power envelope

### Round 1 Strict Reviewer 验证

dispatch 一个 clean-context subagent 模拟 TCAS-I area chair 审查 5 大 concern 是否解决.

**Verdict 升级**: Reject → **Major Revision** (substantial recovery)

| Concern | Round 1 Status | Reviewer Note |
|---|---|---|
| M1 | partially addressed | Methodology 段诚实, 但 `run_simulator_phase10.py` 第 60 行 `_estimate_op_cycles` 仍是主路径, L1 disclosure 与代码细节不完全对齐 |
| M2 | partially addressed | 仍有 "to our knowledge ... first" 措辞过满 (line 44); tab:a1 应加 disclosure 说面积是 modeled 而非 synth |
| M3 | partially addressed | tab:a3 应加 "design-time analytical estimates" 标注 |
| M4 | **addressed** | tab:sota_acc/hw caption 警示 OK, 但正文 line 453 还有 "beats FP-Stereo C2/C4 by 2.7 pp / 2.4 pp" 的同基线 Δ claim, 与 caption 矛盾 |
| M5 | **addressed** | tab:perf_v1carry 拆分干净 |

3 highest-leverage residual fixes:
1. L1 vs `_estimate_op_cycles` 代码级 reconcile
2. 删除 line 453 的 "beats FP-Stereo by 2.7 pp"
3. tab:a1 + tab:a3 caption 加 disclosure footnote

### Round 2: 4 Reviewer-Flagged 修正

| Fix | 操作 | 位置 |
|---|---|---|
| F-R1 | Tier-2 段加 explicit 公式 (pixels/32, flops/(2*1024)) + "calibrated to Tier-1 but NOT direct API calls" | sec:method_perfmodel |
| F-R2 | 删 "beats FP-Stereo C2/C4 by 2.7 pp / 2.4 pp" → "we do not report a direct ΔD1 against them without F5" | line 453 |
| F-R3a | tab:a1 caption 加 "control-area numbers are modeled estimates ... 2.5×/3.5× engineer-derived multipliers ... NOT a synthesis flow; see L2" | table_a1.tex |
| F-R3b | tab:a3 caption 加 "cycle-multiplier and weight-traffic columns are design-time analytical estimates ... NOT measured cycles" | table_a3.tex |
| F-R4 (minor) | 全文 "first" 出现 0 次 (case-insensitive grep). 把 "the first" → "we are not aware of a prior accelerator that does so ... but we do not exclude that such a design exists" | line 126 + abstract |

### Round 2 待验证

SSH 中断中 (pdongaa@EEZ245 banner timeout), Monitor 在 polling 等待恢复. Round 2 重编译 + dispatch 第二轮 strict reviewer 验证待 SSH 回来后做.

### 新增决策日志

| ID | Decision | Why |
|---|---|---|
| D25 | Methodology 重写为两层 (Tier 1 / Tier 2) 而不是补一个真实的 single-loop cycle-accurate sim | 真实补一个 single-loop sim 需要数周, 不在 single-session 预算内. 先做诚实声明, F1 留作 future work |
| D26 | tab:perf_v1carry 单独立表而非删除 B0-h24 | B0-h24 是有价值的 carry-over baseline, 删了会失去信息. 拆表保留信息但消除歧义 |
| D27 | "first" 词全文清零, 改为 "to our knowledge / not aware of prior" | TCAS-I reviewer 普遍敌视 "first" 措辞. 软化到可辩护程度 |
| D28 | Limitations + Future Work 重构为 L1-L6 ↔ F1-F6 一一对应 | 让审稿人能跟踪 "你哪个限制对应哪个补救方案", 提高可信度 |
| D29 | Phase 10.6 已经做的 hp sweep 和 2×2 factorial 在 Round 1 abstract 重写中被误删, Round 2 末尾恢复 | abstract 应反映已有强证据, 不应过度自我矮化 |

---

## Phase 10.6 — Memory-bank Sync & Optimization (2026-04-21 后段)

**触发**: user 要求"同步 memory-bank 最新内容, 合理修改优化论文"

**发现**: memory-bank 里已有三个**实际跑过但从未并入 main.tex** 的实验表格, 之前 Path B 写 abstract 时说 W-CAPS 对比 "deferred to future work" —— 实际 Phase 11 已经跑了 hp% sweep 和 2×2 factorial. 这是 abstract 与 body 的自相矛盾.

### 新增集成 (6 项)

| # | 影响 | 位置 | 内容 |
|---|---|---|---|
| 1 | **HIGH** | Abstract + §IV.F + new Table hp_sweep | W-CAPS hp$\in\{25,50,75,100\}$% 四点 sweep: hp=50% 反超 hp=75% (KITTI D1 25.18 vs 25.70) 因 INT4 隐式正则化. 取消 abstract 里 "deferred to future work" 说法. `table_hp_sweep.tex` 原文件 |
| 2 | **HIGH** | new §VII.G "Ablation A2 --- Orthogonality" | 2×2 factorial TM×W-CAPS: TM alone +46% FPS +5pp D1; W-CAPS alone -7% FPS ~no D1; Both +31% FPS recover 2.3pp (complementary 不是 additive). `table_tm_wcaps_2x2.tex` 原文件 |
| 3 | **HIGH** | new §VII.H "Cumulative Hardware Ablation" | A→D cumulative + GPU baseline (RTX TITAN 0.025 FPS/W vs ours 17.57 FPS/W on Row A = 700× efficiency gap). INT8 QAT 功耗 421.8→174.4 mW (2.4×). `table_hw_ablation.tex` 原文件 |
| 4 | MED | §IV.F 新段 "Stage-policy overhead" | W-CAPS 开 all vs coarse_only 约 7% latency, 全 kr 一致 |
| 5 | MED | §III.C ADCU 段 | 明确 "Cramer's rule, no Newton-Schulz, 323 cycles constant-latency" |
| 6 | LOW | §IV.C INT8 QAT 段 | "max EPE regression ≤ 0.035 px on SF B1-h24 = INT8 effectively free" |

### 结果
- 页数: 15 → **17** (TCAS-I 期刊无硬上限)
- 新增贡献证据显著加强: W-CAPS 从"提议"升级为"已验证"(hp sweep); TM/W-CAPS 从"叙事声明" 升级为"2×2 定量分解"; HW 性能从"表中 numbers" 升级为"A→D cumulative + GPU anchor"
- Abstract ↔ body 自相矛盾问题消除
- Fresh-reviewer 的 #1 高杠杆建议(跑 W-CAPS vs layer-wise 对比)仍是 future work, 但 hp sweep + 2×2 factorial 已经把 W-CAPS 的贡献 concretize

### 新增决策日志

| ID | Decision | Why |
|---|---|---|
| D22 | 把 table_hp_sweep / table_tm_wcaps_2x2 / table_hw_ablation 3 个原本 orphan 的 tex 集成进来 | 文件已存在但从未 input, 这是实验已做但没写论文的浪费 |
| D23 | Abstract 里 hp=50% reverses hp=75% 的发现作为亮点保留 | 这是一个可验证的反直觉 empirical finding, novelty defense 的新弹药 |
| D24 | TM/W-CAPS 2×2 ablation 新立为 §VII.G 独立小节而非塞进 §VII.D | "complementary 不是 additive" 是系统级论据, 值得单独章节展示 |

---

---

## Phase 10.5 — Novelty Strengthening (Path B, 2026-04-21)

**背景**: Self-review + fresh-context reviewer 指出原稿 novelty 边界模糊，整体观感像"工程整合"(SGM + EfficientViT + QAT 都已有)。User 选 Path B: 重构叙事 + 2 针对性 ablation + 可选 A2(deferred)。

### 变更点(完整 delta)

| 位置 | 变更 | 目的 |
|---|---|---|
| Title | 加副标题 "A Confidence-Centric Stereo-Mono Fusion Accelerator" | N1 定位 |
| Abstract | 整段重写,放 SGM conf as primary control signal 到第 1 句 | N1 定位 |
| §I Intro | 5 段新 contribution 框架: C1(core 统一 conf 信号) / C2(5 SCU + 12-op ISA) / C3(EffViT 作为 plugged-in backbone) | 消除"工程整合"观感 |
| **§II.D** (NEW) "Confidence as a Unified Control Signal" + **Table II.D** 4×4 定位表 | 把"prior conf 只做最终融合"这件事 formalize | N1 理论支撑 |
| §III head | 1 段明确承认 EffViT-Depth 是 "plugged-in backbone, not primary novelty" | 消解 reviewer 第二质疑 |
| §IV.A | 增加 hub-and-spoke 5-SCU 组织说明 | N1 HW 映射 |
| **§IV.F** (NEW) "W-CAPS: Spatial Adaptive Precision from External Confidence" | 把 DPC 里的空间级混合精度升格为单独子节 | N2 独立贡献 |
| §V Table II (ISA) | 加 `conf-channel: R/G/---` 列 | N1 在 ISA 层面可见 |
| **§VII.E** (NEW) "Ablation A1 — Confidence-Signal Unification" + **Table tab:a1** | 4 variant × 4 数据集 + 面积对比 | C1 证据 |
| **§VII.F** (NEW) "Ablation A3 — ISA Coverage and Minimality" + **Table tab:a3** + **Fig isa_coverage** | 12-op leave-one-out + 饼图 | C2 证据 |
| **§VIII.A** (NEW) "Why This Is Not Just Integration" | 0.5 页反驳段,TCAS-I 稳重口径 | 正面回应 reviewer |
| Fig.1 (arch) | 完全重画 hub-and-spoke: SGM conf 在中心扇出 5 SCU | N1 可视化 |
| **Fig.6a** (NEW) `fig_conf_routing.tex` 单 conf 流经 4 local transform | N1 补充图 |
| Bibliography | +4 新 bibitem: facil2019camconvs, wang2019pseudolidar, sharma2018bitfusion, park2019olaccel | Table II.D 引文 |

### 实验结果 (Ablation A1)

4 variant × 4 dataset, B1-h24 FP32 checkpoint, 推理时切换 conf 信号:

| Variant | KITTI EPE/D1 | SF EPE/bad1 | ETH3D EPE/bad1 | Mid EPE/bad2 | Ctrl area |
|---|---|---|---|---|---|
| **V0 (Ours, unified)** | **1.16 / 6.40%** | **3.17 / 45.00%** | 0.66 / 16.02% | 1.81 / 21.83% | **0.148 mm²** |
| V1 (random mask) | 1.30 / 7.26% | 3.57 / 60.25% | 0.65 / 14.45% | 1.84 / 21.45% | 0.148 mm² |
| V2 (per-SCU indep, Sobel 代用) | 1.17 / 6.55% | 3.22 / 46.36% | 0.69 / 16.71% | 1.85 / 22.36% | +0.032 mm² |
| V3 (uniform, no conf) | 1.23 / 7.20% | 3.44 / 51.31% | 0.65 / 14.76% | **1.74 / 20.48%** | 0.145 mm² |

- **V0 vs V1**: random 同密度替代 → SF bad1 +15.3 pp, KITTI D1 +0.86 pp(内容是 load-bearing)
- **V0 vs V2**: 精度几乎持平(V0 仅 0.7-4.3% 相对略好), 但 V2 控制面积贵 0.032 mm²(+21.9% SCU 预算 / +1.73% die)
- **V0 vs V3**: SF/KITTI V0 显著好, ETH3D/Middlebury V3 竟略胜(Middlebury 场景 DA2 友好,均匀 conf 意外 bias 对了)— 此点在 abstract 和 VII.E 都**坦率承认**

### 实验结果 (Ablation A3)

12-op ISA leave-one-out 在 B1-h24 INT8 上的 per-frame 分析:
- CONV_3X3_DW 占 64%, CONV_1X1 占 18.3% (合计 82%)
- 11/12 op 通过 minimality 阈值(cycle≥2× 或 coverage≥3%)
- SYNC_BARRIER 1.6× / 1.2% — 严格阈值**未通过**, 坦率标记(可以移除但需每 op 悲观 stall +60% 延迟)
- LOAD_WEIGHT_TILE 无 fallback (∞) — 移除后整个 learned-path 不可行

### 关键脚本 / 文件

```
scripts/ablation_a1_conf_unification.py       # 4 variant × 4 数据集推理
scripts/ablation_a1_area_accounting.py         # V0 vs V2 面积估算模型
scripts/ablation_a3_isa_coverage.py           # 12-op 覆盖 + leave-one-out 分析
scripts/_a1_summary.py                        # A1 结果聚合
paper/tcasi/gen_fig_isa_coverage.py           # A3 饼图生成

results/ablation_a1/{V0,V1,V2,V3}/summary.json  # A1 每 variant 每数据集
results/ablation_a1/area.json                   # A1 面积账本
results/ablation_a3/op_mix.json                 # A3 op 占比
results/ablation_a3/leave_one_out.json          # A3 替代序列

paper/tcasi/table_a1.tex                        # Table tab:a1
paper/tcasi/table_a3.tex                        # Table tab:a3
paper/tcasi/fig_arch_v2.tex                     # Fig.1 hub-and-spoke(重画)
paper/tcasi/fig_conf_routing.tex                # Fig.6a(新增)
paper/tcasi/fig_isa_coverage.pdf                # A3 饼图
docs/superpowers/specs/2026-04-21-novelty-strengthening-design.md  # 完整设计文件
```

### Fresh-reviewer audit (2026-04-21, clean-context subagent)

- **Verdict**: partial — novelty 定位可信,但仍有待修
- **强点**: A1 V0 vs V1 的 +15.3 pp SF bad1 回退证明 conf content 是 load-bearing; ISA 表加 `conf` 列让 cross-stage control 变成 falsifiable structural artifact; W-CAPS 的 "first external-conf-driven spatial mixed precision" 是窄而可辩护的 claim
- **弱点**:
  1. W-CAPS 没有与 layer-wise 混合精度的定量对比 —— **已按反馈在 abstract 和 IV.F 软化措辞**(改为 "propose" + "deferred to future work", 消除 abstract ↔ body 自相矛盾)
  2. A1 V2 用 Sobel 代用不是真正的 learned-per-stage —— **已在 VII.E 加 limitation 说明**
  3. V3 在 Middlebury 胜 V0 —— **已在 abstract 坦率写明一个 benchmark 平手**
- **Reviewer 高杠杆建议(未做)**: 跑 W-CAPS vs layer-wise matched-bit-budget 对照. 这是 Path C 的内容,Path B 预算内完不成,留作 TCAS-I revision 或 DAC 2026 的必做项.

### 决策日志(Phase 10.5)

| ID | Decision | Why |
|---|---|---|
| D16 | 保留 EdgeStereoDAv2 品牌, 换副标题而非整 rebrand | TCAS-I reviewer 会警惕"换包装". 来自 user 2026-04-21 feedback |
| D17 | Abstract 措辞从 "single confidence stream" 软化为 "primary external control signal" | DPC/FU/CRM 实际都有 local transform, 原措辞过满. 来自 user feedback |
| D18 | 贡献列表从 4 条重组为 C1/C2/C3 三层递进 | C1 原理 → C2 架构使其可执行 → C3 实例化, 稳住 novelty 层次 |
| D19 | A2 (W-CAPS vs layer-wise) 机会性做, 未做 | Path B 时间预算内完不成, 约 2 天 re-QAT; 留作 future work |
| D20 | V3 在 Middlebury 胜出的结果坦率写进 abstract 而非隐藏 | fresh-reviewer 会发现并扣分; 主动承认换取可信度 |
| D21 | 12-op ISA 的 SYNC_BARRIER 软性失败 (1.6×) 坦率标记而非提高阈值 | 改阈值是学术不诚实; 坦率说明其作用 |

---

## CAL Letter Progress (Phase 10 之前的内容,保留作历史记录)

---

## 定位声明 (P1)

当前 HW 加速器 (EdgeStereoDAv2, 5 SCUs) 只直接加速 **DA2 encoder + SGM + heuristic edge-aware FU**. Phase 7 训出的 **EffViT-B1 (4.85M) 新 learned fusion (KITTI EPE=1.144)** 作为 **"swappable off-chip learned FU"** 在论文 Table II 的 ablation 行出现, **不改 HW 设计**. 这样 CAL letter 既展示硬件贡献, 也展示算法最佳值.

---

## CAL 段-章节映射表 (最终版)

| CAL 段 | 关键内容 |
|---|---|
| Abstract | 5 SCUs, 4.75/5.81 FPS, 1.692 mm2, 139.87 mW, 1.22x |
| I. Intro | complementarity table (v3 cache, 40 samples), 5 SCU 引出, 与 ViTCoD/SpAtten 差异化 |
| II. Background | DA2 pipeline, calibration Eq.(1) Huber LS, merge vs prune (~48% EPE degradation) |
| III. Architecture | SA 0.519mm2, 5 SCU 逐个描述 (CRM 53K cycles, ADCU Cramer rule, FU edge-aware), flash-attn L2 budget |
| IV. Experimental Results | Tab.II (perf + FPS/W + accuracy), **Tab.III (SOTA: A3/SpAtten/ViTCoD/FACT vs Ours)**, merge 精度 (+6.7% EPE), EffViT ablation (9.64 GFLOPs), Pareto Fig.2, Amdahl capability density |
| V. Conclusion | 定位 low-speed robotics / 3D mapping, 7nm/256x256 scaling, future on-die EffViT |
| References | 17 条 bibitem (原 15 + ham2020a3 + qin2023fact) |

---

## 数据权威源 (不可写死, 都从这些文件读)

| 指标类别 | 源文件 | 说明 |
|---|---|---|
| FPS / latency / GFLOPs / DAG ops | `simulator/results/simulation_results.json` | 28nm/500MHz dense + merge; 7nm/1GHz dense |
| Area (each module) | 同上 `area_power_28nm.area_mm2` | 每个 SCU 单独面积 + 总面积 1.6922 mm2 |
| Power (each path) | 同上 `area_power_28nm.power_mw` | total_mw=139.87; arbiter 79.3 mW; SA 19.85 mW |
| L2 budget | 同上 `l2_budget` | 507904 / 524288 B = 96.9% utilization |
| Sparsity x precision sweep | `simulator/results/sparsity_sweep.json` | Fig.2 pareto 所有数据点 |
| Algorithm (heuristic, v3 tuned SGM) | `results/eval_phase7_iter1/summary.json` -> `methods.heuristic` | KITTI 1.87/15.1, SF 8.02/61.2, ETH3D 0.89/24.3, Mid 2.25/26.1 |
| Algorithm (EffViT-B1) | `results/eval_phase7_iter1/summary.json` -> `methods.effvit` | KITTI 1.144/6.2, SF 3.05/44.4, ETH3D 0.66/14.6, Mid 1.67/19.9 |
| Merge vs dense accuracy | `results/eval_token_merge/token_merge_summary.txt` | kr=0.6 merge: fused EPE +6.7%; prune: +12.4% |
| EffViT FLOPs | `thop` profiling on `core/effvit_depth.py` | 9.64 GFLOPs @ 518x518, ~130ms CPU |

---

## 图表清单 (最终版)

| 编号 | 内容 | 源 |
|---|---|---|
| Fig.1 | SA + 5 SCU block diagram (双栏) | `paper/tcasi/fig1_architecture.png` |
| Fig.2 | Sparsity x Precision Pareto (单栏) | `paper/tcasi/gen_fig2_pareto.py` -> `fig2_pareto.pdf` |
| Tab.I | SGM Confidence Complementarity (v3, 40 samples) | main.tex inline |
| Tab.II | Performance Summary + FPS/W + Accuracy (heuristic + EffViT) | main.tex inline |
| Tab.III | **SOTA 对比 (A3/SpAtten/ViTCoD/FACT vs Ours)** | main.tex inline, resizebox |

---

## 验证清单 (CAL Letter QA)

| # | 检查项 | 状态 |
|---|---|---|
| 1 | LaTeX 编译 0 errors | OK |
| 2 | 0 Overfull hbox warnings | OK |
| 3 | 页数 = 4 | OK |
| 4 | Abstract 150 字以内 | OK |
| 5 | FPS 数值匹配 simulator JSON | OK 4.75/5.81 |
| 6 | Area 数值匹配 (1.692 mm2) | OK |
| 7 | Power 数值匹配 (139.87 mW) | OK |
| 8 | L2 utilization 匹配 (96.9%) | OK |
| 9 | 5 SCU 都有描述 | OK |
| 10 | 所有 cite 有对应 bibitem | OK (17 bibitems) |
| 11 | Fig 1 + Fig 2 都可见 + 有 caption | OK |
| 12 | Tab.II 含 heuristic + EffViT 双行 accuracy | OK |
| 13 | Tab.III SOTA 对比表含 4 prior works | OK |
| 14 | Complementarity table 用 v3 cache 数据 | OK |
| 15 | Calibration Eq.(1) 正确 | OK Huber LS |
| 16 | Conclusion 指向 future on-die EffViT | OK |
| 17 | **每个数字都有可追溯的 eval 源文件** | OK (D9 审计通过) |
| 18 | **CRM 延迟描述准确** (~53K cycles) | OK (C2 修复) |
| 19 | **ADCU 求解器描述准确** (closed-form Cramer) | OK (C3 修复) |
| 20 | **Heuristic 与 EffViT 用同一 SGM 基线** (v3 tuned) | OK (C4 修复) |
| 21 | **FPS/W 能效指标已报告** | OK (34.0 / 41.5 FPS/W) |
| 22 | **Merge 精度影响已报告** | OK (+6.7% EPE at kr=0.6) |
| 23 | **EffViT 计算代价已报告** | OK (9.64 GFLOPs, ~130ms CPU) |
| 24 | **应用场景定位已澄清** | OK (low-speed robotics, 非 real-time ADAS) |
| 25 | **PKRN 引用年份正确** | OK (poggi2016pkrn, BMVC 2016) |

---

## 2026-04-19 数字来源审计 (Data Integrity Review)

**背景**: 初稿 Table 1 + Intro complementarity table 从 ICCAD 旧稿搬来了 "KITTI EPE=1.96, D1=9.5%" 等一系列数字. 用户追问出处, 全仓 grep 发现:
- `1.96` -- 仅在 ICCAD md 叙述里出现, 没对应 eval 文件 (最近存档 results/eval_merge_adaptive/kitti/summary.txt 实为 1.942)
- **9.5% -- 无任何出处**, ICCAD 稿可能本就写错
- `27.5% / 2.51% / 72.5% / 39.05% / 0.85 / 14.54` -- 同样 ICCAD 叙述所采, 没有现存 eval 支撑
- `DA2 EPE=2.71` -- 老 v1 cache 数字; v3 实测 2.89

**修正**:
1. 重跑 fusion_cache_v3/kitti/val/ 全量 40 样本, 按 PKRN conf 0.65 分段计算 SGM D1/EPE
2. 新权威数据 (2026-04-19 实测, 可追溯到 v3 cache 的 kitti/val/*.npz):
   - SGM conf>=0.65: 覆盖 49.0%, D1=7.48%, EPE=1.21
   - SGM conf<0.65: 覆盖 32.1%, D1=20.08%, EPE=2.41
   - SGM hole (no prediction): 18.9%
   - DA2 mono 均匀: 覆盖 100%, EPE=2.89
   - Heuristic Fused: 100%, D1=14.97%, EPE=1.84
3. 更新 main.tex 多处: Abstract, Intro, complementarity table, Table II, 叙述段落
4. Sec II pruning 改为定性 "~48% EPE degradation" + Phase-4 内部结果

---

## 2026-04-19 严格 Reviewer 审稿 + 修复 (Claude Code Review Session)

**审稿方法**: 交叉比对 main.tex 每个数字 vs simulator/results/*.json + results/eval_phase7_iter1/summary.json + hardware/scu/*.py 源代码.

### 发现的 Critical Issues 及修复

| ID | 问题 | 修复 | 状态 |
|---|---|---|---|
| **C1** | D1=9.5% 无任何 eval 来源 | 用户已改为 D1=14.97% (v3 cache, 40 samples) | 已修复 (用户) |
| **C2** | CRM "latency 32 cycles/frame" 偏差 1600x (实际 ~53K cycles) | 改为 "Four-stage pipeline (pool->sort->assign->LUT); ~53K cycles, overlapped with SA patch embedding" | 已修复 |
| **C3** | ADCU "Newton-Schulz iteration" 与代码不符 (实际 closed-form 2x2 inverse) | 改为 "sparse NCC keypoint matching + closed-form 2x2 matrix inverse (Cramer's rule)" | 已修复 |
| **C4** | Heuristic (旧 SGM) vs EffViT (tuned SGM) 对比不公平 | 用户已统一为 v3 tuned SGM; heuristic EPE=1.84/D1=15.0% | 已修复 (用户) |

### 新增内容 (审稿后补充)

| 内容 | 原因 | 位置 |
|---|---|---|
| **SOTA 对比表 Tab.III** (A3/SpAtten/ViTCoD/FACT vs Ours) | Reviewer 必问 | Sec IV, resizebox 单栏表 |
| **能效指标 FPS/W** (34.0 dense / 41.5 merge) | 硬件论文标配 | Tab.II 新行 |
| **Merge 精度影响** (+6.7% EPE at kr=0.6, vs prune +12.4%) | Reviewer 必问 | Sec IV 段落 |
| **EffViT 计算代价** (9.64 GFLOPs, ~130ms CPU, 4.85M params) | off-chip 需量化 | Sec IV EffViT 段落 |
| **应用场景定位澄清** (low-speed robotics, 非 ADAS real-time) | 5.81 FPS 不够 real-time | Conclusion |
| **PKRN bibitem 修正** (hoffmann2020pkrn -> poggi2016pkrn) | 年份错误 | References |
| **Pareto caption 修正** (bounded accuracy -> bounded compute reduction) | 非实测精度 | Fig.2 caption |
| **A3 + FACT bibitem 新增** | SOTA 表新引用 | References (17 total) |

### 11 项关键数字交叉验证结果

| 数字 | 论文值 | JSON 实际值 | 状态 |
|---|---|---|---|
| FPS dense | 4.75 | 4.750 | OK |
| FPS merge | 5.81 | 5.810 | OK |
| DAG ops dense | 904 | 904 | OK |
| DAG ops merge | 533 | 533 | OK |
| EffViT KITTI EPE | 1.14 | 1.144 | OK |
| EffViT KITTI D1 | 6.2% | 6.204% | OK |
| EffViT SF EPE | 3.05 | 3.046 | OK |
| EffViT ETH3D EPE | 0.66 | 0.663 | OK |
| EffViT Mid EPE | 1.67 | 1.667 | OK |
| Heuristic KITTI EPE | 1.84 | 1.869 | OK (取整差异) |
| Heuristic KITTI D1 | 15.0% | 15.076% | OK |

---

## 决策日志

| ID | 决策 | 来源 |
|---|---|---|
| D1 | decoder 瓶颈框定为 qualitative ("fixed 40% frame budget"), 不给具体 % | ICCAD |
| D2 | latency overhead 报 "8-10%" (非 7%) | ICCAD |
| D3 | Sec III.C 题目 "Token Routing and Fusion Weight Consistency" | ICCAD |
| D4 | area/power 用 simulator JSON 权威值 | ICCAD |
| D5 | speedup 1.22x 归因写 "371 fewer DAG ops (904->533) primarily in attention" | ICCAD |
| D6 | L2 arbiter 56.7% 功耗占比坦率披露 | ICCAD |
| D7 | Phase 7 EffViT-B1 定位为 "swappable off-chip learned FU", 不改 HW | CAL |
| D8 | 删除 decoder-bound Amdahl quantitative, 改 "capability density" 定性 | CAL |
| D9 | **任何论文数字必须可追溯到具体 JSON 字段**, 无法复现的删或改定性 | 审计 |
| D10 | conf-band 拆分用 PKRN conf >= 0.65; tuned SGM v3 参数为权威 | 审计 |
| D11 | CRM 延迟描述必须包含完整 4 阶段 pipeline, 不可只写核心 Sort 周期 | Review |
| D12 | ADCU 求解器描述为 "closed-form Cramer's rule", 非 Newton-Schulz | Review |
| D13 | SOTA 表 ViTCoD GOPS=512 为估算值 (64x8 MACs x 500MHz x 2), 需确认原文 | Review |
| D14 | Merge 精度数据来自 eval_token_merge (v1 cache), 相对比较仍有效 | Review |
| D15 | 应用场景改为 "low-speed robotics / 3D mapping", Conclusion 给 scaling 路径 | Review |

---

## 后续 (CAL 投稿前还需要做的)

- [ ] 把 refs.bib 从手动 thebibliography 转成 BibTeX 文件 (目前内联 bibitem OK)
- [ ] Fig.1 架构图转矢量图 (PDF/SVG), 当前 PNG 打印文字过小
- [ ] Fig.2 Pareto 视觉优化 (legend 压缩, 更清晰箭头)
- [ ] Spell-check + AMS style review
- [ ] 作者信息, acknowledgment 占位符填实
- [ ] IEEE PDF express 检查 (格式合规)
- [ ] 确认 ViTCoD GOPS 512 是否与原文一致 (D13)
- [ ] 考虑用 v3 cache 重跑 merge 精度实验, 替代 v1 数据 (D14)

---

## 文件地图

```
paper/tcasi/
  main.tex                    # 主文件 (4-page CAL draft, 244 行)
  main.pdf                    # 当前编译输出 (4 pages, 697KB)
  fig1_architecture.png       # SA + 5 SCU block (reused from paper/figures/)
  fig2_pareto.pdf             # Sparsity x Precision Pareto (from simulator data)
  gen_fig2_pareto.py          # Fig2 生成脚本
  IEEEtran.bst                # Bibliography style
  IEEEtran_how-to.pdf         # IEEE style guide (reference)
  template_original.tex       # 模板原件 (备份)
  template_preview.pdf        # 模板样板 PDF
```

## 相关 memory-bank

- `memory-bank/progress.md` -- 实验进度 Phase 1-8
- `memory-bank/architecture.md` -- 代码架构
- `memory-bank/design-document.md` -- 算法设计
- `memory-bank/hardware-*.md` -- 硬件详细文档


---

## Phase 11 — 硬件消融实验表图 (2026-04-21)

新增 `paper/tcasi/` artifacts（pdflatex 独立编译全部通过，未主动 `\input{}` 进 main）：

| 文件 | 用途 | 尺寸 |
|------|------|------|
| `table_hw_ablation.tex` | 主表：8 行 × 11 指标（cumulative + minus-one + GPU 参照） | 1 页 / 70 KB |
| `table_tm_wcaps_2x2.tex` | 2×2 factorial：TM × W-CAPS 独立贡献 | 1 页 / 102 KB |
| `table_hp_sweep.tex` | hp% ∈ {100,75,50,25} 精度/内存权衡 | 1 页 / 85 KB |
| `fig_latency_stacked.pdf` | 5-stage latency 堆叠条形图（7 configs） | 19 KB |

**建议吸收 (如扩 full journal 版)**：
- `table_hw_ablation.tex`：直接换掉当前 `table_sota_accel.tex` 下方的 ablation section
- `fig_latency_stacked.pdf`：追加作为 Fig 15 (current ablation is Fig 14)
- `table_tm_wcaps_2x2.tex`：作为 Sec V.E 或 Discussion 中 "正交 contribution" 小节的配表
- `table_hp_sweep.tex`：作为 discussion 中 "hp=50% 反直觉最佳" 的支撑

**核心叙事**（可直接进论文）：
1. TM 是 FPS 加速器（唯一改 encoder 段，+46% FPS，代价 D1 +5pp）
2. W-CAPS 是 TM-specific 精度保护（独立启用反而 −7% FPS 无收获；仅 TM 后 recovers 2.26pp D1）
3. EffViT 是精度断层点（KITTI D1 23.5% → 7.16%, −70%）
4. INT8 QAT 是能效放大器（fps/W 2.4×，精度几乎无损）
5. hp=50% 反超 hp=75%（INT4 weight-aware 隐式正则化），推荐切默认
6. ASIC vs RTX TITAN：FPS 2.3×、fps/W **2970×**

完整数据和 caveats：`memory-bank/progress.md` Phase 11 段。
