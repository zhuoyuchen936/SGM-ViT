# memory-bank — 项目记忆库索引

**项目**：SGM-ViT 硬件加速器协同设计（CAL 2024 letter [偶 prelim] → **TCAS-I 2026 full journal paper**, 现 Phase 10.11 收尾）  
**最后更新**：2026-04-27（追加 phase9_regcal_lessons.md 踩坑记录）

本目录是项目的长期记忆，面向新人 onboarding 和跨会话一致性维护。所有文件**持续追加不删历史**；每次重大迭代结束要更新对应进度文件。

---

## 文件清单

| 文件 | 定位 | 写作对象 |
|---|---|---|
| **`README.md`**（本文件） | 顶层索引 | 新人第一页 |
| `architecture.md` | **代码/产物地图**：目录、核心文件、artifacts/results 布局 | 想找某个东西在哪 |
| `design-document.md` | **算法设计**：Phase 1-5 RCF 族 + Phase 6-8 EffViT 端到端 | 想懂为什么这样设计 |
| `progress.md` | **算法进度日志**：Phase 1-8 的每一步（含失败实验 Fix B/C） | 想看完整迭代历史 |
| `paper-progress.md` | **TCAS-I 投稿跟踪**（19-page journal, Phase 10.5-10.11, 6 轮 reviewer 迭代记录） | 论文团队 |
| `hardware-architecture.md` | **硬件侧代码地图** + API 速查 | 想找某个 HW 模块 |
| `hardware-design-document.md` | **硬件侧正式 spec**（Specs A-D + 11 模块 + 事件驱动仿真器） | 想懂 HW 设计 |
| `hardware-progress.md` | **硬件侧 Step 1-14 进度日志** | 想看 HW 迭代历史 |
| **`phase9_regcal_lessons.md`** | **Phase 9 RegCal 失败踩坑记录**（已废弃路线，5 iter 完整失败模式 + DO NOT REVISIT 列表） | 任何想做 mono+stereo 融合架构的人，先读此 |

---

## 新人 onboarding 路径（30 分钟）

1. **`README.md`**（这里，1 min）— 看文件职责
2. **`PROJECT_STATUS_CN.md`**（项目根目录，3 min）— 当前活跃 mainline
3. **`design-document.md` 第 10 章**（10 min）— Phase 6-8 算法主线
4. **`architecture.md`** 顶层结构 + Phase 6-8 代码追踪（10 min）— 代码在哪
5. **`progress.md` Phase 7/8 小节**（5 min）— 最近关键决策 + 踩坑

如果是硬件方向：跳过 3-5，改读 `hardware-architecture.md` → `hardware-design-document.md`。

如果是论文/审稿方向：跳过 3-5，改读 `paper-progress.md` Phase 10.5-10.11（6 轮 reviewer 迭代历史 + D16-D34 决策日志）。

---

## 当前活跃 mainline 快照（2026-04-22）

### 算法侧
- **最强 ckpt**：`artifacts/fusion_phase8_pareto/b2_h48/mixed_finetune/best.pt`  
  - 15.2M params, 33.1 GFLOPs @ 384×768, avg EPE 1.638
- **性价比最佳**：`artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt`  
  - 4.7M params, 9.85 GFLOPs, avg EPE 1.702
- **极致压缩（FPGA 起点）**：`artifacts/fusion_phase8_pareto/b0_h24/mixed_finetune/best.pt`  
  - **735K params**, 4.95 GFLOPs, avg EPE 1.726（仍 8/8 超 heuristic）

全 6 个 Pareto 变体在 in-house 评测协议下全部超越 heuristic baseline（4 数据集 × EPE & bad）。

### 训练数据
- 主 cache：`artifacts/fusion_cache_v3/`（SF driving 全 8 splits + KITTI/ETH3D/Middlebury）
- tuned SGM 中间产物：`/tmp/sgm_tuned_all/`（P2=3.0, Win=5, range per-dataset）

### Pareto 入口
```bash
python scripts/pareto_analyze.py
# → results/phase8_pareto/{summary_table.md, pareto_plot.png, summary_table.csv}
```

### 硬件侧
- Phase 10 unified FusionEngineV2 + WeightStreamer + 12-op ISA 完整建模（`hardware/` + `simulator/`）
- 端到端 pipeline breakdown 已跑通（`scripts/run_simulator_phase10.py`）
- 两层模型: Tier-1 event-driven simulator (671 行 `simulator/core/event_simulator.py`) + Tier-2 critical-path scheduler (`_estimate_op_cycles`, calibrated 至 Tier-1 内 ±8%)
- 详见 `hardware-progress.md` + `paper-progress.md` Phase 10.6+

### 论文侧（最新, 2026-04-22）
- **TCAS-I journal 投稿稿**：`paper/tcasi/main.pdf` (19 页 / 1.70 MB / 0 errors / 0 undefined refs)
- 标题: *EdgeStereoDAv2: A Confidence-Centric Stereo-Mono Fusion Accelerator for Edge Depth Estimation*
- **6 轮 reviewer 迭代**已收敛: aware track Reject→Major→AcceptWithMinor→Accept, cold-eyes track Major→Minor→AcceptWithMinor
- Phase 10.5-10.11 完整迭代日志见 `paper-progress.md`
- Camera-ready follow-up (非阻塞): F2 W-CAPS vs layer-wise head-to-head, F3 trained per-stage uncertainty head 替换 A1 V2 Sobel surrogate, F4 编译器-regenerated A3 leave-one-out
- 主论文 ICCAD 草稿（备份）: `paper/EdgeStereoDAv2_ICCAD.md`

---

## 文件维护约定

### 追加原则
- `progress.md` / `hardware-progress.md` / `paper-progress.md`：**只追加不删改**，每次重大迭代结尾追加一个 `## Phase N` 或 `## Step N` 小节。历史细节保留为项目演进记录。
- `design-document.md`：主线架构演进时追加新章节（Phase 6-8 → 第 10 章）。原章节不改，保留历史决策脉络。
- `architecture.md`：代码 / 产物布局变化时**就地更新**（目录树、文件表、工作流示例），同步代码现状。表格尽量保留历史列。

### 新文件原则
- 如果新增独立研究线（例如未来"Phase X 新训练方式"），可在 `progress.md` 追加；新研究线如果代码+数据完全独立（如 HW 仿真器），才考虑新建 `<topic>-progress.md`。
- `paper/` 目录下的文档归论文团队（`EdgeStereoDAv2_ICCAD.md` 等），memory-bank 只在 `paper-progress.md` 追投稿状态。

### 不做的
- 不删 Phase 1-5 历史（即使 Phase 7 全面碾压），它们是踩坑记录
- 不改 commit 过的 hardware-*.md（HW 侧改动走硬件 progress）
- 不把代码/ckpt 路径硬编码到多个文件：以 `architecture.md` 为 ground truth

---

## 相关 plans

目录 `plans/`（项目外，`~/.claude/plans/`）保留每次任务的详细计划：
- `stateful-hatching-spark.md` — Phase 8 Pareto 完成计划 + Phase 9 HW 重构草案
- `brainstorm-pipeline-sgm-cost-volume-valiant-seal.md` — 本次 memory-bank 整理 plan
- 其他 `<adj>-<adj>-<adj>.md` — 历史任务 plan（每个对应一次大迭代）

---

## Git 约定

本 memory-bank 目录**纳入版本控制**（2026-04-20 首次 commit）。  
每次重大节点（Phase 完成、paper 投稿、HW milestone）提交一次：
```bash
git add memory-bank/
git commit -m "docs(memory-bank): <subject>"
git push
```
