# memory-bank — 项目记忆库索引

**项目**：SGM-ViT 硬件加速器协同设计（ICCAD 2025 CAL letter + DAC/ICCAD 2026 full paper）  
**最后更新**：2026-04-20

本目录是项目的长期记忆，面向新人 onboarding 和跨会话一致性维护。所有文件**持续追加不删历史**；每次重大迭代结束要更新对应进度文件。

---

## 文件清单

| 文件 | 定位 | 写作对象 |
|---|---|---|
| **`README.md`**（本文件） | 顶层索引 | 新人第一页 |
| `architecture.md` | **代码/产物地图**：目录、核心文件、artifacts/results 布局 | 想找某个东西在哪 |
| `design-document.md` | **算法设计**：Phase 1-5 RCF 族 + Phase 6-8 EffViT 端到端 | 想懂为什么这样设计 |
| `progress.md` | **算法进度日志**：Phase 1-8 的每一步（含失败实验 Fix B/C） | 想看完整迭代历史 |
| `paper-progress.md` | **CAL letter 投稿跟踪**（4-page IEEE 提交） | 论文团队 |
| `hardware-architecture.md` | **硬件侧代码地图** + API 速查 | 想找某个 HW 模块 |
| `hardware-design-document.md` | **硬件侧正式 spec**（Specs A-D + 11 模块 + 事件驱动仿真器） | 想懂 HW 设计 |
| `hardware-progress.md` | **硬件侧 Step 1-14 进度日志** | 想看 HW 迭代历史 |

---

## 新人 onboarding 路径（30 分钟）

1. **`README.md`**（这里，1 min）— 看文件职责
2. **`PROJECT_STATUS_CN.md`**（项目根目录，3 min）— 当前活跃 mainline
3. **`design-document.md` 第 10 章**（10 min）— Phase 6-8 算法主线
4. **`architecture.md`** 顶层结构 + Phase 6-8 代码追踪（10 min）— 代码在哪
5. **`progress.md` Phase 7/8 小节**（5 min）— 最近关键决策 + 踩坑

如果是硬件方向：跳过 3-5，改读 `hardware-architecture.md` → `hardware-design-document.md`。

---

## 当前活跃 mainline 快照（2026-04-20）

### 算法侧
- **最强 ckpt**：`artifacts/fusion_phase8_pareto/b2_h48/mixed_finetune/best.pt`  
  - 15.2M params, 33.1 GFLOPs @ 384×768, avg EPE 1.638
- **性价比最佳**：`artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt`  
  - 4.7M params, 9.85 GFLOPs, avg EPE 1.702
- **极致压缩（FPGA 起点）**：`artifacts/fusion_phase8_pareto/b0_h24/mixed_finetune/best.pt`  
  - **735K params**, 4.95 GFLOPs, avg EPE 1.726（仍 8/8 超 heuristic）

全 6 个 Pareto 变体 **8/8 指标全面超越 heuristic**（4 数据集 × EPE & bad）。

### 训练数据
- 主 cache：`artifacts/fusion_cache_v3/`（SF driving 全 8 splits + KITTI/ETH3D/Middlebury）
- tuned SGM 中间产物：`/tmp/sgm_tuned_all/`（P2=3.0, Win=5, range per-dataset）

### Pareto 入口
```bash
python scripts/pareto_analyze.py
# → results/phase8_pareto/{summary_table.md, pareto_plot.png, summary_table.csv}
```

### 硬件侧
- 现有 Phase 1-6 HW 仿真器完整（`hardware/` + `simulator/`），跑 Phase 5 RCF pipeline 可用
- **Phase 9 HW 重构待办**（post-CAL）：详见 `plans/stateful-hatching-spark.md`

### 论文侧
- CAL letter：4-page IEEE，2026-04-19 数据完整性审校通过，见 `paper-progress.md`
- 主论文草稿：`paper/EdgeStereoDAv2_ICCAD.md`

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
