# SGM-ViT 当前状态

**最后更新**: 2026-04-20（Phase 8 Pareto 完成）

## 2026-04-20 快照（Phase 8 完成后）

### 算法侧主线

算法 pipeline 的"融合"这一阶段已**从 heuristic 全面切换到 learned EffViT**。Phase 7/8 结果：

| 变体（7ch 输入）| params | GFLOPs @ 384×768 | avg EPE | 角色 |
|---|---|---|---|---|
| `b0_h24` | **0.735M** | 4.95 | 1.726 | 极致小（FPGA 部署起点）|
| `b0_h48` | 0.879M | 15.89 | 1.747 | — |
| `b1_h24` | 4.703M | **9.85** | 1.702 | **性价比最佳（推荐默认）** |
| `b1_h48` | 4.853M | 20.85 | 1.686 | Phase 7 Iter 1 baseline |
| `b2_h24` | 15.042M | 22.03 | 1.679 | ETH3D/Mid 单项最佳 |
| `b2_h48` | 15.198M | 33.09 | **1.638** | 最强精度 |

**全 6 变体**在 KITTI / SceneFlow / ETH3D / Middlebury 4 数据集 × EPE & bad 的 8 项指标上**全面超越 heuristic_fused** 和 Phase 6 Iter 2 baseline。

- 最强 ckpt：`artifacts/fusion_phase8_pareto/b2_h48/mixed_finetune/best.pt`
- 推荐部署 ckpt：`artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt`
- Pareto 图 + 表：`results/phase8_pareto/pareto_plot.png` + `summary_table.md`

### 当前主线 pipeline

```
左图 ──→ DA2 + Token Merge + W-CAPS decoder ──→ mono disparity
右图 ──┬→ 
左图 ──┘→ SGM (tuned P2=3.0 Win=5) ──→ sgm_disp + hole_mask + PKRN conf

↓
align_depth_to_sgm (Huber)
↓
[RGB, mono, sgm, conf, sgm_valid] (7 channels)
↓
EffViTDepthNet (MIT EfficientViT backbone + FPN decoder)
↓
residual-on-mono → final disparity
```

### CAL letter（投稿中）
- 4-page IEEE CAL，状态：数据完整性审校通过（2026-04-19）
- 论文主稿：`paper/EdgeStereoDAv2_ICCAD.md`
- 投稿追踪：`memory-bank/paper-progress.md`

### Phase 9 HW 重构待办（post-CAL）
现有 HW（Phase 1-6 EdgeStereoDAv2）跑不了 EffViT（FP32 + 权重 4.85MB > L2 512KB + FU 不支持 FPN decoder dataflow）。Phase 9 计划：
1. QAT INT8 EffViT
2. Unified Fusion Engine v2（FPN decoder + 3×3 DW 阵列）
3. WeightStreamer + DRAM streaming
4. Simulator + DAC/ICCAD 2026 full paper

详见 `~/.claude/plans/stateful-hatching-spark.md`。

---

## 历史主线（Phase 1-6 维护期）

以下为 Phase 7 之前的活跃主线（保留代码兼容，不再作为默认路径）：

- `token merge` — 保留
- `decoder-aware W-CAPS / adaptive precision` — 保留
- `edge_aware_residual` heuristic fusion — **作为 heuristic baseline** 保留对比
- `FusionNet`（Phase 1-5 轻量搜索族）— `rcf_refine` ckpt 仍可用，但 Phase 7 EffViT 已全面领先

### 已确认结论（Phase 1-6 时期）

1. `hard token pruning` 会明显伤害 dense DA2 预测，不再作为主线
2. `merge` 比 pruning 更适合 dense prediction，因为它保留了完整空间位置
3. `decoder` 侧降精度明显比 `encoder` 侧温和，尤其 coarse decoder 更有希望
4. ~~`heuristic fusion` 目前仍比 learned fusion 更稳~~ — **此结论已被 Phase 7 EffViT 推翻**：learned fusion 在 4 数据集 × 2 指标上全面碾压 heuristic，边界观感和噪声控制也优于 heuristic_fused。

### Phase 7 的关键突破

Phase 6 内 learned fusion 始终 7/8 胜（唯独 Middlebury bad2 差 1.2pp），**其根因不在网络本身**，而在于：
- v1 fusion_cache 把 SGM 的 hole 填充后的脏值当作"真 SGM"喂给模型
- 训练 SceneFlow 只用了 driving/35mm/scene_forwards/fast 1 个 split 共 300 帧
- 输入通道 sgm_valid 基于 `(sgm>0)` 推断，在 hole-filled cache 下全为 True

Phase 7 一次性解决：
- tuned SGM 重跑（P2=3.0, Win=5）+ v3 cache 格式（explicit `sgm_valid`, sgm 在 hole 处严格 = 0）
- SF driving 全 8 splits = 4400 pair（18× 扩大）
- hole-aug 增强 + per-sample τ 混合 batch
- **结果**：Middlebury bad2 从 27% 降到 20%（超 heuristic 的 26% 有 5.8pp 裕度）

---

## 仓库入口

### Phase 7+ 主入口（推荐）
- `scripts/train_effvit.py` — EffViT 训练（两 stage 链式）
- `scripts/eval_effvit.py` — 4 数据集 eval
- `scripts/build_fusion_cache_v3.py` — v3 cache 构建（tuned SGM + sgm_valid）
- `scripts/rerun_sgm_tuned.py` — 4 数据集统一 tuned SGM
- `scripts/profile_effvit.py` — params + GFLOPs
- `scripts/pareto_analyze.py` — Pareto 汇总
- `scripts/run_phase8_pareto.sh` — auto-chain 6 变体
- `scripts/demo_phase7.py` — 最终 demo（6 面板）

### Phase 1-6 历史入口（兼容保留）
- `demo.py` — Phase 1-5 demo（含 RCF 启发式）
- `scripts/train_fusion_net.py` / `scripts/eval_fusion_net.py` — Phase 1-5 FusionNet 训练/eval
- `scripts/build_fusion_cache.py` — v1 cache（Phase 1-6 用）
- `scripts/precompute_sgm_hole.py` — 旧 SGM 预计算（hardcoded params）
- `scripts/eval_merge_adaptive.py` / `eval_token_merge.py` / `eval_fusion.py` — W-CAPS / token merge / heuristic fusion eval
- `scripts/search_fusion_arch.py` / `run_four_dataset_demos.py` — 架构搜索 / demo

---

## 历史路线归档

- `paper/prior_experiment.md` — hard pruning / two-pass / early FusionNet 等淘汰方向
- `memory-bank/progress.md` — Phase 1-8 完整日志（含 Fix B/C 失败分支实验）

---

## 文档说明

- `README.md` 描述公开入口
- `PROJECT_STATUS_CN.md`（本文件）是**当前快照**
- `paper/EdgeStereoDAv2_ICCAD.md` 是论文主稿
- `paper/prior_experiment.md` 是历史实验档案
- `memory-bank/` 是项目长期记忆库（设计、进度、架构、HW、paper 投稿，共 8 份）；从此目录开始阅读：`memory-bank/README.md`

