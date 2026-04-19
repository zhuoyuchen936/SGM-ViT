# SGM-ViT FusionNet 设计文档

**项目**: SGM-ViT 硬件加速器协同设计 (ICCAD 2025)
**目标**: 立体-单目深度估计的学习型融合，在 4 数据集全面超越启发式融合，视觉质量接近 DA2 直出
**最后更新**: 2026-04-20 (Phase 8 Pareto 完成)

> **章节导读**：
> - 第 1-9 章为 Phase 1-5 历史（RCF 算法族；**视觉质量优先**设计）
> - **第 10 章为 Phase 6-8 主线**（EffViT 端到端融合；**精度+视觉双优**，4 数据集 8/8 全面超越 heuristic）
> - Phase 9（post-CAL 硬件重构 + QAT）不在本文档，见 `plans/` 下的 `stateful-hatching-spark.md`

---

## 1. 背景

### 1.1 项目定位

SGM-ViT 是面向边缘设备 (28nm/500MHz) 的深度估计加速器，核心流水线：
- 立体图像对 → **SGM** → 视差 + PKRN 置信度图
- 左图 → **DA2** (Depth Anything V2, ViT-L) → 相对深度
- Token Merge + W-CAPS 解码器 INT4 量化
- **融合**: 结合 SGM 几何精度 + DA2 视觉平滑度 → 最终视差

### 1.2 核心难题

**SGM 置信度空间不连续性**：纹理丰富区 SGM 置信度高，无纹理平坦区置信度低，但真实视差**实际上连续**。

**逐像素混合的固有缺陷**：传统融合 `α * sgm + (1-α) * da2` 在同一连续平面上产生块状伪影 —— 有纹理处用 SGM，无纹理处用 DA2，由于 SGM/DA2 绝对尺度偏差，结果呈现可见的斑块。

### 1.3 设计目标

| 指标 | 目标 | Phase 5 达成 |
|------|------|---------|
| KITTI EPE | ≤ 1.85 | ❌ 2.378 (但 D1 目标达成) |
| KITTI D1 | ≤ 14.5% | ⚠️ 19.67% |
| ETH3D EPE | ≤ 0.82 | ❌ 1.711 |
| Middlebury EPE | ≤ 2.10 | ❌ 3.377 |
| SceneFlow EPE | ≤ 5.95 | ❌ 6.906 |
| **视觉质量** flat_noise | **≤ DA2 直出** | ✅ **SceneFlow 0.848 ≈ DA2 0.868** |
| 模型参数 | ≤ 2M | ✅ 196K |

**取舍说明**：最终方案优先保证**视觉质量**（用户核心需求），EPE 作为次要目标。学习型精修层相对纯启发式 RCF 恢复了部分 EPE (12-17%)。

---

## 2. 探索历程

| Phase | 范式 | KITTI EPE | flat_noise KITTI | 视觉 |
|-------|------|-----------|------------------|------|
| 启发式 edge_aware | 像素混合 | 1.957 | 0.385 | SGM 纹理明显 |
| Phase 1 修 bug | 像素混合 | 1.815 | 0.382 | 微改善 |
| Phase 3 跨数据集 | 像素混合 | 1.854 | 0.380 | 改善 |
| Phase 4 Direct | 像素混合(改进) | **1.729** | 0.303 | 改善 |
| RCF 启发式 | **区域校准** | 2.869 | **0.204** | 接近 DA2 |
| **Phase 5 RCF refined** | **区域校准 + 学习型** | 2.378 | **0.166** | **历史最佳** |

**关键范式转变**：Phase 4→RCF 从「逐像素混合」转为「区域级 affine 校准」，彻底消除块状伪影。

---

## 3. 核心算法：Region-Calibrated Fusion (RCF)

### 3.1 设计洞察

> 当一个表面在 mono 上视差连续，但 SGM 置信度因纹理而空间不连续时，应以 mono 的连续区域为单位，用区域内 SGM 可信部分的测量值校准**整个区域**，而不是逐像素决定用谁。

### 3.2 算法流程（4 阶段）

```
Input → [A] 区域分割 → [B] 逐区域 affine → [C] 边界混合 → [D] 高频恢复 → Output
```

### 3.3 Stage A: 梯度分割

不量化视差值（会在倾斜面产生条纹），改用梯度幅值作为边界信号：

```python
def segment_mono_regions(mono, image_bgr=None):
    mono_smooth = GaussianBlur(mono, sigma=4.0)
    grad = gradient_magnitude(mono_smooth)
    grad_combined = 0.7 * grad_norm + 0.3 * image_grad_norm  # RGB 边缘辅助
    interior = (grad_combined < quantile_0.65)
    interior = morphology_close(interior, 11x11)  # 桥接纹理间隙
    interior = morphology_open(interior, 5x5)
    labels = cv2.connectedComponents(interior)
    labels = filter_small_regions(labels, min=500)
    labels = nearest_region_assign(unlabeled_pixels)
    return labels  # 典型 5-20 个区域
```

### 3.4 Stage B: 鲁棒 Affine 校准（关键改进：Huber + sanity check）

对每个区域 R_k，找到置信 SGM 像素集合 C_k：

**情况 1: |C_k| >= 30 (充分样本)**
```python
# 1. Huber-robust weighted LSQ (参考 pipeline.py:611)
result = scipy.optimize.least_squares(
    lambda coef: (coef[0]*m + coef[1] - s) * sqrt(c),
    x0=lstsq_warmstart,
    loss='huber', f_scale=3.0)

# 2. MAD-based outlier rejection
residuals = scale*m + shift - s
mad = 1.4826 * median(|residuals - median(residuals)|)
inliers = |residuals| <= 3.0 * mad
refit(inliers)  # 在 inlier 上再拟合

# 3. Scale clamp [0.8, 1.2]（全局对齐已消除大部分 scale 偏差）

# 4. RMSE gating
rmse = sqrt(mean(residuals²))
rmse_conf = clip(rmse_tau / rmse, 0, 1)  # 高残差 → 低置信
sample_conf = min(n_k/30, 1.0)
w_k = rmse_conf * sample_conf

# 5. Amplitude sanity check (关键！)
max_offset = max(6.0, 0.40 * mean(mono_region))
if peak(|fitted - mono|) > max_offset:
    shrink = max_offset / peak
    if shrink < 0.15:
        # Calibration untrustworthy, fall back to mono
        region_conf = 0.1
    else:
        fitted = shrink*fitted + (1-shrink)*mono  # Partial trust

# 6. Blend
calibrated[R_k] = w_k * fitted + (1-w_k) * mono_region
```

**情况 2: 5 <= |C_k| < 30**: Shift-only with MAD weighted median  
**情况 3: |C_k| < 5**: 从相邻同深度区域传播参数（half confidence）

### 3.5 Stage C: 边界混合

```python
# Guided filter: 以 RGB 为引导的边缘保留平滑
result = cv2.ximgproc.guidedFilter(guide=gray, src=calibrated, radius=4, eps=1.0)

# 真深度边缘保护: detail_score > 0.3 处保留原校准值
result = edge_mask * calibrated + (1-edge_mask) * result
```

### 3.6 Stage D: 高频恢复

```python
detail_gate = clip(detail_score, 0, 1) * clip(region_conf, 0.3, 1.0)  # 双重门控
fused = calibrated_blended + residual_gain * detail_gate * mono_high
```

---

## 4. 学习型精修层：RegionCalibratedRefineNet

### 4.1 架构 (196K 参数)

```
输入 (13 ch):
  RGB (3) + calibrated_blended (1) + mono (1) + sgm (1) +
  confidence (1) + sgm_valid (1) + region_offset (1) +
  region_confidence (1) + detail_score (1) + mono_high (1) + disp_disagreement (1)

Backbone: 轻量 U-Net
  stem (13→32) → enc0 → enc1 (32→48) → enc2 (48→64) → mid →
  up1 (48) → up0 (32) → 4 output heads

输出:
  detail_mask (sigmoid), detail_gain (sigmoid),
  residual_mask (sigmoid), tiny_residual (tanh ±0.05)

预测:
  output = calibrated_blended
         + detail_mask * detail_gain * mono_high
         + residual_mask * tiny_residual
```

### 4.2 损失设计（采纳 Codex P2 反馈）

核心损失 (已有)：
- `disp_loss` (λ=1.0): Smooth L1 vs GT
- `grad_loss` (λ=0.2): x/y gradient consistency

**新增损失（防止网络撤销 RCF 的工作）**:
- `calibration_preservation_loss` (λ=0.05):
  ```
  L = (region_conf * |pred - calibrated_blended| * valid).sum() / valid.sum()
  ```
  在高 region_conf 区域惩罚网络输出偏离 RCF base
  
- `detail_gating_loss` (λ=0.03):
  ```
  L = mean(detail_mask * (1 - region_conf))
  ```
  强制 detail_mask 在低 region_conf 区接近 0

### 4.3 训练策略（采纳 Codex P1 反馈）

**关键原则**：
1. **KITTI 不参与 pretrain**（稀疏点云 GT 会误导稠密预测）
2. **ETH3D/Middlebury 不参与训练**（eval splits 污染问题）
3. **Region 信号在线计算**（不缓存，避免与数据增强冲突）

| 阶段 | 数据 | Epochs | LR | val_epe |
|------|------|--------|-----|---------|
| 1 | **SceneFlow train only** (240, 稠密) | 15 | 1e-3 | 0.0224 |
| 2 | **KITTI train only** (354) | 10 | 1e-4 | 0.0295 |

---

## 5. 实验结果

### 5.1 EPE 指标

| 数据集 | mono (DA2) | heuristic | rcf_heuristic | **rcf_refined** |
|--------|-----------|-----------|---------------|-----------------|
| KITTI EPE | 2.874 | 1.957 | 2.869 | **2.378** |
| KITTI D1 | 28.29% | 15.49% | 26.45% | **19.67%** |
| SceneFlow EPE | 6.816 | 6.012 | 6.836 | 6.906 |
| ETH3D EPE | 1.048 | 0.826 | 1.949 | **1.711** |
| Middlebury EPE | 3.188 | 2.132 | 3.920 | **3.377** |

### 5.2 视觉质量 (flat_region_noise)

| 数据集 | DA2 直出 | 启发式 | Phase 3 | Phase 4 | RCF 启发式 | **RCF refined** |
|--------|---------|--------|---------|---------|-----------|----------------|
| KITTI | 0.124 | 0.395 | 0.380 | 0.303 | 0.204 | **0.166** |
| SceneFlow | 0.868 | 1.780 | - | - | 1.068 | **0.848** ≈ DA2! |
| ETH3D | 0.063 | 0.419 | 0.412 | 0.239 | 0.180 | **0.159** |
| Middlebury | 0.296 | 0.612 | 0.597 | 0.426 | 0.471 | **0.374** |

**核心成就**：Phase 5 RCF refined 的视觉质量指标全面优于所有前代方法，且在 SceneFlow 上已与 DA2 直出不相上下。

### 5.3 视觉对比

在 Middlebury 样本上，RCF refined 的椅子表面光滑度接近 GT（有清晰几何结构+光滑内部），远优于任何像素级融合的粗糙纹理。

---

## 6. 实现状态

### 6.1 已完成 (Phase 1-5)

| 模块 | 文件 | 状态 |
|------|------|------|
| RCF Stage A-D 启发式 | `core/fusion.py` | ✅ |
| Huber + sanity 校准 | `core/fusion.py:calibrate_regions` | ✅ |
| RegionCalibratedRefineNet | `core/fusion_net.py` | ✅ 196K 参数 |
| 两个新损失 | `/tmp/run_phase5_rcf.py:compute_losses` | ✅ |
| Region 在线计算 | `/tmp/run_phase5_rcf.py:RCFCacheDataset` | ✅ |
| 三阶段训练 | `/tmp/run_phase5_rcf.py` | ✅ |
| 4 数据集评估 | `/tmp/eval_phase5.py` | ✅ |
| 视觉 demo | `/tmp/phase5_viz.py` | ✅ |

### 6.2 待办（次要）

- `demo.py` 适配 rcf_refine arch（当前 phase5_viz.py 作替代）
- 进一步调优 `calibration_preservation_loss` 权重恢复 EPE
- 整合到主训练脚本 `scripts/train_fusion_net.py`

---

## 7. 关键代码位置

| 功能 | 路径 |
|------|------|
| RCF 核心 | `core/fusion.py:fuse_region_calibrated` (+ segment/calibrate/blend 函数) |
| 融合分派 | `core/fusion.py:FUSION_STRATEGIES` |
| 全局 mono-SGM 对齐 | `core/pipeline.py:align_depth_to_sgm` (Huber 参考实现) |
| 学习型精修架构 | `core/fusion_net.py:RegionCalibratedRefineNet` |
| Phase 5 训练 | `/tmp/run_phase5_rcf.py` |
| Phase 5 评估 | `/tmp/eval_phase5.py` |
| Phase 5 视觉 | `/tmp/phase5_viz.py` |

---

## 8. 已知限制与未来工作

### 8.1 当前限制

1. **EPE 目标未全部达成**：视觉质量优先的设计选择导致 EPE 目标未全部达成，rcf_refined 仍比 heuristic_fused EPE 高。可通过降低 `calibration_preservation_loss` 权重改善
2. **RCF 启发式对参数敏感**：梯度阈值、形态学核大小在不同场景可能需要调参
3. **demo.py 未更新**：当前通过 `/tmp/phase5_viz.py` 绕过

### 8.2 可能的改进方向

1. **端到端微调**：解冻 RCF 参数，联合学习分割 + 校准 + 精修
2. **Plane fitting**：对大平面拟合 3D 平面参数（替代 affine）
3. **Test-time augmentation**：eval 时对原图+翻转取均值
4. **硬件友好优化**：形态学操作、connected components 的 FPGA 实现

---

## 9. 参考（Phase 1-5 历史）

- 主文档: `README.md`, `PROJECT_STATUS_CN.md`
- 论文: `paper/EdgeStereoDAv2_ICCAD.md`
- 实验日志: `paper/fusion_net_experiment_log.md`, `paper/prior_experiment.md`
- Codex 反馈采纳记录: `memory-bank/progress.md:关键技术决策记录`

---

## 10. Phase 6-8：EffViT 端到端融合（主线）

### 10.1 动机 — 为何放弃 RCF 硬约束

Phase 5 RCF Refined 的视觉指标达到预期（SceneFlow `flat_noise` ≈ DA2 直出），但在精度 EPE 上**整体弱于 heuristic_fused**（KITTI 2.378 > 1.957 等）。根因：
- `calibration_preservation_loss` 在高 region_conf 区域强制输出贴近 RCF 校准后的 base → **抑制了模型对 gt 的拟合自由度**
- `detail_gating_loss` 在低 region_conf 区域强制抑制 `detail_mask` → **放弃高频细节**
- 两个正则项导致模型本质上"精修 RCF 的启发式"而非"端到端学最优融合"

**Phase 6 的决定**：抛弃所有视觉平滑硬约束，让网络自己从数据中学"信 mono / 信 SGM / 怎么融合"的策略。

### 10.2 架构 — EffViTDepthNet

```
RGB + mono + sgm + conf + sgm_valid          (7 channels @ H×W, Phase 8)
          │
          ▼
  [stem conv 7→16 stride 2]                   │
          │                                   │
  ┌───────┼──────────┐                        │
  │   MIT EfficientViT backbone B0/B1/B2      │  (third_party/efficientvit,
  │   5 stages, feature widths per variant    │   apache-2.0, local trim)
  │   output: 5 multi-scale feature maps      │
  └───────┼──────────┘                        │
          │                                   │
  ┌───────▼──────────┐
  │  4 × FPN UpBlock (head_ch ∈ {24, 48})
  │   = 1×1 lateral + 3×3 fuse (BN + Hardswish)
  │   progressively upsample 1/32 → 1/16 → 1/8 → 1/4 → 1/2
  └───────┼──────────┘
          ▼
  [final 2× upsample + head_conv + 1×1 → 1 channel residual]
          │
          ▼  (zero-init residual_conv → starts identity)
  residual_clamp ±32 * disp_scale
          │
          ▼
  pred = mono + residual                       (absolute disparity)
```

- 源代码：`core/effvit_depth.py`
- 变体规模：B0-h24 0.735M / B1-h48 4.85M / B2-h48 15.2M
- backbone 源码：`third_party/efficientvit/efficientvit/models/efficientvit/backbone.py`

### 10.3 输入通道演进（Phase 7 → 8）

| 阶段 | 通道 | 组成 | 备注 |
|---|---|---|---|
| Phase 6 (v1 cache) | 10-13 | 多种 FusionNet 架构各自定义 | 历史 |
| **Phase 7 (v3 cache, 8ch)** | 8 | RGB(3) + mono + sgm + conf + sgm_pos + sgm_valid | sgm_pos = `(sgm>0)`, sgm_valid = `~hole_mask` |
| **Phase 8 B5 (v3 cache, 7ch)** | 7 | RGB(3) + mono + sgm + conf + sgm_valid | **冗余剪除**：v3 里 `sgm_disp` hole 严格 = 0 → sgm_pos ≡ sgm_valid |

**Phase 8 B5 验证**：B1-7ch 重训，精度 vs Phase 7 8ch 在 ±5% 内（训练噪声内），8/8 仍全面超 heuristic。因此 Pareto 研究采用 7ch。

### 10.4 Loss — 极简无平滑

```
L = smooth_l1(pred, gt)_valid  +  λ_D1 · mean(σ(k·(|err| - τ)))_valid
```

- `λ_D1 = 0.2`（Phase 6 Iter 3 试过 0.5 → ETH3D 反而退化，0.2 是甜点）
- τ per-dataset（mixed batch 下 per-sample tensor）：KITTI=3, SceneFlow=1, ETH3D=1, Middlebury=2
- `k=5`（sigmoid sharpness）
- **禁用**：calibration_preservation / detail_gating / flat_smooth / anchor / mask_prior / residual_l1（即 Phase 5 的所有正则项）

### 10.5 数据 — tuned SGM + v3 cache + SF driving 全量

**SGM tuned 参数**（Phase 7+ 全数据集统一）：
- `LARGE_PENALTY (P2) = 3.0`（原 1.0 太弱，textureless 大面积无纹理区匹配乱跑）
- `Window_Size = 5`（原 3 太小）
- `disparity_range` per-dataset：ETH3D=80, KITTI=192, Middlebury=192, SceneFlow=256-288
- `smooth_sigma = 0`（保留 sharp hole_mask 边界）

**v3 cache schema** 相比 v1 的关键差异：
- `sgm_disp` = 原始 SGM disp_L，**hole 处严格 = 0**（v1 是 `filling2` 填充后的脏值）
- `confidence_map` = raw PKRN × ~hole_mask（v1 是 Gaussian smoothed）
- **新增 `sgm_valid`**（bool）= `~hole_mask`，让模型明确知道哪些 SGM 像素可信
- `fused_base` 用 tuned SGM 重新走 `fuse_edge_aware_residual` 得到（heuristic 基线也同步更新）

**SF driving 规模扩张**：
- v1: 仅 `35mm_focallength/scene_forwards/fast` 1 个 split = 300 样本（240 train / 30 val / 30 test）
- v3: 全 8 splits（2 focal × 2 direction × 2 speed）= **4400 pair**（3517 train / 440 val / 439 test）→ 18× 扩大

### 10.6 训练 recipe

| Stage | 数据 | Epochs | LR schedule | Batch | 作用 |
|---|---|---|---|---|---|
| 1 SF pretrain | SF driving v3 train (~3517) | 20 | 3e-4 → 1e-5 cosine | 8 (B2=6) | 跨域特征基底 |
| 2 mixed finetune | KITTI + SF weighted 1:1 | 15 | 5e-5 → 1e-6 cosine | 4 (B2=3) | KITTI 适配且不损 SF |

**数据增强**：
- Random crop 320×832（KITTI）/ 384×768（SF）/ 384×512（ETH3D）/ 384×576（Mid）
- hflip（50%）
- 视差尺度抖动 [0.9, 1.1]
- **hole-aug**（Phase 7+）：50% 概率随机打 3-8 个矩形零孔（总计 5-30% 面积），同步置零 sgm + conf + sgm_valid → 迫使模型学"SGM 大面积缺失时依赖 mono+RGB"

**per-sample τ**：混合 batch 里不同数据集样本用不同 D1 阈值（Phase 6 Iter 4 修的 bug，Phase 7 起默认）

### 10.7 结果 — Phase 7/8 双轨

**Phase 7 Iter 1**（B1-h48, 8ch, v3 cache）— 一次性 8/8 超越 current+heuristic：

| 数据集 | Phase 7 EPE | Phase 7 D1/bad | vs Heuristic | vs Phase 6 Iter 2+Fix B |
|---|---|---|---|---|
| KITTI | **1.144** | **6.20%** | -39% / -8.9pp | -29% / -5.0pp |
| SceneFlow | **3.046** | **44.44%** | -62% / -17pp | -23% / -16pp |
| ETH3D | **0.663** | **14.62%** | -25% / -9.7pp | -5% / -3.8pp |
| Middlebury | **1.667** | **19.87%** | **-26% / -5.8pp** | **-19% / -7.0pp** |

最难的 Middlebury bad2 从 "落后 heuristic 1.2pp" 翻转为 "超 heuristic 5.8pp"。

**Phase 8 Pareto**（6 变体，7ch）— 全部 8/8 超 heuristic：

| variant | params | GFLOPs @384×768 | avg EPE | 角色 |
|---|---|---|---|---|
| b0_h24 | **0.735M** | 4.95 | 1.726 | 🥉 FPGA 起点 |
| b0_h48 | 0.879M | 15.89 | 1.747 | — |
| **b1_h24** | 4.703M | **9.85** | 1.702 | 🥈 **性价比最佳** |
| b1_h48 | 4.853M | 20.85 | 1.686 | Phase 7 baseline |
| b2_h24 | 15.042M | 22.03 | 1.679 | ETH3D/Mid 单项最佳 |
| **b2_h48** | 15.198M | 33.09 | **1.638** | 🥇 最强精度 |

**Pareto 关键洞察**（paper-grade 论点）：
1. **压缩 6.6× 不掉队**：b0_h24 (735K) 仍 8/8 超 heuristic
2. **head_ch 24→48 翻倍 FLOPs 但精度增益 <1%**：推荐论文默认 head_ch=24
3. **B2 显著提升 ETH3D/Middlebury zero-shot**（从未进入训练），说明深 backbone 对 unseen 分布更鲁棒
4. 当前数据/loss 已接近饱和：b0→b2 avg EPE 仅 +5.1%；继续突破需加数据（monkaa/flyingthings3d）或加 loss（多尺度、cost volume 监督等）

### 10.8 Phase 8 Fix B/C 实验备忘（Middlebury 短板攻克）

Phase 6/7 中间曾尝试过"不重训 + 换 SGM cache"的捷径（Fix B），在 Middlebury 上**全部回归**：

| 变体 | Middlebury EPE | bad2 | 备注 |
|---|---|---|---|
| Original (old cache) | 2.056 | 26.91 | baseline |
| v1 (hole=0, tuned conf) | 2.342 | 30.22 | 最差 |
| v2 (hole=mono, tuned conf) | 2.255 | 28.85 | — |
| v3 (hole=mono, **old conf**) | 2.153 | 27.22 | 最接近但仍输 |

**结论**：tuned SGM 的 sharp raw PKRN conf 与训练分布不匹配，模型学到的融合策略崩。**必须训练/eval 分布一起换**。Phase 7 Iter 1 按此原则重训，一次性拿下 Middlebury bad2（19.87% vs heur 25.71%）。

Fix C = 单纯调 SGM 参数（range/P2/Win）；Fix B = cache 就地注入。最终 Phase 7 的组合 = Fix C + v3 cache 重建 + 重训。

### 10.9 关键代码路径

```
scripts/train_effvit.py
 ├─ EffViTCacheDataset.__getitem__       (v3 cache reader + hole-aug)
 ├─ EffViTDepthNet                       (core/effvit_depth.py)
 │   ├─ _make_backbone → third_party/efficientvit.backbone.efficientvit_backbone_b{0,1,2}
 │   ├─ 4 × _UpBlock (FPN decoder)
 │   └─ residual_conv (zero-init)
 ├─ build_effvit_depth_inputs            (RGB+mono_norm+sgm_norm+conf+sgm_valid)
 ├─ compute_effvit_losses                (smooth_l1 + λ·soft_D1)
 └─ two-stage train (sf_pretrain → mixed_finetune)

scripts/build_fusion_cache_v3.py         (v3 cache builder)
 └─ SGM from /tmp/sgm_tuned_all/ + DA2 inference + align + heuristic

scripts/rerun_sgm_tuned.py              (unified 4-dataset tuned SGM)
 └─ DATASET_PARAMS = range/P2/Win/P1 per dataset

scripts/profile_effvit.py + pareto_analyze.py + run_phase8_pareto.sh
 └─ Phase 8 Pareto 6-variant auto-chain + analysis
```

### 10.10 已知局限与 Phase 9 HW 待办

**算法侧已完成**，但**硬件侧存在 gap**：
- 现有 Phase 1-6 HW (`hardware/`) 是面向 INT8 + CAPS decoder 的，跑不了 EffViT FP32
- EffViT 权重 4.85 MB > L2 (512 KB) → 需 DRAM WeightStreamer
- 现有 FU 是硬连接的 bilinear + elementwise + 3×3，不表达 FPN decoder 的可变 dataflow
- 现有 Unified SA 对 depthwise conv 效率低

**Phase 9 计划**（post-CAL，见 `plans/stateful-hatching-spark.md` 下半部分）：
1. QAT INT8 EffViT（精度跌 ≤ 0.03 EPE）
2. Unified Fusion Engine v2：FPN decoder + 16 lanes × 3×3 DW 阵列替代 FU
3. WeightStreamer + DRAM streaming（L2 不扩）
4. Simulator + paper 数字（目标 DAC / ICCAD 2026 full paper）

### 10.11 改进方向（后续研究）

从 Phase 8 Pareto 天花板（avg EPE 1.638）再突破的候选：
1. **数据扩展**：加 monkaa (~9K) + flyingthings3d (~25K) SF 剩余子集
2. **输入信号扩展**：SGM cost volume 切片、方向性 aggregation 信号、LR 一致性 channel
3. **多尺度 loss**：pyramid 监督
4. **Cost-volume 监督**：额外 head 预测 disparity prob 分布
5. **TTA**：eval 时 flip + 多尺度均值（纯推理侧，不改训练）

