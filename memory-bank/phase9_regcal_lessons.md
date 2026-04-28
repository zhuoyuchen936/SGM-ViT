# Phase 9 RegCal-Fusion 失败踩坑记录（已废弃路线）

**结论先行**：以"`lerp(mono_aligned, region-affine·mono, gate) + small residual_head`" 为核心的 RegCal 架构（参数量 1.2-1.8M）在 4 数据集 8 项主要指标上**全部输给 Phase 8 EffViT-b1_h24（4.7M）41-181%**，且在 zero-shot 泛化和 finetune 单数据集之间存在不可调和的 trade-off。**此路不通，未来不要再走**。所有相关代码、ckpt、eval 结果、日志已 2026-04-27 删除，分支 `phase9-regcal` 与 tags 已删除。

---

## 1. 时间线与目标

- **起点**：2026-04-26，master `dcf2c1e`（Phase 7-11 wrap-up）
- **目标**：用 SGM 引导的"区域级仿射校准 + ill-posed mask"替代 Phase 8 的 EffViT 7ch concat residual，参数量 ~1.5M（vs Phase 8 4.7M），主打 HW friendliness + 可解释 + zero-shot 反光场景
- **必须超越的 Phase 8 baseline**（artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt）：
  - KITTI 1.158 / D1 6.40
  - SceneFlow 3.172 / bad1 45.00
  - ETH3D 0.664 / bad1 16.02
  - Middlebury 1.814 / bad2 21.83

## 2. 架构设计（已废弃）

```
DA2 ViT-S encoder (frozen) -> tokens at layers [5,8,11]
  ├-> DA2 DPTHead (frozen) -> mono_disp ──────────────────┐
  └-> LiteDPTDecoder (1.18M, 镜像 DPT 后 2 stage) -> f_14 │
                                                          │
SGM (P2=3, Win=5) -> sgm_disp + PKRN_conf + sgm_valid    │
align_depth_to_sgm (Huber) -> mono_aligned               │
                                                          ↓
RegCalFusionHead:
  Sobel(mono_aligned) -> binary edge_map
  cv2.connectedComponents -> region labels (≤256)
  per-region 1-step Huber-weighted IRLS LS (no_grad) -> α_r, β_r
  IllPosedRefiner (34K, lite_f14 + 5 hand signals) -> σ ∈ [0,1]
  residual_head (6K-262K) -> Δd
  final = (1-gate)·mono_aligned + gate·(α·mono+β) + Δd
  其中 gate = calib_mask · (1 - σ)
```

5 个子模块定义详见已删除的 `core/regcal_net.py`（commit `784a44f` 历史可查）。

## 3. 5 个 iter 的失败模式

### iter-1: 无梯度通路（dead 1.18M params）
- **配置**：no residual_head，仅 lerp 路径，cross-flaw 自监督但默认关闭
- **训练**：15 epoch SF, batch 8
- **结果**：SF val EPE 5.83 ≈ mono baseline 5.92（**完全没学**）
- **根因**：AffineFitter `@torch.no_grad` 切断梯度。lite_decoder 的梯度仅能经 IllPosedRefiner → ill_posed_mask → gate 流回。当 region affine 在 SF 上拟合不好时，模型最优解就是 σ=1（全信任 mono）→ gate=0 → 退化为 mono baseline。lite_decoder 的 1.18M 参数训练了 15 epoch 但毫无作用。
- **教训**：**不要在能学的 backbone 后接「两个常数输入做 lerp」的结构**。必须有显式的 residual 路径让 backbone 梯度流到最终预测。

### iter-2: 加 residual_head + GT-direct ill-posed 监督
- **修复**：加 `residual_head` (6K params 深度可分离卷积，zero-init) + GT-direct illposed 标签（`(|sgm-gt|>2px) & sgm_valid`）。`final = lerp + residual.clamp(-32,32)`，pixel domain。
- **训练**：S1 SF 15 epoch + S2 KITTI 25 epoch
- **结果**：
  - 4 数据集：KITTI 2.32 / SF 5.72 / ETH3D **1.91**（vs mono 1.02 退 87%）/ Mid **3.29**（vs mono 3.21 微退）
  - iter-2 S1-only（无 KITTI ft）on KITTI: **EPE 8.09**（vs mono 2.88 退 3 倍）
- **根因**：**residual_head 输出在 pixel domain**。SF 训练时学到 SF-scale (~10-30 px) 的残差。KITTI/ETH3D test 时同样的"残差方向 + 量级"完全错配。Scale-invariance 缺失导致跨域 catastrophic generalization。
- **教训**：**任何 residual head 必须 scale-invariant**——要么输出归一化值再乘 disp_scale，要么用足够紧的 clamp 限幅。

### iter-3: 容量消融（确认容量不是瓶颈）
- **修复**：residual_head 从 6K 提到 262K（45×，full Conv3x3 mid=96 ×3 blocks）。其他不变（仍 pixel domain，无 scale 修复）。
- **结果**：
  - KITTI 2.29（iter-2 2.32，仅 +1%）
  - SF 6.67（iter-2 5.72，**反而 -17%**）
  - ETH3D 1.73（iter-2 1.91，+9%）
  - Mid 3.60（iter-2 3.29，**-9%**）
- **根因**：容量增加在 KITTI/ETH3D 略有改善，但在 SF/Mid 反而退化（更大 head 在 KITTI ft 时 overfit 更深）。**45× 容量改变 → 整体仅 ±10% 波动 → 容量不是核心瓶颈**。
- **教训**：**遇到性能瓶颈先诊断"哪条轴"，再决定是否堆参数**。没有 scale 修复的情况下，更大的 residual_head 学到更精确但更不可迁移的 scale 偏置，反而坏事。

### iter-4: scale-invariant 修复 + RGB context
- **修复**：
  ```python
  norm_residual = tanh(residual_head(...))   # [-1, 1]
  residual = norm_residual * disp_scale       # 自动适配数据集尺度
  final = base + residual.clamp(±32)
  ```
  + 把 RGB 加到 residual_head 输入（之前只有 lite_f14 + 5 hand signals）。
- **训练**：S1 SF 15 epoch + S2 KITTI 25 epoch
- **结果**：
  - **iter-4 S1-only（无 KITTI ft）zero-shot 大胜**：KITTI 3.93 (-51% vs iter-2 8.09) / SF 4.44 / ETH3D **1.15** (-87% vs iter-2 9.20) / Mid **3.50** (-66% vs 10.26)。**ETH3D 1.15 ≈ mono 1.02**，泛化几乎不退步。
  - **但 iter-4 S2 KITTI ft 后 SF 崩坏**：KITTI 2.31 / **SF 10.24**（vs mono 5.92 退 73%）/ ETH3D 1.09 / Mid 3.71
- **根因**：scale-invariance 修复**非对称**。tanh·scale 在「test scale ≤ train scale」时安全（ETH3D scale < SF scale → tanh 输出适用），但「test scale > train scale」时崩溃（SF scale > KITTI scale → tanh 学到 KITTI-典型小值，但乘以 SF 大 scale 后残差被放大到错误方向）。S2 KITTI ft 让 tanh 输出适配 KITTI 小 scale，再 SF eval 时残差量级 + 方向都偏。
- **教训**：**scale-invariance 不能仅靠 tanh·disp_scale 这一招**。还需考虑训练分布的 scale 范围与 test 分布的差异。或者训练时显式覆盖多 scale（mixed_finetune），单数据集 ft 必然 forgetting。

### iter-5: SF 恢复（低 lr 重训 SF 救 forgetting）
- **修复**：从 iter-4 S2 best.pt 启动，SF lr=1e-5 跑 5 epoch（救 SF 不退步太多）
- **结果**：KITTI **3.28**（vs iter-4 2.31 退 +14%）/ SF **4.46**（成功恢复！）/ ETH3D 1.05 / Mid 3.28
- **是 5 个 iter 中最均衡的 ckpt**：1/4 显著赢（SF -25% vs mono）+ 2/4 ≈ mono（ETH3D / Mid 差 < 3%）+ 1/4 微退（KITTI +14%）
- **但仍全部输 Phase 8 EffViT 41-181%**
- **根因**：单数据集 fine-tune 必然 trade-off 另一个数据集。Phase 8 用的是 "mixed_finetune"（SF+KITTI 交替 batch）才避开了这个 trade-off。RegCal 架构没有跳出这个局限。
- **教训**：**S1 → S2 顺序 finetune 在多数据集场景下天然有 forgetting**。如果要走多数据集 SOTA 路线，必须从 day 1 就用 mixed dataloader。

## 4. 最终性能对照表

| Dataset | mono | heuristic | iter-1 | iter-2 | iter-3 | iter-4 | **iter-5** | Phase 8 EffViT |
|---|---|---|---|---|---|---|---|---|
| KITTI EPE | 2.88 | 1.87 | 失败 | 2.32 | 2.29 | **2.31** | 3.28 | **1.16** |
| SF EPE | 5.92 | 8.02 | 5.83 | 5.72 | 6.67 | 10.24 | 4.46 | **3.17** |
| ETH3D EPE | 1.02 | 0.89 | - | 1.91 | 1.73 | 1.09 | 1.05 | **0.66** |
| Mid EPE | 3.21 | 2.25 | - | 3.29 | 3.60 | 3.71 | 3.28 | **1.81** |

**所有 RegCal iter 在所有 4 数据集 EPE 上均显著输 Phase 8**，差距 41-181%。

## 5. 不要再做的事（DO NOT REVISIT）

1. **不要再用 "lerp(常数A, 常数B, 学习的 gate) + 小 residual" 结构覆盖容量大的 backbone**——梯度通路单一，backbone 成为高功耗装饰品。
2. **不要在 residual head 输出 pixel domain 残差**——必然跨域 catastrophic。最低标准是 `tanh(.) * disp_scale_clamp`，但即便如此也只解决一半问题。
3. **不要把 region affine 的 `no_grad`-计算和 residual 一起当作 fusion 主路径**——AffineFitter 必须搭配显式可学习残差，否则 backbone 梯度白学。
4. **不要期待 1-1.5M 参数的 head 在 4 数据集 8 指标上击败 4.7M EffViT**——参数量差距 ≈ 3×，accuracy gap 不可弥合（实证 41-181%）。如果要拼精度，先把 head 容量做到 3-4M。
5. **不要 S1 单数据集 → S2 单数据集 sequential ft**——必然 forgetting。multi-dataset 必须 mixed dataloader 从头训。
6. **不要因为某个 iter "感觉对了" 就堆容量**——iter-3 证明，未诊断 scale 问题就堆容量纯属浪费 5h compute。先做 single-axis ablation 确认瓶颈轴。
7. **不要把"HW efficiency angle"当作 accuracy 已经输的兜底叙事**——如果 accuracy 输 100%，HW efficiency 论文 reviewer 也接受不了。需要 accuracy 至少不退步 mono baseline 才有讨价还价资本。

## 6. 如果以后真的还要做 mono+stereo 融合，可考虑的方向

（**仅备忘，未验证**）：

1. **Phase 8 EffViT 当主干 + RegCal 当辅助监督**：把 region affine、ill-posed mask 当作 auxiliary loss / regularizer，不当作主预测路径。Phase 8 主线保持，论文卖点是 "interpretability injection without accuracy loss"。
2. **直接 fork MonSter (CVPR 2025) 的 RAFT-stereo + DA2 cross-flaw GRU 路线**，缩参（vits 替 vitl）+ 单 iter（不要 12 iter）。预估能到 KITTI EPE ~1.5。这条路有 SOTA 验证，但参数量必然 ≥ 5M。
3. **抛弃 mono+stereo 融合方向，直接做 Phase 8 EffViT 的 INT4 / INT8 QAT 极致量化**——这是 Phase 9 QAT 已经在走的路，CAL letter 已投，TCAS-I 在写。继续这条路 ROI 比硬冲精度新架构高。

## 7. 复现路线（仅供反思，不再实际执行）

如果未来某人想验证本文档结论，重建步骤：
- master `dcf2c1e` 之上，按 commit `784a44f` 的内容（已删除，但可从 git reflog 恢复 ~30 天）创建 `core/regcal_net.py` + `core/regcal_losses.py` + `scripts/train_regcal.py` + `scripts/eval_regcal.py` + `scripts/build_regcal_demo.py`
- 5 个 iter 的训练命令在已删除的 `results/phase9_*_train.log` 头部
- Phase 8 baseline `results/eval_phase8_baseline_for_phase9/summary.json` 也已删除，重跑命令：
  ```
  python scripts/eval_effvit.py \
    --ckpt artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt \
    --cache-root artifacts/fusion_cache_v3 \
    --out-dir results/eval_phase8_baseline_for_phase9
  ```

但**强烈不建议复现**——实证已经清晰。

---

*最后更新：2026-04-27 · 完成 5 iter 失败循环后归档。后续若有相关想法，先回看本文档。*
