# SGM-ViT: A Confidence-Guided Sparse Token Accelerator for Monocular Depth Estimation on FPGA

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FPGA](https://img.shields.io/badge/FPGA-Xilinx%20ZCU102-FF6600?logo=xilinx)](https://www.xilinx.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ICCAD](https://img.shields.io/badge/Venue-ICCAD%202025-purple)](https://iccad.com/)

---

## Overview

Deploying large Vision Transformer (ViT)-based depth estimation models — such as
**DepthAnythingV2** — on resource-constrained edge devices (autonomous vehicles,
drones, embedded robotics) remains a fundamental challenge.  The quadratic
complexity of multi-head self-attention scales poorly to the large token
sequences produced by high-resolution inputs, leading to unacceptable latency
and power consumption on FPGA SoCs.

**SGM-ViT** proposes a novel **hardware-software co-design** framework that
bridges the gap between classical stereo vision efficiency and modern
transformer semantic accuracy through two tightly coupled innovations:

1. **Gather-Attend-Scatter (GAS) Sparse Attention** — an AI network-level
   optimization that uses the SGM confidence map to physically exclude
   low-information tokens from the ViT attention computation, reducing
   attention FLOPs while preserving depth quality.

2. **Confidence-Guided Output Fusion** — a hardware-friendly pixel-level
   fusion that uses SGM disparity where stereo matching is reliable and
   falls back to aligned DepthAnythingV2 predictions elsewhere.

> **Core idea:** The *confidence map* produced by the traditional Semi-Global
> Matching (SGM) algorithm serves as a *free geometric prior* that drives
> both **network-level acceleration** (GAS sparse attention) and
> **output-level fusion** (SGM + DA2) — exploiting the complementary
> strengths of classical stereo and modern ViT on dedicated FPGA hardware.

On KITTI 2012+2015 (394 images), the full pipeline — **Fused SGM + GAS Sparse
DA2** — achieves accuracy comparable to the dense fusion baseline while
being **19% faster** in ViT inference, with the GAS mechanism mapping
directly to FPGA token-dispatcher hardware.

---

## Key Innovations

### 1. Gather-Attend-Scatter (GAS) Sparse Attention

Traditional token pruning via **zeroing** (setting pruned token features to
zero) causes catastrophic quality degradation because `softmax(Q * 0 / sqrt(d))`
distributes uniform attention weight to zero vectors, diluting kept tokens'
features across all ViT blocks (EPE 8.29, D1% 66.5% — unusable).

GAS **physically excludes** pruned tokens from the attention computation:

```
For each ViT block >= prune_layer:
  1. GATHER   — extract [CLS, kept_patches] into compact sequence (N_keep + 1 tokens)
  2. ATTEND   — standard multi-head self-attention on compact sequence
  3. SCATTER  — write outputs back to kept positions; pruned tokens retain features
  4. FFN      — MLP on all tokens (per-token, no cross-token dependency)

At DPT intermediate layers: Gaussian reassembly fills pruned positions
```

**Why it works with pretrained weights**: DINOv2 attention is content-based (no
relative positional encoding), positional encoding is added once before all
blocks, and all linear layers / LayerScale are per-token operations.

**Result**: GAS Sparse DA2 EPE **3.27** vs zeroing EPE **8.29** — a **60%
quality improvement** with **19% inference speedup**.

### 2. FPGA Hardware Correspondence

| Stage | Software Operation | FPGA Hardware Unit |
|-------|-------------------|-------------------|
| Routing | `conf > theta` per token | 1 LUT per token (single comparator) |
| Gather | `x[:, gather_idx]` | Token dispatcher reads only kept tokens from BRAM |
| Attend | Standard MHSA on N_keep | Attention PE array, FLOPs proportional to N_keep^2 |
| Scatter | Index write-back | Scatter DMA to original BRAM positions |
| FFN | MLP on all tokens | Streaming MLP PE (full token stream) |
| Reassembly | Gaussian interpolation | Bilinear unit (reuses DPT pyramid HW) |
| Fusion | `alpha * SGM + (1-alpha) * DA2` | 1 DSP + 2 LUT per pixel |

Attention FLOPs reduced by `1 - (N_keep / N)^2`.

### 3. System Pipeline

```
                      SGM-ViT Inference Pipeline
    Stereo Images (L + R)
         |
         +-------------------------------+
         |                               |
         v  [HW: FPGA DSP slices]        v  [HW: FPGA AI Accelerator]
    +-----------+                  +-------------------+
    |    SGM    |                  | DepthAnythingV2   |
    |   Engine  |                  | (ViT-S + GAS)     |
    +-----+-----+                  +--------+----------+
          |                                 |
    Disparity Map (px)              Relative Mono Depth
    PKRN Confidence                         |
    + LR-Check Mask                         |
          |                                 |
          |    +------- GAS Control --------+
          |    |  SGM Confidence -> Token   |
          |    |  Routing Mask -> Gather/   |
          |    |  Scatter in ViT Blocks     |
          |    +----------------------------+
          |                                 |
          |       +-------------------------+
          |       | Disparity-Space Align   |
          +------>| s * depth + t = disp    |
          |       +------------+------------+
          |                    |
          |           Aligned DA2 Disparity
          |                    |
          |  [HW: 1 DSP + 2 LUT per pixel]
          |       +------------+------------+
          +------>| Confidence-Guided Fusion |
          |       | alpha = clip(conf/t,0,1) |
          |       | fused = a*SGM+(1-a)*DA2 |
          |       +------------+------------+
          |                    |
          |           Fused Disparity Map
          |
    (SGM where confident, DA2 where SGM fails)
```

### Confidence Map: PKRN + LR-Check Masking

The confidence map is derived from the aggregated SGM cost volume using the
**Peak Ratio Naive (PKRN)** metric — a hardware-friendly classical stereo
confidence measure requiring only the left cost volume:

```
C = 1 - BestCost / SecondBestCost
```

where *SecondBestCost* is the minimum cost at least `pkrn_min_dist` disparity
steps away from the winning disparity.  Pixels identified as holes (occlusions
or mismatches) by the left-right consistency check are forced to confidence = 0.

This single confidence map serves double duty:
- **Token routing**: High confidence -> prune (SGM is reliable, ViT attention unnecessary)
- **Output fusion**: High confidence -> use SGM disparity, low -> use DA2

---

## Experiment Results (KITTI 2012 + 2015, 394 images)

### Main Results

| Method | EPE (all) | D1% (all) | EPE (sv) | D1% (sv) |
|--------|-----------|-----------|----------|----------|
| SGM (pre-computed) | 11.14 | 29.86 | 1.33 | 8.32 |
| Dense DA2 + align | 2.91 | 30.17 | 2.54 | 26.89 |
| Sparse DA2 (GAS) + align | 3.27 | 35.54 | 2.94 | 32.70 |
| Fused SGM + Dense DA2 | 1.98 | 16.29 | 1.27 | 8.27 |
| Fused SGM + Sparse DA2 | 2.19 | 17.00 | 1.30 | 8.50 |

> **all** = all GT-valid pixels; **sv** = GT-valid pixels where SGM also predicts (SGM coverage region only).

### GAS vs. Token Zeroing

| Sparse Method | EPE (all) | D1% (all) | vs Dense DA2 |
|--------------|-----------|-----------|-------------|
| Token zeroing (broken) | 8.29 | 66.47 | +185% degradation |
| **GAS sparse attention** | **3.27** | **35.54** | +12% degradation |
| Dense DA2 (reference) | 2.91 | 30.17 | -- |

GAS eliminates the attention pollution problem. The remaining 12% gap vs dense
represents the inherent information loss from pruning ~6% of tokens, which is
recovered through output-level fusion with SGM.

### Inference Timing

| Component | Time/image |
|-----------|-----------|
| Dense DA2 | 354.5 ms |
| GAS Sparse DA2 | 287.5 ms (**19% faster**) |

### Best Fusion Configuration per Strategy

| Strategy | Best Params | EPE (all) | D1% (all) | FPGA Cost |
|----------|-------------|-----------|-----------|-----------|
| **soft_blend** | **theta=0.55** | **1.957** | **16.20%** | 1 DSP + 2 LUT |
| two_threshold | theta_l=0.05, theta_h=0.50 | 1.960 | 16.27% | 2 CMP + 1 DSP |
| hard_switch | theta=0.20 | 1.984 | 16.31% | 1 CMP + 1 MUX |
| outlier_aware | theta=0.30, ot=30 | 2.001 | 16.54% | 1 DSP + 3 LUT |

---

## Directory Structure

```
SGM-ViT/
|
+-- demo.py                     # END-TO-END DEMO <- start here
|
+-- SGM/                        # Classical SGM algorithm (Numba-JIT accelerated)
|   +-- SGM.py                  #   Full pipeline: cost volume, aggregation, L-R check
|   +-- gaussian.py             #   Gaussian blur utilities
|   +-- gen_config_param.py     #   Parameter generation helpers
|   +-- stereo_config.c         #   C reference implementation
|
+-- Depth-Anything-V2/          # Official DepthAnythingV2 repo (local clone)
|   +-- depth_anything_v2/      #   DINOv2 ViT backbone + DPT decoder source
|   +-- checkpoints/            #   Pre-trained weights (.pth, git-ignored)
|   |   +-- depth_anything_v2_vits.pth  # ViT-S weights (99 MB) -- used by demo
|   +-- run.py                  #   Official single-image inference CLI
|
+-- core/                       # Our contribution: GAS attention, routing & fusion
|   +-- __init__.py             #   Public exports
|   +-- sparse_attention.py     #   GAS block forward + intermediate layer extraction
|   +-- token_router.py         #   SGMConfidenceTokenRouter
|   +-- token_reassembly.py     #   Gaussian-weighted feature interpolation at pruned sites
|   +-- hybrid_model.py         #   SGMViTHybridModel assembly scaffold
|   +-- sgm_wrapper.py          #   Programmatic SGM API with PKRN confidence
|
+-- asserts/                    # Bundled test stereo pairs
|   +-- left/                   #   Left views (KITTI)
|   |   +-- 000005_10.png       #   KITTI seq 000005 frame 10 (demo default)
|   +-- right/                  #   Corresponding right views
|       +-- 000005_10.png
|
+-- hw/                         # FPGA hardware design (ICCAD HW contribution)
|   +-- README_HW.md            #   HLS/RTL structure, target specs, design notes
|
+-- scripts/                    # Experiment and evaluation scripts
|   +-- eval_latency.py         #   FLOPs reduction & wall-clock latency sweep
|   +-- eval_kitti.py           #   KITTI 2012+2015 accuracy benchmark (EPE, D1)
|   +-- eval_fusion.py          #   Fusion strategy sweep (4 strategies x parameter grids)
|   +-- eval_pruning.py         #   Progressive pruning ablation (prune_layer sweep)
|
+-- docs/                       # Architecture diagrams and paper figures
+-- data/weights/               # Additional model weights (git-ignored)
+-- results/                    # Experiment outputs (git-ignored)
|   +-- demo/                   #   demo.py writes here
|   +-- eval_kitti/             #   eval_kitti.py results
|   +-- eval_fusion/            #   eval_fusion.py sweep results (CSV + summary)
|   +-- eval_pruning/           #   eval_pruning.py results
|
+-- requirements.txt
+-- .gitignore
```

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the full end-to-end demo (GAS sparse attention by default)
python demo.py
# First run: ~30-90 s Numba JIT compilation, then ~5 s per frame.
# Outputs written to results/demo/

# 3. Fast path -- skip SGM, test DAv2 + routing only (instant start)
python demo.py --no-sgm
```

### Demo outputs (`results/demo/`)

| File | Description |
|------|-------------|
| `00_summary.png` | Composite figure -- all panels in one image |
| `01_left_image.png` | Input left stereo image |
| `02_sgm_disparity.png` | Filled SGM disparity map (plasma colourmap) |
| `03_sgm_confidence.png` | PKRN + LR-check confidence (RdYlGn) |
| `04_da2_depth.png` | Dense DepthAnythingV2 ViT-S depth prediction |
| `05_token_routing.png` | 37x37 token grid (green=keep, red=prune) |
| `06_routing_overlay.png` | Routing decision overlaid on input image |
| `07_sparse_da2_depth.png` | GAS Sparse DA2 depth (pruning + Gaussian re-assembly) |
| `08_aligned_disparity.png` | Sparse mono depth aligned to SGM disparity space (pixels) |
| `09_fused_sgm_dense_da2.png` | Fused SGM + Dense DA2 |
| `09b_fused_sgm_sparse_da2.png` | **Fused SGM + Sparse DA2** (GAS-accelerated) |
| `10_diff_map.png` | \|Dense - Sparse\| per-pixel difference |
| `11_comparison.png` | 4-panel paper figure |

### CLI Options

```bash
# SGM & alignment
python demo.py --disparity-range 64     # smaller range -> faster SGM
python demo.py --no-sgm                 # skip SGM (use uniform confidence)
python demo.py --no-align               # skip disparity-space alignment
python demo.py --conf-threshold 0.55    # confidence threshold for alignment & fusion

# GAS sparse attention (default: --sparse-mode mask)
python demo.py --sparse-mode mask       # GAS: physically exclude pruned tokens (default)
python demo.py --sparse-mode zero       # legacy: zero pruned tokens (for comparison)
python demo.py --threshold 0.5          # lower theta -> keep more tokens
python demo.py --prune-layer 6          # start GAS pruning from block 6 (not 0)
python demo.py --no-reassembly          # disable Gaussian feature re-assembly

# Fusion strategies
python demo.py --fusion-strategy soft_blend      # alpha = clip(conf/theta) -- default
python demo.py --fusion-strategy hard_switch     # binary SGM/DA2 selector
python demo.py --fusion-strategy outlier_aware --outlier-threshold 10
python demo.py --fusion-strategy two_threshold --theta-low 0.3 --theta-high 0.7

# Model & I/O
python demo.py --encoder vitb \
  --weights Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth
python demo.py --left path/to/left.png --right path/to/right.png
```

---

## Core Module Reference

### `gas_block_forward` / `gas_get_intermediate_layers` — [core/sparse_attention.py](core/sparse_attention.py)

GAS sparse attention for DINOv2 ViT blocks.

```python
from core.sparse_attention import gas_get_intermediate_layers

# Run ViT backbone with GAS sparse attention
features = gas_get_intermediate_layers(
    backbone     = model.pretrained,    # DINOv2 ViT backbone
    x_input      = image_tensor,        # (B, 3, H, W) input
    layer_indices = [2, 5, 8, 11],      # DPT tap layers for ViT-S
    keep_indices  = keep_idx,           # (N_keep,) LongTensor, kept patch indices
    prune_layer   = 0,                  # first block using GAS (0 = all sparse)
    ffn_on_all    = True,               # run FFN on all tokens (True) or kept only
)
```

### `SGMConfidenceTokenRouter` — [core/token_router.py](core/token_router.py)

Routes ViT patch tokens into **keep** and **prune** sets based on SGM confidence.

```python
router = SGMConfidenceTokenRouter(
    token_grid_size      = 37,    # 518px input / 14px patch = 37
    confidence_threshold = 0.65,  # theta: prune if conf > theta
    learnable_threshold  = False, # set True for end-to-end fine-tuning
)
out = router(conf_map_tensor, tokens)
# out["keep_idx"]    : list[LongTensor]
# out["prune_idx"]   : list[LongTensor]
# out["prune_ratio"] : float
```

### `run_sgm_with_confidence` — [core/sgm_wrapper.py](core/sgm_wrapper.py)

Programmatic SGM pipeline with **PKRN** confidence returning NumPy arrays.

```python
from core import run_sgm_with_confidence

disp_norm, conf_map, disp_raw = run_sgm_with_confidence(
    "asserts/left/000005_10.png",
    "asserts/right/000005_10.png",
    disparity_range = 128,
    smooth_sigma    = 5.0,
    pkrn_min_dist   = 1,
)
```

### `reassemble_token_features` — [core/token_reassembly.py](core/token_reassembly.py)

Gaussian-weighted feature interpolation at pruned token positions, applied
before the DPT decoder to restore spatial completeness.

```python
from core.token_reassembly import reassemble_token_features

filled_features = reassemble_token_features(
    features,          # list of 4 (B, N, D) intermediate feature tensors
    prune_mask_2d,     # (H_t, W_t) bool -- True = pruned token
    patch_h, patch_w,  # token grid dimensions
    sigma=3.0,         # Gaussian spread radius
    kernel_size=9,     # convolution kernel size
)
```

### Fusion Strategies — [demo.py](demo.py)

Four hardware-friendly fusion functions combine SGM and aligned DA2 predictions:

| Function | Strategy | FPGA Cost | Best EPE |
|----------|----------|-----------|----------|
| `fuse_sgm_da2` | alpha = clip(conf/theta) soft blend | 1 DSP + 2 LUT | **1.957** |
| `fuse_hard_switch` | Binary: SGM if conf >= theta, else DA2 | 1 CMP + 1 MUX | 1.984 |
| `fuse_outlier_aware` | Soft blend attenuated by \|SGM-DA2\| | 1 DSP + 3 LUT | 2.001 |
| `fuse_two_threshold` | Dead-zone cascade (theta_low, theta_high) | 2 CMP + 1 DSP | 1.960 |

### KITTI Evaluation — [scripts/eval_kitti.py](scripts/eval_kitti.py)

Evaluates SGM, Dense DA2, Sparse DA2 (GAS), Fused SGM+Dense DA2, and
Fused SGM+Sparse DA2 against KITTI 2012+2015 ground-truth disparity maps.

```bash
python scripts/eval_kitti.py \
    --kitti-root /path/to/kitti \
    --sparse-mode mask               # GAS sparse attention (default)

python scripts/eval_kitti.py --max-samples 20     # quick sanity check
python scripts/eval_kitti.py --no-sparse           # skip sparse DA2
python scripts/eval_kitti.py --sparse-mode zero    # legacy token zeroing
```

### Fusion Strategy Sweep — [scripts/eval_fusion.py](scripts/eval_fusion.py)

Systematic evaluation of all fusion strategies across parameter grids on KITTI.

```bash
python scripts/eval_fusion.py                      # full 394-image sweep
python scripts/eval_fusion.py --max-samples 20     # quick sanity check
```

### Progressive Pruning Ablation — [scripts/eval_pruning.py](scripts/eval_pruning.py)

Sweeps `prune_layer` (and optionally threshold) to characterise the tradeoff
between attention FLOPs savings and depth accuracy.

```bash
python scripts/eval_pruning.py                          # full 394-image sweep
python scripts/eval_pruning.py --max-samples 20         # quick sanity check
python scripts/eval_pruning.py --sweep-threshold        # cross-sweep (prune_layer × threshold)
```

---

## Roadmap for ICCAD

### Phase 1 — Software Proof of Concept `[complete]`
- [x] `SGMConfidenceTokenRouter` PyTorch module
- [x] `SGMViTHybridModel` assembly scaffold
- [x] `run_sgm_with_confidence()` programmatic API
- [x] PKRN hardware-friendly confidence map (cost-volume peak ratio + LR-check mask)
- [x] Token re-assembly (`core/token_reassembly.py`) — Gaussian feature interpolation
- [x] Progressive pruning (`--prune-layer`) — tokens bypass from a chosen ViT block
- [x] **GAS sparse attention** (`core/sparse_attention.py`) — Gather-Attend-Scatter
- [x] Disparity-space alignment — Huber-loss robust regression, no intrinsics needed
- [x] Output-level SGM + DA2 fusion — 4 hardware-friendly strategies (Dense + Sparse)
- [x] End-to-end demo on KITTI stereo pair + DAv2 ViT-S
- [x] FLOPs / latency evaluation script (`scripts/eval_latency.py`)
- [x] KITTI 2012 + 2015 accuracy benchmark (`scripts/eval_kitti.py`, EPE + D1)
- [x] Fusion strategy sweep (`scripts/eval_fusion.py`, 4 strategies x 80 configs)
- [x] Progressive pruning ablation (`scripts/eval_pruning.py`, prune_layer x threshold sweep)

### Phase 2 — Hardware HLS Implementation
- [ ] HLS C++ SGM confidence engine with PKRN (Vitis HLS)
- [ ] HLS GAS token dispatcher + scatter unit
- [ ] HLS confidence-guided fusion kernel (AXI-Stream, soft_blend)
- [ ] Vivado block design on ZCU102
- [ ] Fixed-point (INT16) fusion pipeline characterisation

### Phase 3 — Evaluation & Paper Writing
- [ ] FPGA latency measurement (target: < 100 ms per frame)
- [ ] FLOPs breakdown figure for paper
- [ ] Comparison table vs. AdaBins, DPT, ZoeDepth
- [ ] ICCAD camera-ready submission

---

## Citation

```bibtex
@inproceedings{sgmvit2025,
  title     = {SGM-ViT: A Confidence-Guided Sparse Token Accelerator for
               Monocular Depth Estimation on FPGA},
  author    = {[Authors]},
  booktitle = {Proceedings of the IEEE/ACM International Conference on
               Computer-Aided Design (ICCAD)},
  year      = {2025},
}
```

---

## References

1. Hirschmuller, H. *Stereo Processing by Semiglobal Matching and Mutual Information.* IEEE TPAMI, 2008.
2. Yang, L. et al. *Depth Anything V2.* NeurIPS, 2024.
3. Dosovitskiy, A. et al. *An Image is Worth 16x16 Words.* ICLR, 2021.
4. Kong, Z. et al. *SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning.* ECCV, 2022.
5. Rao, Y. et al. *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification.* NeurIPS, 2021.
6. Hu, X. & Mordohai, P. *A Quantitative Evaluation of Confidence Measures for Stereo Vision.* IEEE TPAMI, 2012.
