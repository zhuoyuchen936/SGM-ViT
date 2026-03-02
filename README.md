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

**SGM-ViT** proposes a novel hardware-software co-design framework that bridges
the gap between classical stereo vision efficiency and modern transformer
semantic accuracy:

> **Core idea:** Use the *confidence map* produced by the traditional
> Semi-Global Matching (SGM) algorithm as a *free geometric prior* to identify
> image regions where ViT attention is **not needed** — and skip it entirely
> on dedicated FPGA hardware.

This selective, confidence-guided token pruning achieves **≥ 40 % FLOPs
reduction** in the attention layers of DepthAnythingV2, while preserving
depth accuracy in semantically complex, low-confidence regions where the ViT
backbone is most valuable.

---

## Key Idea

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     SGM-ViT Inference Pipeline                   │
│                                                                  │
│   Stereo Images (L + R)                                         │
│        │                                                         │
│        ├──────────────────────────────────────┐                 │
│        │                                      │                 │
│        ▼  [HW: FPGA DSP slices]               ▼  [HW/SW]       │
│   ┌──────────┐                        ┌──────────────┐         │
│   │   SGM    │                        │  ViT Patch   │         │
│   │  Engine  │                        │   Embed      │         │
│   └────┬─────┘                        └──────┬───────┘         │
│        │                                     │                  │
│   Disparity Map                          Tokens (B, N, D)       │
│   Confidence Map ──────────────────────────► │                  │
│        │              [SW: Python / HLS]     │                  │
│        │          SGMConfidenceTokenRouter   │                  │
│        │               ┌────────────────┐   │                  │
│        │               │  θ-threshold   │◄──┘                  │
│        │               │  comparison    │                       │
│        │               └───┬────────┬───┘                       │
│        │                   │        │                            │
│        │              keep_idx  prune_idx                        │
│        │                   │        │                            │
│        │       [HW: Sparse │Attention PE Array]                  │
│        │       ┌───────────┴──┐     │                           │
│        │       │ Multi-Head   │     │  (bypass — no attn)       │
│        │       │ Self-Attention│     │                           │
│        │       └───────┬──────┘     │                           │
│        │               │            │                            │
│        │        ┌──────┴────────────┴──────────────┐           │
│        └───────►│   Token Re-assembly & SGM Fusion  │           │
│                 └──────────────────┬───────────────┘            │
│                                    │                             │
│                             DPT Decoder Head                    │
│                                    │                             │
│                           Depth Map Output                      │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Logic

| SGM Confidence | Region Type         | Token Action | Hardware Path        |
|:--------------:|---------------------|:------------:|----------------------|
| **High** (> θ) | Textured / flat     | **Prune**    | Bypass attention PE  |
| **Low**  (≤ θ) | Occluded / complex  | **Keep**     | Full attention stack |

---

## Directory Structure

```
SGM-ViT/
│
├── demo.py                     # END-TO-END DEMO ← start here
│
├── SGM/                        # Classical SGM algorithm (Numba-JIT accelerated)
│   ├── SGM.py                  #   Full pipeline: cost volume, aggregation, L-R check
│   ├── gaussian.py             #   Gaussian blur utilities
│   ├── gen_config_param.py     #   Parameter generation helpers
│   └── stereo_config.c         #   C reference implementation
│
├── Depth-Anything-V2/          # Official DepthAnythingV2 repo (local clone)
│   ├── depth_anything_v2/      #   DINOv2 ViT backbone + DPT decoder source
│   ├── checkpoints/            #   Pre-trained weights (.pth, git-ignored)
│   │   └── depth_anything_v2_vits.pth  # ViT-S weights (99 MB) — used by demo
│   └── run.py                  #   Official single-image inference CLI
│
├── core/                       # Our contribution: routing & fusion modules
│   ├── __init__.py             #   Public exports
│   ├── token_router.py         #   SGMConfidenceTokenRouter  ← ICCAD key module
│   ├── hybrid_model.py         #   SGMViTHybridModel assembly scaffold
│   └── sgm_wrapper.py          #   Programmatic SGM API (returns arrays, not files)
│
├── asserts/                    # Bundled test stereo pairs
│   ├── left/                   #   Left views (KITTI, Middlebury, ETH3D)
│   │   └── 000005_10.png       #   KITTI seq 000005 frame 10 (demo default)
│   └── right/                  #   Corresponding right views
│       └── 000005_10.png
│
├── hw/                         # FPGA hardware design (ICCAD HW contribution)
│   └── README_HW.md            #   HLS/RTL structure, target specs, design notes
│
├── scripts/                    # Experiment and evaluation scripts
│   └── eval_latency.py         #   FLOPs reduction & wall-clock latency sweep
│
├── docs/                       # Architecture diagrams and paper figures
├── data/weights/               # Additional model weights (git-ignored)
├── results/                    # Experiment outputs (git-ignored)
│   └── demo/                   #   demo.py writes here
│
├── requirements.txt
└── .gitignore
```

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the full end-to-end demo
python demo.py
# First run: ~30-90 s Numba JIT compilation, then ~5 s per frame.
# Outputs written to results/demo/

# 3. Fast path — skip SGM, test DAv2 + routing only (instant start)
python demo.py --no-sgm
```

### Demo outputs (`results/demo/`)

| File | Description |
|------|-------------|
| `00_summary.png` | 2×3 composite figure — all panels in one image |
| `01_left_image.png` | Input left stereo image |
| `02_sgm_disparity.png` | SGM disparity map (plasma colourmap) |
| `03_sgm_confidence.png` | Per-pixel SGM reliability (red=uncertain, green=confident) |
| `04_da2_depth.png` | DepthAnythingV2 ViT-S depth prediction (Spectral_r) |
| `05_token_routing.png` | 37×37 token grid (green=keep, red=prune) |
| `06_routing_overlay.png` | Routing decision overlaid on input image |

### CLI Options

```bash
python demo.py --threshold 0.5          # lower θ → keep more tokens
python demo.py --disparity-range 64     # smaller range → faster SGM
python demo.py --no-sgm                 # skip SGM (use uniform confidence)
python demo.py --encoder vitb \         # use ViT-B (larger model)
  --weights Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth
python demo.py --left  path/to/left.png \
               --right path/to/right.png
```

---

## Core Module Reference

### `SGMConfidenceTokenRouter` — [core/token_router.py](core/token_router.py)

Routes ViT patch tokens into **keep** and **prune** sets.

```python
router = SGMConfidenceTokenRouter(
    token_grid_size      = 37,    # 518px input / 14px patch = 37
    confidence_threshold = 0.65,  # θ: prune if conf > θ
    learnable_threshold  = False, # set True for end-to-end fine-tuning
)
out = router(conf_map_tensor, tokens)
# out["keep_idx"]    : list[LongTensor] — indices to feed to attention
# out["prune_idx"]   : list[LongTensor] — indices to bypass
# out["prune_ratio"] : float            — fraction of tokens pruned
```

### `run_sgm_with_confidence` — [core/sgm_wrapper.py](core/sgm_wrapper.py)

Programmatic SGM pipeline returning NumPy arrays.

```python
from core import run_sgm_with_confidence

disp_norm, conf_map, disp_raw = run_sgm_with_confidence(
    "asserts/left/000005_10.png",
    "asserts/right/000005_10.png",
    disparity_range = 128,   # must be multiple of 4
    smooth_sigma    = 5.0,   # Gaussian σ for soft confidence edges
)
# disp_norm : (H, W) float32  — normalised disparity [0, 1]
# conf_map  : (H, W) float32  — SGM reliability     [0, 1]
# disp_raw  : (H, W) float32  — raw disparity in pixels
```

### `SGMViTHybridModel` — [core/hybrid_model.py](core/hybrid_model.py)

Assembly scaffold that injects `SGMConfidenceTokenRouter` into a DepthAnythingV2
backbone.  The `vit_backbone` and `dpt_head` arguments accept real DAv2 modules.

---

## Roadmap for ICCAD

### Phase 1 — Software Proof of Concept `[active]`
- [x] `SGMConfidenceTokenRouter` PyTorch module
- [x] `SGMViTHybridModel` assembly scaffold
- [x] `run_sgm_with_confidence()` programmatic API
- [x] End-to-end demo on KITTI stereo pair + DAv2 ViT-S
- [x] FLOPs / latency evaluation script (`scripts/eval_latency.py`)
- [ ] Full DAv2 backbone integration into `SGMViTHybridModel`
- [ ] KITTI depth benchmark (δ₁ accuracy vs. prune ratio)
- [ ] Accuracy-efficiency Pareto curve across θ

### Phase 2 — Hardware HLS Implementation
- [ ] HLS C++ SGM confidence engine (Vitis HLS)
- [ ] HLS token gating kernel (AXI-Stream)
- [ ] Sparse attention accelerator IP (INT8 / FP16)
- [ ] Vivado block design on ZCU102

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
3. Dosovitskiy, A. et al. *An Image is Worth 16×16 Words.* ICLR, 2021.
4. Kong, Z. et al. *SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning.* ECCV, 2022.
5. Rao, Y. et al. *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification.* NeurIPS, 2021.
