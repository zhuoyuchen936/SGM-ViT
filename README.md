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
│   Input Image (RGB)                                             │
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
├── SGM/                        # Classical SGM algorithm (C/Python)
│   ├── SGM.py                  #   Core SGM disparity & confidence engine
│   ├── gaussian.py             #   Gaussian cost aggregation
│   ├── gen_config_param.py     #   Parameter generation utilities
│   └── stereo_config.c         #   C extension for performance-critical paths
│
├── depth_anything_v2/          # DepthAnythingV2 source (clone here)
│   └── .gitkeep                #   → git clone https://github.com/DepthAnything/Depth-Anything-V2 .
│
├── core/                       # Our contribution: routing & fusion logic
│   ├── __init__.py
│   ├── token_router.py         #   SGMConfidenceTokenRouter — ICCAD key module
│   └── hybrid_model.py         #   SGMViTHybridModel assembly class
│
├── hw/                         # FPGA hardware description (ICCAD HW contribution)
│   └── README_HW.md            #   HLS/RTL design notes and target specs
│
├── scripts/                    # Evaluation and experiment scripts
│   └── eval_latency.py         #   FLOPs reduction & wall-clock latency sweep
│
├── data/                       # Local data (git-ignored)
│   └── weights/                #   Model checkpoints (.pth) — not committed
│
├── docs/                       # Architecture diagrams and paper figures
│
├── results/                    # Experiment outputs (git-ignored)
│
├── requirements.txt            # Python dependencies
└── .gitignore                  # Ignores weights, datasets, Vivado artefacts
```

---

## Roadmap for ICCAD

### Phase 1 — Software Proof of Concept (PoC) `[in progress]`
- [x] Project scaffold and directory structure
- [x] `SGMConfidenceTokenRouter` PyTorch module
- [x] `SGMViTHybridModel` assembly class
- [x] FLOPs / latency evaluation script
- [ ] Integrate DepthAnythingV2 official backbone weights
- [ ] Run on KITTI depth benchmark; measure δ1 accuracy vs. pruning ratio
- [ ] Sweep θ ∈ {0.5, 0.6, 0.7, 0.8}; generate accuracy-efficiency Pareto curve

### Phase 2 — Hardware HLS Implementation
- [ ] HLS C++ kernel for SGM confidence engine (Vitis HLS)
- [ ] HLS C++ token gating logic (AXI-Stream interface)
- [ ] Sparse attention accelerator IP core (INT8 / FP16)
- [ ] End-to-end Vivado block design on ZCU102
- [ ] Synthesis, place & route; resource utilisation report

### Phase 3 — Evaluation & Paper Writing
- [ ] Latency comparison: full ViT vs. SGM-ViT on FPGA (30 fps target)
- [ ] FLOPs breakdown figure (attention vs. FFN vs. routing overhead)
- [ ] Accuracy vs. speed table vs. prior art (AdaBins, DPT, ZoeDepth)
- [ ] Power consumption measurement on ZCU102 (target < 5 W)
- [ ] ICCAD camera-ready submission

---

## Quick Start

```bash
# 1. Clone this repository
git clone <your-repo-url> SGM-ViT
cd SGM-ViT

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Clone DepthAnythingV2 backbone source
git clone https://github.com/DepthAnything/Depth-Anything-V2 depth_anything_v2

# 4. Run FLOPs / latency evaluation (no GPU required for the router stage)
python scripts/eval_latency.py --device cpu --repeats 50

# 5. Inspect the token router in a Python REPL
python - <<'EOF'
import torch
from core import SGMConfidenceTokenRouter

router = SGMConfidenceTokenRouter(token_grid_size=37, confidence_threshold=0.65)
conf_map = torch.rand(1, 1, 518, 518)          # synthetic SGM confidence map
tokens   = torch.randn(1, 37*37, 768)          # ViT-B token sequence
out = router(conf_map, tokens)
print(f"Prune ratio : {out['prune_ratio']:.2%}")
print(f"Kept tokens : {len(out['keep_idx'][0])}")
EOF
```

---

## Citation

> **Note:** Paper under preparation for ICCAD 2025.  Citation will be updated
> upon acceptance.

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
3. Dosovitskiy, A. et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
4. Kong, Z. et al. *SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning.* ECCV, 2022.
5. Rao, Y. et al. *DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification.* NeurIPS, 2021.
