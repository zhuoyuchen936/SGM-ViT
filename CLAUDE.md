# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

SGM-ViT is a hardware-software co-design framework for FPGA-accelerated monocular depth estimation (ICCAD 2025). It combines classical Semi-Global Matching (SGM) stereo with DepthAnythingV2 (DINOv2 ViT) using the SGM confidence map as a free geometric prior to both prune ViT tokens and fuse depth predictions.

## Commands

```bash
# Demo (end-to-end pipeline)
python demo.py
python demo.py --no-sgm                        # skip SGM, test DA2 path only
python demo.py --encoder vitb --weights <path>  # use ViT-B

# KITTI evaluation
python scripts/eval_kitti.py --max-samples 20   # quick sanity check
python scripts/eval_kitti.py                    # full 394-image benchmark

# Ablation scripts
python scripts/eval_fusion.py                   # fusion strategy sweep
python scripts/eval_pruning.py                  # progressive pruning ablation
python scripts/eval_pruning.py --sweep-threshold
python scripts/eval_strategies.py --max-samples 5  # pruning strategy exploration

# Syntax/import check (no GPU needed)
python -c "import core; print(core.__all__)"
```

There is no test suite, linter config, or build system. The project uses direct Python imports. First SGM run triggers Numba JIT compilation (30-90s).

## Architecture

### Inference Pipeline

```
Stereo pair (L+R)
  |-> SGM Engine (Numba-JIT) -> disparity + PKRN confidence map
  |                                    |
  |                     SGMConfidenceTokenRouter
  |                     (pool conf to token grid -> threshold -> binary mask)
  |                                    |
  +-> DepthAnythingV2 ViT --> GAS Sparse Attention (blocks >= prune_layer)
                                       |
                              Token Re-assembly (Gaussian fill)
                                       |
                              DPT Decoder -> mono depth
                                       |
                              Alignment (least-squares scale+shift to SGM disparity space)
                                       |
                              Confidence-Guided Fusion (a*SGM + (1-a)*DA2)
```

### Key Modules

- **`demo.py`** — Main entry point. Contains `load_da2_model()`, `run_masked_sparse_da2()`, `align_depth_to_sgm()`, `fuse_sgm_da2()`, and 4 fusion strategies. Most eval scripts import from here.
- **`core/sparse_attention.py`** — Gather-Attend-Scatter (GAS): physically excludes pruned tokens from attention (vs zeroing which causes attention pollution). `gas_block_forward()` handles a single block, `gas_get_intermediate_layers()` runs the full backbone with intermediate taps for DPT.
- **`core/token_router.py`** — `SGMConfidenceTokenRouter`: pools confidence to token grid via adaptive_avg_pool2d, thresholds, returns `keep_idx`/`prune_idx`. High confidence = prune (SGM reliable, ViT unnecessary).
- **`core/token_reassembly.py`** — Gaussian-weighted interpolation at pruned token positions before DPT decoder. Restores spatial completeness without modifying weights.
- **`core/sgm_wrapper.py`** — Wraps the Numba-JIT SGM engine. Returns disparity + PKRN confidence. `confidence_to_token_grid()` delegates to `pool_confidence()`.
- **`core/pruning_strategies.py`** — 6 alternative mask generators for ablation (random, topk, inverse, checkerboard, CLS attention, hybrid SGM+CLS).
- **`core/eval_utils.py`** — Shared evaluation utilities: `compute_attn_reduction()` (FLOPs), `pareto_frontier()`, `pool_confidence()` (adaptive avg pool to token grid), `compute_token_grid_size()`.
- **`core/hybrid_model.py`** — `SGMViTHybridModel` assembly scaffold (PoC for integrated deployment).
- **`scripts/eval_kitti.py`** — KITTI benchmark utilities: `build_sample_list()`, `read_pfm()`, `read_kitti_gt()`, `load_pkrn_confidence()`, `compute_metrics()`, `aggregate()`. Reused by other eval scripts.
- **`scripts/eval_fusion.py`** — Fusion strategy sweep (4 strategies x parameter grids).
- **`scripts/eval_pruning.py`** — Progressive pruning ablation (prune_layer x threshold).
- **`scripts/eval_strategies.py`** — Pruning strategy exploration (6 alternatives vs SGM baseline).
- **`scripts/eval_latency.py`** — FLOPs reduction and wall-clock latency profiling.

### Critical Constants

- `TOKEN_GRID_SIZE = compute_token_grid_size(518, 14)` = 37 (518px input / 14px patch size)
- `N = 37 * 37 = 1369` spatial tokens
- ViT-S: 12 blocks, embed_dim=384; ViT-B: 12 blocks, embed_dim=768
- Prune mask convention: `True = pruned`, `False = kept`
- DPT intermediate layer indices: `[2, 5, 8, 11]` for ViT-S

### Data Layout

- `Depth-Anything-V2/checkpoints/` — pretrained weights (git-ignored)
- `asserts/left/`, `asserts/right/` — bundled KITTI test pair for demo
- `results/` — experiment outputs (git-ignored)
- `SGM/` — classical SGM stereo engine (Numba-JIT kernels)

### Import Pattern

Eval scripts use a standard path setup:
```python
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "Depth-Anything-V2"))
```

Then import from `demo` (core pipeline functions), `core.eval_utils` (shared utilities), and `scripts.eval_kitti` (KITTI utilities).
