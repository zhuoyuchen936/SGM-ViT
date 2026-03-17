# SGM-ViT Claude TODO — Practical Engineering Tasks

Based on a thorough audit of the existing codebase (2026-03-17).

## Status Legend
- [ ] pending
- [x] done

---

## Phase A: Bug Fixes

- [x] **A1. Fix `hybrid_model.py` p_idx assignment bug**
  Removed dead `p_idx = keep_idx[b]` variable that shadowed correct prune indices.

## Phase B: Code Deduplication & Shared Utilities

- [x] **B1. Create `core/eval_utils.py` for shared evaluation helpers**
  Centralised `compute_attn_reduction()`, `pareto_frontier()`, `pool_confidence()`,
  and `compute_token_grid_size()`. Updated imports in `eval_pruning.py`,
  `eval_strategies.py`, and `pruning_strategies.py`.

- [x] **B2. Consolidate `confidence_to_token_grid` with `pool_confidence`**
  `sgm_wrapper.confidence_to_token_grid()` now delegates to
  `eval_utils.pool_confidence()` (PyTorch adaptive_avg_pool2d) for consistency
  with `SGMConfidenceTokenRouter`.

## Phase C: Code Quality Improvements

- [x] **C1. Remove dead code in `hybrid_model.py._sparse_attention`**
  Dead `p_idx` variable removed (was part of A1).

- [x] **C2. Add `core/eval_utils.py` to `core/__init__.py` exports**
  All 4 functions exported: `compute_attn_reduction`, `pareto_frontier`,
  `pool_confidence`, `compute_token_grid_size`.

- [x] **C3. Update `core/__init__.py` docstring**
  Added `sparse_attention` and `eval_utils` module descriptions.

## Phase D: Dynamic Token Grid Computation

- [x] **D1. Make TOKEN_GRID_SIZE dynamically computed**
  Added `compute_token_grid_size(input_size, patch_size)` to `core/eval_utils.py`.
  Updated `demo.py` and `scripts/eval_latency.py` to use it.

## Phase E: CLAUDE.md Update

- [x] **E1. Update CLAUDE.md to reflect new modules and structure**
  Added `core/eval_utils.py`, all eval scripts, and `core/hybrid_model.py`
  to the key modules section. Updated critical constants and import pattern.
