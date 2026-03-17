"""
core/eval_utils.py
==================
Shared evaluation utilities used by multiple scripts in ``scripts/``.

Centralises:
  - ``compute_attn_reduction`` — relative attention FLOPs reduction
  - ``pareto_frontier``        — Pareto-optimal subset extraction
  - ``pool_confidence``        — pool per-pixel confidence to token grid
  - ``compute_token_grid_size``— derive token grid dimensions from input/patch size
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def compute_attn_reduction(
    prune_layer: int,
    n_keep: int | float,
    N: int,
    n_blocks: int = 12,
) -> float:
    """
    Relative attention FLOPs reduction for a ``prune_layer`` configuration.

    Attention cost per block is proportional to sequence length squared.
    - Blocks ``[0, prune_layer)`` : each costs N**2  (dense)
    - Blocks ``[prune_layer, n_blocks)`` : each costs n_keep**2  (GAS)

    Returns the fractional reduction in [0, 1].
    """
    if N == 0:
        return 0.0
    sparse_blocks = n_blocks - prune_layer
    return sparse_blocks / n_blocks * (1.0 - (n_keep / N) ** 2)


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

def pareto_frontier(points: list[dict]) -> list[dict]:
    """
    Given a list of dicts with keys ``'attn_reduction'`` and ``'fused_epe'``,
    return the Pareto-optimal subset (best EPE for each FLOPs budget).

    A point is Pareto-optimal if no other point has both higher
    ``attn_reduction`` AND lower ``fused_epe``.
    """
    pts = sorted(points, key=lambda p: -p["attn_reduction"])
    frontier: list[dict] = []
    best_epe = float("inf")
    for p in pts:
        if p["fused_epe"] <= best_epe:
            frontier.append(p)
            best_epe = p["fused_epe"]
    return frontier


# ---------------------------------------------------------------------------
# Confidence pooling
# ---------------------------------------------------------------------------

def pool_confidence(
    conf_map: np.ndarray,
    token_grid_size: int,
) -> np.ndarray:
    """
    Pool a full-resolution confidence map to the token grid via
    ``F.adaptive_avg_pool2d`` (matches ``SGMConfidenceTokenRouter``).

    Parameters
    ----------
    conf_map        : (H, W) float32
    token_grid_size : target grid side length

    Returns
    -------
    conf_grid : (token_grid_size, token_grid_size) float32
    """
    conf_t = torch.from_numpy(conf_map).float().unsqueeze(0).unsqueeze(0)
    conf_grid = F.adaptive_avg_pool2d(
        conf_t, output_size=(token_grid_size, token_grid_size)
    )
    return conf_grid.squeeze(0).squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Token grid geometry
# ---------------------------------------------------------------------------

def compute_token_grid_size(
    input_size: int = 518,
    patch_size: int = 14,
) -> int:
    """
    Compute the side length of the square ViT token grid.

    Parameters
    ----------
    input_size : int  — model input resolution (default: 518 for DA2)
    patch_size : int  — ViT patch size (default: 14 for DINOv2)

    Returns
    -------
    int : token_grid_size  (e.g. 518 // 14 = 37)
    """
    return input_size // patch_size
