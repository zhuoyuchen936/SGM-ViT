"""
core/adaptive_precision.py
==========================
Confidence-aware adaptive precision planning for token-merged SGM-ViT.

This module implements CAPS-Merge-v1:
  1. Build merge groups from PKRN confidence.
  2. Score each representative/group by sensitivity.
  3. Assign high precision to the most sensitive groups and low precision to
     the remaining groups.

The current implementation is an algorithmic prototype intended to validate
the routing signal, not a hardware-faithful quantized kernel.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .token_merge import build_token_merge_groups


def fake_quantize_tensor(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    """
    Symmetric fake quantization with per-token scaling on the last dimension.

    ``bits >= 16`` or ``None`` is treated as full precision.
    """
    if bits is None or bits >= 16:
        return x
    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        return x
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / float(qmax)
    return (x / scale).round().clamp(-qmax, qmax) * scale


def build_caps_merge_plan(
    conf_map: np.ndarray,
    token_grid_size: int | tuple[int, int],
    keep_ratio: float,
    high_precision_ratio: float = 0.25,
    conf_weight: float = 1.0,
    var_weight: float = 0.5,
    range_weight: float = 0.5,
    radius_weight: float = 0.25,
) -> dict[str, object]:
    """
    Build a merge plan and assign adaptive precision to each representative.

    Group sensitivity is defined as:

        score =
            conf_weight  * (1 - mean_conf)
          + var_weight   * std_conf
          + range_weight * (max_conf - min_conf)
          + radius_weight * mean_group_radius
    """
    merge_plan = build_token_merge_groups(conf_map, token_grid_size, keep_ratio)

    grid_h, grid_w = merge_plan["grid_shape"]
    conf_flat = torch.from_numpy(merge_plan["conf_grid"].reshape(-1)).float()
    rep_idx = merge_plan["rep_idx"]
    groups = merge_plan["groups"]
    rep_count = int(merge_plan["rep_count"])

    all_idx = torch.arange(grid_h * grid_w, dtype=torch.long)
    all_coords = torch.stack([all_idx // grid_w, all_idx % grid_w], dim=1).float()
    rep_coords = all_coords.index_select(0, rep_idx)
    max_radius = max(math.sqrt((grid_h - 1) ** 2 + (grid_w - 1) ** 2), 1.0)

    scores = []
    stats: list[dict[str, float]] = []
    for rep_local, members in enumerate(groups):
        vals = conf_flat.index_select(0, members)
        member_coords = all_coords.index_select(0, members)
        rep_coord = rep_coords[rep_local].unsqueeze(0)
        mean_conf = float(vals.mean().item())
        std_conf = float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0
        conf_range = float((vals.max() - vals.min()).item()) if vals.numel() > 1 else 0.0
        mean_radius = float(
            torch.norm(member_coords - rep_coord, dim=1).mean().item() / max_radius
        )
        score = (
            conf_weight * (1.0 - mean_conf)
            + var_weight * std_conf
            + range_weight * conf_range
            + radius_weight * mean_radius
        )
        scores.append(score)
        stats.append(
            {
                "mean_conf": mean_conf,
                "std_conf": std_conf,
                "conf_range": conf_range,
                "mean_radius": mean_radius,
                "score": float(score),
            }
        )

    score_t = torch.tensor(scores, dtype=torch.float32)
    hp_count = max(0, min(rep_count, int(round(rep_count * high_precision_ratio))))
    high_precision_local_mask = torch.zeros(rep_count, dtype=torch.bool)
    if hp_count > 0:
        _, hp_idx = score_t.topk(hp_count, largest=True)
        high_precision_local_mask[hp_idx] = True

    merge_plan.update(
        {
            "group_scores": score_t,
            "group_stats": stats,
            "high_precision_ratio": high_precision_ratio,
            "high_precision_count": hp_count,
            "low_precision_count": rep_count - hp_count,
            "high_precision_local_mask": high_precision_local_mask,
            "low_precision_local_mask": ~high_precision_local_mask,
        }
    )
    return merge_plan
