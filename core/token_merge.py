"""
core/token_merge.py
===================
Confidence-guided token merge planning for SGM-ViT.

This module constructs a merge plan from an SGM-derived confidence map.
Unlike hard pruning, merge keeps the full spatial token grid intact for the
decoder while reducing attention sequence length by routing each token to a
representative token.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_grid_shape(token_grid_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return ``(grid_h, grid_w)`` from an int or explicit tuple."""
    if isinstance(token_grid_size, tuple):
        return int(token_grid_size[0]), int(token_grid_size[1])
    size = int(token_grid_size)
    return size, size


def pool_confidence_to_grid(
    conf_map: np.ndarray,
    token_grid_size: int | tuple[int, int],
) -> np.ndarray:
    """
    Pool a full-resolution confidence map to the token grid.

    The implementation matches the router's adaptive average pooling, but also
    supports non-square token grids encountered after resizing.
    """
    grid_h, grid_w = _normalize_grid_shape(token_grid_size)
    conf_t = torch.from_numpy(conf_map).float().unsqueeze(0).unsqueeze(0)
    conf_grid = F.adaptive_avg_pool2d(conf_t, output_size=(grid_h, grid_w))
    return conf_grid.squeeze(0).squeeze(0).cpu().numpy()


def build_token_merge_groups(
    conf_map: np.ndarray,
    token_grid_size: int | tuple[int, int],
    keep_ratio: float,
) -> dict[str, object]:
    """
    Build a confidence-guided merge plan.

    Parameters
    ----------
    conf_map : (H, W) float32
        PKRN confidence map in image space.
    token_grid_size : int or (grid_h, grid_w)
        Spatial token-grid shape.
    keep_ratio : float
        Fraction of representative tokens to keep.

    Returns
    -------
    plan : dict
        ``rep_idx``             : (K,) LongTensor of representative token indices
        ``member_to_rep``       : (N,) LongTensor of global representative indices
        ``member_to_rep_local`` : (N,) LongTensor in ``[0, K-1]``
        ``groups``              : list[LongTensor], member indices per group
        ``rep_count``           : int
        ``effective_keep_ratio``: float
        ``grid_shape``          : (grid_h, grid_w)
        ``conf_grid``           : (grid_h, grid_w) pooled confidence map
    """
    grid_h, grid_w = _normalize_grid_shape(token_grid_size)
    total_tokens = grid_h * grid_w
    rep_count = max(1, min(total_tokens, int(round(total_tokens * keep_ratio))))

    conf_grid = pool_confidence_to_grid(conf_map, (grid_h, grid_w))
    conf_flat = torch.from_numpy(conf_grid.reshape(-1)).float()

    _, sorted_idx = conf_flat.sort()
    rep_idx = torch.sort(sorted_idx[:rep_count].long()).values

    all_idx = torch.arange(total_tokens, dtype=torch.long)
    all_coords = torch.stack(
        [all_idx // grid_w, all_idx % grid_w],
        dim=1,
    ).float()
    rep_coords = all_coords.index_select(0, rep_idx)

    distances = torch.cdist(all_coords, rep_coords, p=2)
    member_to_rep_local = distances.argmin(dim=1).long()
    member_to_rep = rep_idx.index_select(0, member_to_rep_local)
    groups = [
        torch.where(member_to_rep_local == rep_local)[0]
        for rep_local in range(rep_count)
    ]

    return {
        "rep_idx": rep_idx,
        "member_to_rep": member_to_rep,
        "member_to_rep_local": member_to_rep_local,
        "groups": groups,
        "rep_count": rep_count,
        "effective_keep_ratio": rep_count / float(total_tokens),
        "grid_shape": (grid_h, grid_w),
        "conf_grid": conf_grid,
    }


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
