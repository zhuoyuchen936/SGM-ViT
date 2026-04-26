"""Minimal Phase 7 stub for `core.fusion_net`.

Historical note: this module used to host ~1,700 lines of Phase 1-6 fusion network
architectures (FusionResidualNet, MaskResidualLiteNet, DualBranchMaskedNet,
PyramidEdgeMaskedNet, DetailRestoreV2Net, RegionCalibratedRefineNet, DirectFusionNet,
and training/inference helpers). They were deprecated by Phase 7's `core.effvit_depth`
(a much smaller 4.85M-param EffViT-B1 network that surpasses all prior variants on all
four datasets — see memory-bank/progress.md Phase 7 Iter 1).

Only `compute_disp_scale` is still imported externally (by
`scripts/build_fusion_cache_v3.py`), so that is all we keep.
"""
from __future__ import annotations

import numpy as np


def compute_disp_scale(fused_base: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Return the 95th-percentile disparity as a per-sample normalization scale.

    Used by the v3 cache builder to store a consistent scale across caches,
    so downstream training can normalize disparity by per-sample scale.
    """
    fused_base = np.asarray(fused_base, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(fused_base) & (fused_base > 0)
    else:
        valid_mask = valid_mask & np.isfinite(fused_base) & (fused_base > 0)
    if int(valid_mask.sum()) == 0:
        return 1.0
    scale = float(np.percentile(fused_base[valid_mask], 95.0))
    return max(scale, 1.0)
