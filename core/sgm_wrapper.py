"""
core/sgm_wrapper.py
===================
Programmatic wrapper around the SGM engine that returns NumPy arrays
instead of writing images to disk.

The original SGM/SGM.py drives the full pipeline but only outputs PNG
files.  This wrapper re-uses the same Numba-JIT compiled kernels (imported
directly from SGM.SGM) and captures intermediate and final arrays so that
downstream Python code — in particular the SGMConfidenceTokenRouter — can
consume them without disk I/O.

Public API
----------
run_sgm_with_confidence(left_path, right_path, **kwargs)
    → disparity_norm : (H, W) float32  — disparity, normalised to [0, 1]
    → confidence_map : (H, W) float32  — per-pixel reliability in [0, 1]
    → disparity_raw  : (H, W) float32  — disparity in pixels

Confidence Map Derivation  (PKRN + LR-check masking)
------------------------------------------------------
Confidence is derived from the aggregated left cost volume using the
**Peak Ratio Naive (PKRN)** metric — a classical, hardware-friendly
stereo confidence measure:

    C = 1 − BestCost / SecondBestCost

where *SecondBestCost* is the minimum cost found at least ``pkrn_min_dist``
disparity steps away from the winner.  Intuition:

* If the second-best cost is close to the best (PKRN ≈ 0) the matching
  is *ambiguous* → low confidence.
* If the second-best cost is much larger (PKRN ≈ 1) the match is
  *unambiguous* → high confidence.

After computing PKRN we zero out pixels flagged as holes by the
left-right consistency check (occlusions and mismatches), because these
locations have no geometrically valid disparity regardless of cost-volume
shape.  The combined map is optionally smoothed with a Gaussian so that
token-level pooling produces gradual confidence gradients.

Hardware Note
-------------
PKRN is computed in a *single forward pass* over the left aggregated cost
volume, without needing the right disparity map.  This maps to a simple
comparator array that tracks the running minimum and second-minimum along
the disparity search direction — a natural fit for an FPGA DSP column.
The confidence value requires only one arithmetic division per pixel after
the WTA stage, costing approximately one multiplier and one DSP slice per
processing element.

Author : [Your Name]
Venue  : ICCAD 2025 (submission)
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import core._paths  # noqa: F401  — ensures project root + DA2 on sys.path

# Import the Numba-JIT kernels from the existing SGM engine.
# First call will trigger Numba compilation (~30-90 s on first run).
from SGM.SGM import (
    aggregate_costs_0,
    aggregate_costs_135,
    calculate_pixel_cost_all,
    compute_disparity,
    compute_gradient,
    filling2,
    left_right_check_window,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pkrn(cost_vol: np.ndarray, min_dist: int = 1) -> np.ndarray:
    """
    Peak Ratio Naive (PKRN) confidence from an aggregated SGM cost volume.

    PKRN is a classical, hardware-friendly stereo confidence metric.
    For each pixel the winner disparity ``d*`` is identified (lowest cost).
    The *second-best* cost is then the minimum cost over all disparities
    that are at least ``min_dist`` steps away from ``d*``:

        C = 1 − cost[d*] / cost[d_2nd]

    Interpretation
    --------------
    * C close to 0  →  best and second-best costs are similar  →  match is
                       ambiguous  →  *low confidence*.
    * C close to 1  →  second-best cost is much larger  →  match is
                       unambiguous  →  *high confidence*.

    Parameters
    ----------
    cost_vol : (H, W, D) float array
        Aggregated SGM cost volume.  Lower values indicate better matches.
        The disparity dimension is the last axis.
    min_dist : int
        Minimum disparity distance (exclusive) from the winner before a
        cost value is eligible as the second-best candidate.  A value of 1
        (default) excludes only the immediately adjacent disparities, which
        avoids the interpolation peak but is permissive.  Increase to 2-3
        for stricter uniqueness.

    Returns
    -------
    conf : (H, W) float32
        PKRN confidence in [0, 1].  Pixels where no valid second-best
        exists (e.g., D ≤ 2·min_dist) are assigned 0.
    """
    H, W, D = cost_vol.shape

    # Winner disparity index and its cost — both (H, W)
    best_idx  = cost_vol.argmin(axis=2).astype(np.int32)   # (H, W)
    best_cost = cost_vol.min(axis=2)                        # (H, W)

    # Build distance tensor |d - d*| for every candidate disparity d.
    # Shape broadcast: (1, 1, D) - (H, W, 1) → (H, W, D)
    d_range = np.arange(D, dtype=np.int32)
    dist = np.abs(d_range[None, None, :] - best_idx[:, :, None])  # (H, W, D)

    # Mask out the neighbourhood of the winner; replace with +inf
    masked_cost = np.where(dist > min_dist, cost_vol, np.inf)

    # Second-best cost: minimum over the remaining candidates
    second_best_cost = masked_cost.min(axis=2)              # (H, W)

    # Compute PKRN where a valid second-best exists and second_best > 0
    conf = np.zeros((H, W), dtype=np.float32)
    valid = np.isfinite(second_best_cost) & (second_best_cost > 0)
    conf[valid] = 1.0 - best_cost[valid] / second_best_cost[valid]

    return conf.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_sgm_with_confidence(
    left_path: str,
    right_path: str,
    disparity_range: int = 128,
    Grad: float = 32.0,
    Ctg: float = 16.0,
    Window_Size: int = 3,
    LARGE_PENALTY: float = 1.0,
    SMALL_PENALTY: float = 0.3,
    smooth_sigma: float = 5.0,
    pkrn_min_dist: int = 1,
    verbose: bool = True,
    return_debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Run the full SGM stereo-matching pipeline and return arrays.

    Parameters
    ----------
    left_path, right_path : str
        Paths to the rectified left / right stereo images (any format
        readable by OpenCV; images are converted to greyscale internally).
    disparity_range : int
        Maximum disparity to search in pixels.  Must be a multiple of 4
        (the SGM cost-volume is stored compressed 4:1).  Default: 128.
    Grad : float
        Gradient cost threshold for the exponential cost function.
    Ctg : float
        Census transform gain parameter.
    Window_Size : int
        Aggregation window size (cost computation).
    LARGE_PENALTY : float
        SGM penalty P2 for discontinuities larger than 1 pixel.
    SMALL_PENALTY : float
        SGM penalty P1 for 1-pixel disparity changes.
    smooth_sigma : float
        Standard deviation of the Gaussian used to smooth the PKRN
        confidence map.  Set to 0 to return the unsmoothed map.
    pkrn_min_dist : int
        Minimum disparity distance from the winning disparity before a
        cost value qualifies as the PKRN second-best.  Default: 1 (exclude
        only the immediately adjacent bin).  Increase to 2–3 for stricter
        uniqueness requirements.
    verbose : bool
        Print timing and progress information.
    return_debug : bool
        When True, also return a dictionary with intermediate SGM arrays
        needed for exporting ``sgm_hole`` assets.

    Returns
    -------
    disparity_norm : ndarray, shape (H, W), dtype float32
        Filled disparity map normalised to [0, 1] (0 = near, 1 = far
        for typical rectified cameras — verify for your calibration).
    confidence_map : ndarray, shape (H, W), dtype float32
        Per-pixel SGM reliability score in [0, 1], derived from PKRN
        on the aggregated left cost volume, masked by the LR-check:
        1.0  →  unambiguous cost-volume peak, passes LR-check.
        0.0  →  ambiguous match OR occlusion/mismatch flagged by LR-check.
    disparity_raw : ndarray, shape (H, W), dtype float32
        Raw filled disparity in pixels before normalisation.
    """
    assert disparity_range % 4 == 0, "disparity_range must be a multiple of 4."

    if verbose:
        print("[SGM] Loading images ...")
    left_gray  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left_gray is None or right_gray is None:
        raise FileNotFoundError(
            f"Cannot load stereo images:\n  left  → {left_path}\n  right → {right_path}"
        )
    H, W = left_gray.shape
    if verbose:
        print(f"[SGM] Image size: {W}×{H}  disparity_range={disparity_range}")

    # ------------------------------------------------------------------
    # Stage 1: Image pre-processing
    # ------------------------------------------------------------------
    left_blur  = cv2.blur(left_gray,  (3, 3))
    right_blur = cv2.blur(right_gray, (3, 3))

    left_grad_x,  left_grad_y,  grad_left  = compute_gradient(left_blur)
    right_grad_x, right_grad_y, grad_right = compute_gradient(right_blur)

    # ------------------------------------------------------------------
    # Stage 2: Cost volume computation (Census + gradient hybrid)
    # NOTE: Numba JIT compilation happens on the *first* call.
    #       Expect 30-90 s overhead on the first run; subsequent calls
    #       use the cached compiled kernels.
    # ------------------------------------------------------------------
    if verbose:
        print("[SGM] Computing cost volumes (Numba JIT — slow on first run) ...")
    t0 = time.perf_counter()

    idx_L, cost_L, idx_R, cost_R = calculate_pixel_cost_all(
        grad_left, grad_right,
        left_grad_x, left_grad_y,
        right_grad_x, right_grad_y,
        disparity_range, Grad, Ctg, Window_Size,
    )

    if verbose:
        print(f"[SGM] Cost volume done in {time.perf_counter()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage 3: Semi-global cost aggregation (2 directions)
    # ------------------------------------------------------------------
    if verbose:
        print("[SGM] Aggregating costs (left to right + diagonal) ...")
    t1 = time.perf_counter()

    agg0_L   = aggregate_costs_0(H, W, disparity_range, cost_L, LARGE_PENALTY, SMALL_PENALTY)
    agg135_L = aggregate_costs_135(H, W, disparity_range, cost_L, LARGE_PENALTY, SMALL_PENALTY)
    agg0_R   = aggregate_costs_0(H, W, disparity_range, cost_R, LARGE_PENALTY, SMALL_PENALTY)
    agg135_R = aggregate_costs_135(H, W, disparity_range, cost_R, LARGE_PENALTY, SMALL_PENALTY)

    agg_L = agg0_L + agg135_L
    agg_R = agg0_R + agg135_R

    if verbose:
        print(f"[SGM] Aggregation done in {time.perf_counter()-t1:.1f}s")

    # ------------------------------------------------------------------
    # Stage 4: Winner-Take-All disparity selection
    # ------------------------------------------------------------------
    if verbose:
        print("[SGM] Computing disparity maps (WTA) ...")
    disp_L = compute_disparity(agg_L, H, W, disparity_range, idx_L)
    disp_R = compute_disparity(agg_R, H, W, disparity_range, idx_R)

    # ------------------------------------------------------------------
    # Stage 4b: PKRN confidence from the aggregated left cost volume
    # ------------------------------------------------------------------
    # Computed immediately after WTA so agg_L is still in memory.
    # agg_L shape: (H, W, disparity_range)  — lower cost = better match.
    # _pkrn() is a single NumPy pass over the disparity axis; no right
    # disparity map is required (FPGA-friendly datapath).
    if verbose:
        print("[SGM] Computing PKRN confidence ...")
    pkrn_conf = _pkrn(agg_L, min_dist=pkrn_min_dist)  # (H, W) ∈ [0, 1]

    # ------------------------------------------------------------------
    # Stage 5: Left-right consistency check → hole mask
    # ------------------------------------------------------------------
    # occlusion : True  = pixel is behind an occluding surface (no valid match)
    # mismatches: True  = L-R disparity disagreement            (unreliable)
    # These pixels receive confidence = 0 regardless of PKRN value.
    if verbose:
        print("[SGM] Left-right consistency check ...")
    occlusion, mismatches, _ = left_right_check_window(disp_L, disp_R, 5)

    # Zero PKRN at LR-check holes → combined confidence
    hole_mask = occlusion | mismatches                              # True = bad pixel
    raw_conf  = pkrn_conf * (~hole_mask).astype(np.float32)       # (H, W) ∈ [0, 1]

    # Stage 6: Hole filling on the left disparity map
    if verbose:
        print("[SGM] Filling holes ...")
    disp_filled = filling2(disp_L, disp_R, occlusion, mismatches)
    disp_filled = disp_filled.astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 7: Build smooth confidence map
    # ------------------------------------------------------------------
    # Gaussian smoothing converts the per-pixel PKRN signal into gradual
    # token-level gradients, giving the threshold comparator a smoother
    # operating point than a hard binary mask.
    if smooth_sigma > 0:
        confidence_map = gaussian_filter(raw_conf, sigma=smooth_sigma).clip(0.0, 1.0)
    else:
        confidence_map = raw_conf

    # ------------------------------------------------------------------
    # Stage 8: Normalise disparity to [0, 1]
    # ------------------------------------------------------------------
    d_max = disp_filled.max()
    disparity_norm = (disp_filled / d_max).astype(np.float32) if d_max > 0 else disp_filled

    if verbose:
        valid_pct = 100.0 * (~hole_mask).mean()
        pkrn_mean = pkrn_conf[~hole_mask].mean() if (~hole_mask).any() else 0.0
        print(f"[SGM] Done.  LR-check pass: {valid_pct:.1f}%  "
              f"Mean PKRN (valid): {pkrn_mean:.3f}  "
              f"Max disparity: {d_max:.1f} px")

    if not return_debug:
        return disparity_norm, confidence_map, disp_filled

    debug = {
        "disp_left": disp_L.astype(np.float32),
        "disp_right": disp_R.astype(np.float32),
        "disp_filled": disp_filled.astype(np.float32),
        "occlusion": occlusion.astype(bool),
        "mismatches": mismatches.astype(bool),
        "pkrn_raw": pkrn_conf.astype(np.float32),
        "confidence_raw": raw_conf.astype(np.float32),
    }
    return disparity_norm, confidence_map, disp_filled, debug


def confidence_to_token_grid(
    confidence_map: np.ndarray,
    token_grid_size: int = 37,
) -> np.ndarray:
    """
    Average-pool the per-pixel confidence map to the ViT token grid.

    Delegates to ``core.eval_utils.pool_confidence`` which uses
    ``F.adaptive_avg_pool2d`` — the same operation used inside
    ``SGMConfidenceTokenRouter`` for exact consistency.

    Parameters
    ----------
    confidence_map : (H, W) float32
    token_grid_size : int  — side length of the token grid (default: 37)

    Returns
    -------
    token_conf : (token_grid_size, token_grid_size) float32
    """
    from .eval_utils import pool_confidence
    return pool_confidence(confidence_map, token_grid_size)
