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

Confidence Map Derivation
-------------------------
After computing left and right disparity maps we run a left-right
consistency check (left_right_check_window).  A pixel is considered
*reliable* (high confidence) if it passes the check — i.e., it is
neither an occlusion nor a mismatch.  The raw binary reliability mask
is optionally smoothed with a Gaussian so that token-level pooling
produces gradual confidence gradients rather than hard edges.

Hardware Note
-------------
On the FPGA the same logic is implemented as a shift-register comparator
that runs one clock cycle after the disparity winner-take-all selection.
The confidence bit per pixel costs 1 bit of storage and maps directly to
the token router's threshold comparator.

Author : [Your Name]
Venue  : ICCAD 2025 (submission)
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# ---------------------------------------------------------------------------
# Path setup — ensure both project root and SGM package are importable
# regardless of the caller's CWD.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the Numba-JIT kernels from the existing SGM engine.
# First call will trigger Numba compilation (~30-90 s on first run).
from SGM.SGM import (
    compute_gradient,
    calculate_pixel_cost_all,
    aggregate_costs_0,
    aggregate_costs_135,
    compute_disparity,
    left_right_check_window,
    filling2,
)


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
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Standard deviation of the Gaussian used to smooth the binary
        L-R consistency mask into a soft confidence map.  Set to 0 to
        return the hard binary mask.
    verbose : bool
        Print timing and progress information.

    Returns
    -------
    disparity_norm : ndarray, shape (H, W), dtype float32
        Filled disparity map normalised to [0, 1] (0 = near, 1 = far
        for typical rectified cameras — verify for your calibration).
    confidence_map : ndarray, shape (H, W), dtype float32
        Per-pixel SGM reliability score in [0, 1].
        1.0  →  pixel passes L-R consistency check (SGM is reliable here).
        0.0  →  pixel is an occlusion or mismatch (uncertain).
    disparity_raw : ndarray, shape (H, W), dtype float32
        Raw filled disparity in pixels before normalisation.
    """
    assert disparity_range % 4 == 0, "disparity_range must be a multiple of 4."

    if verbose:
        print(f"[SGM] Loading images ...")
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
    # Stage 5: Left-right consistency check → confidence map
    # ------------------------------------------------------------------
    # occlusion : True  = pixel is behind an occluding surface (low conf)
    # mismatches: True  = L-R disparity disagreement     (low conf)
    if verbose:
        print("[SGM] Left-right consistency check ...")
    occlusion, mismatches, _ = left_right_check_window(disp_L, disp_R, 5)

    # Binary reliability mask: 1.0 where SGM is trustworthy
    reliable = (~occlusion & ~mismatches).astype(np.float32)

    # Stage 6: Hole filling on the left disparity map
    if verbose:
        print("[SGM] Filling holes ...")
    disp_filled = filling2(disp_L, disp_R, occlusion, mismatches)
    disp_filled = disp_filled.astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 7: Build smooth confidence map
    # ------------------------------------------------------------------
    # A hard binary mask creates token-grid artefacts after pooling.
    # Convolving with a Gaussian spreads the confidence signal so that
    # partially-reliable token regions get intermediate confidence values,
    # giving the threshold comparator a smooth operating point.
    if smooth_sigma > 0:
        confidence_map = gaussian_filter(reliable, sigma=smooth_sigma).clip(0.0, 1.0)
    else:
        confidence_map = reliable.copy()

    # ------------------------------------------------------------------
    # Stage 8: Normalise disparity to [0, 1]
    # ------------------------------------------------------------------
    d_max = disp_filled.max()
    disparity_norm = (disp_filled / d_max).astype(np.float32) if d_max > 0 else disp_filled

    if verbose:
        reliable_pct = 100.0 * reliable.mean()
        print(f"[SGM] Done.  Reliable pixels: {reliable_pct:.1f}%  "
              f"Max disparity: {d_max:.1f} px")

    return disparity_norm, confidence_map, disp_filled


def confidence_to_token_grid(
    confidence_map: np.ndarray,
    token_grid_size: int = 37,
) -> np.ndarray:
    """
    Average-pool the per-pixel confidence map to the ViT token grid.

    This mirrors the F.adaptive_avg_pool2d operation inside
    SGMConfidenceTokenRouter for quick NumPy-only visualisation.

    Parameters
    ----------
    confidence_map : (H, W) float32
    token_grid_size : int  — side length of the token grid (default: 37)

    Returns
    -------
    token_conf : (token_grid_size, token_grid_size) float32
    """
    H, W = confidence_map.shape
    ph = H // token_grid_size  # patch height
    pw = W // token_grid_size  # patch width
    if ph == 0 or pw == 0:
        # Fall-back: bilinear resize
        return cv2.resize(confidence_map, (token_grid_size, token_grid_size),
                          interpolation=cv2.INTER_AREA)

    # Crop to exact multiple, then reshape and average
    H_crop = ph * token_grid_size
    W_crop = pw * token_grid_size
    cropped = confidence_map[:H_crop, :W_crop]
    token_conf = (cropped
                  .reshape(token_grid_size, ph, token_grid_size, pw)
                  .mean(axis=(1, 3)))
    return token_conf.astype(np.float32)
