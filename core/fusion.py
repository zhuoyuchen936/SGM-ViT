"""Confidence-guided SGM + DA2 fusion strategies.

Four pixel-level fusion functions that combine SGM stereo disparity with
DepthAnythingV2 monocular depth, gated by the SGM confidence map.
A dispatcher (``fuse_dispatch``) routes to the correct function by name.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares as _scipy_least_squares


DEFAULT_FLAT_DETAIL_THRESHOLD = 0.20


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gradient_energy(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _compute_detail_guidance(
    sgm_smooth: np.ndarray,
    da2_aligned: np.ndarray,
    image_bgr: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if image_bgr is None:
        gray = np.zeros_like(sgm_smooth, dtype=np.float32)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    g_img = _normalize_map(_gradient_energy(gray))
    g_da2 = _normalize_map(_gradient_energy(da2_aligned))
    g_sgm = _normalize_map(_gradient_energy(sgm_smooth))
    edge_adv = np.maximum(g_da2 - g_sgm, 0.0).astype(np.float32)
    disp_disagree = _normalize_map(np.abs(da2_aligned - sgm_smooth))

    detail_score = np.clip(
        0.45 * g_da2
        + 0.20 * g_img
        + 0.20 * edge_adv
        + 0.15 * disp_disagree,
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "detail_score": detail_score,
        "g_img": g_img.astype(np.float32),
        "g_da2": g_da2.astype(np.float32),
        "g_sgm": g_sgm.astype(np.float32),
        "edge_adv": edge_adv.astype(np.float32),
        "disp_disagree": disp_disagree.astype(np.float32),
    }


def build_flat_region_mask(
    detail_score: np.ndarray,
    threshold: float = DEFAULT_FLAT_DETAIL_THRESHOLD,
) -> np.ndarray:
    detail_score = np.asarray(detail_score, dtype=np.float32)
    return (detail_score <= float(threshold)).astype(np.float32)


def build_region_stable_confidence(
    confidence_map: np.ndarray,
    flat_region_mask: np.ndarray,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 0.12,
    bilateral_sigma_space: float = 7.0,
    flat_gaussian_sigma: float = 5.0,
) -> np.ndarray:
    conf = np.clip(np.asarray(confidence_map, dtype=np.float32), 0.0, 1.0)
    flat_mask = np.asarray(flat_region_mask, dtype=np.float32) > 0.5

    conf_bilateral = cv2.bilateralFilter(
        conf,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    ).astype(np.float32)
    conf_gaussian = cv2.GaussianBlur(
        conf_bilateral,
        (0, 0),
        sigmaX=flat_gaussian_sigma,
        sigmaY=flat_gaussian_sigma,
    ).astype(np.float32)
    conf_median = cv2.medianBlur(
        np.clip(conf_bilateral * 255.0, 0.0, 255.0).astype(np.uint8),
        5,
    ).astype(np.float32) / 255.0

    conf_region = conf_bilateral.copy()
    conf_region[flat_mask] = (
        0.55 * conf_gaussian[flat_mask] + 0.45 * conf_median[flat_mask]
    ).astype(np.float32)
    return np.clip(conf_region, 0.0, 1.0).astype(np.float32)


def build_region_stable_base(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    theta_low: float = 0.15,
    theta_high: float = 0.55,
    lowpass_sigma: float = 2.0,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 4.0,
    bilateral_sigma_space: float = 3.0,
    flat_detail_threshold: float = DEFAULT_FLAT_DETAIL_THRESHOLD,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    sgm_disp = np.asarray(sgm_disp, dtype=np.float32)
    da2_aligned = np.asarray(da2_aligned, dtype=np.float32)
    confidence_map = np.asarray(confidence_map, dtype=np.float32)

    sgm_valid = (sgm_disp > 0).astype(np.float32)
    sgm_low = cv2.bilateralFilter(
        sgm_disp,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    ).astype(np.float32)
    sgm_low = np.where(sgm_valid > 0, sgm_low, 0.0).astype(np.float32)
    mono_low = cv2.GaussianBlur(
        da2_aligned.astype(np.float32),
        (0, 0),
        sigmaX=lowpass_sigma,
        sigmaY=lowpass_sigma,
    ).astype(np.float32)

    detail_debug = _compute_detail_guidance(
        sgm_smooth=sgm_low,
        da2_aligned=da2_aligned,
        image_bgr=image_bgr,
    )
    detail_score = detail_debug["detail_score"]
    flat_region_mask = build_flat_region_mask(
        detail_score=detail_score,
        threshold=flat_detail_threshold,
    )
    conf_region = build_region_stable_confidence(
        confidence_map=confidence_map,
        flat_region_mask=flat_region_mask,
    )
    span = max(theta_high - theta_low, 1e-6)
    alpha_region = sgm_valid * np.clip((conf_region - theta_low) / span, 0.0, 1.0)
    absolute_base = alpha_region * sgm_low + (1.0 - alpha_region) * mono_low
    absolute_base = np.where(sgm_valid > 0, absolute_base, mono_low).astype(np.float32)

    if not return_debug:
        return absolute_base.astype(np.float32)

    debug = {
        "absolute_base": absolute_base.astype(np.float32),
        "sgm_low": sgm_low.astype(np.float32),
        "mono_low": mono_low.astype(np.float32),
        "conf_region": conf_region.astype(np.float32),
        "alpha_region": alpha_region.astype(np.float32),
        "flat_region_mask": flat_region_mask.astype(np.float32),
        **detail_debug,
    }
    return absolute_base.astype(np.float32), debug


def fuse_region_stable_detail(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    theta_low: float = 0.15,
    theta_high: float = 0.55,
    residual_gain: float = 0.90,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 4.0,
    bilateral_sigma_space: float = 3.0,
    lowpass_sigma: float = 2.0,
    flat_detail_threshold: float = DEFAULT_FLAT_DETAIL_THRESHOLD,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    absolute_base, debug = build_region_stable_base(
        sgm_disp=sgm_disp,
        da2_aligned=da2_aligned,
        confidence_map=confidence_map,
        image_bgr=image_bgr,
        theta_low=theta_low,
        theta_high=theta_high,
        lowpass_sigma=lowpass_sigma,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        flat_detail_threshold=flat_detail_threshold,
        return_debug=True,
    )
    mono_low = debug["mono_low"]
    mono_high = (np.asarray(da2_aligned, dtype=np.float32) - mono_low).astype(np.float32)
    detail_term = (
        float(residual_gain) * debug["detail_score"] * mono_high
    ).astype(np.float32)
    fused = np.clip(absolute_base + detail_term, 0.0, None).astype(np.float32)

    if not return_debug:
        return fused

    debug.update(
        {
            "mono_high": mono_high.astype(np.float32),
            "detail_term": detail_term.astype(np.float32),
            "fused_v2": fused.astype(np.float32),
        }
    )
    return fused, debug


def fuse_sgm_da2(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """
    Pixel-level SGM + DA2 fusion.

    Decision rule per pixel
    -----------------------
    * SGM confident  (sgm_disp > 0  AND  conf >= threshold) -> use SGM directly.
      SGM is geometrically accurate in these regions (sv D1 ~ 8%).
    * SGM uncertain / hole (sgm_disp = 0  OR  conf < threshold) -> use DA2.
      DA2 provides semantic coverage where SGM fails.

    A confidence-weighted soft blend is applied instead of a hard threshold
    to avoid discontinuities at region boundaries:

        alpha  = clip(conf / threshold, 0, 1)  if sgm_disp > 0  else 0
        fused  = alpha * sgm_disp + (1 - alpha) * da2_aligned

    When conf >= threshold the alpha saturates at 1 and SGM is used exclusively.
    When conf = 0 or sgm_disp = 0 the DA2 prediction is used exclusively.

    Parameters
    ----------
    sgm_disp       : (H, W) float32  raw SGM disparity; 0 = hole / invalid
    da2_aligned    : (H, W) float32  DA2 depth aligned to SGM disparity space
    confidence_map : (H, W) float32  per-pixel SGM confidence [0, 1]
    conf_threshold : float           confidence level at which alpha saturates to 1

    Returns
    -------
    fused : (H, W) float32  fused disparity map in SGM pixel units
    """
    sgm_valid = (sgm_disp > 0).astype(np.float32)
    # alpha in [0,1]: 1 = full SGM, 0 = full DA2
    alpha = sgm_valid * np.clip(confidence_map / max(conf_threshold, 1e-6), 0.0, 1.0)
    fused = alpha * sgm_disp + (1.0 - alpha) * da2_aligned
    return fused.clip(0.0, None).astype(np.float32)


def fuse_hard_switch(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """
    Binary per-pixel selector: use SGM where valid AND confident, DA2 elsewhere.

    Decision rule:
        if sgm_disp > 0 AND conf >= conf_threshold -> use SGM
        else -> use DA2

    Simplest FPGA mapping: 1 comparator + 1 mux per pixel.
    """
    use_sgm = (sgm_disp > 0) & (confidence_map >= conf_threshold)
    fused = np.where(use_sgm, sgm_disp, da2_aligned)
    return fused.clip(0.0, None).astype(np.float32)


def fuse_outlier_aware(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    conf_threshold: float = 0.5,
    outlier_threshold: float = 10.0,
) -> np.ndarray:
    """
    Soft blend with outlier attenuation.

    Catches SGM wrong matches (repetitive texture / thin objects) that pass
    the LR-check but have high |SGM - DA2| disagreement:

        alpha_base      = clip(conf / theta, 0, 1) * sgm_valid
        discrepancy     = |sgm_disp - da2_aligned|
        outlier_factor  = clip(1 - discrepancy / outlier_threshold, 0, 1)
        alpha           = alpha_base * outlier_factor
        fused           = alpha * SGM + (1 - alpha) * DA2

    FPGA cost: 1 DSP + 3 LUTs per pixel.
    """
    sgm_valid = (sgm_disp > 0).astype(np.float32)
    alpha_base = sgm_valid * np.clip(confidence_map / max(conf_threshold, 1e-6), 0.0, 1.0)
    discrepancy = np.abs(sgm_disp - da2_aligned)
    outlier_factor = np.clip(1.0 - discrepancy / max(outlier_threshold, 1e-6), 0.0, 1.0)
    alpha = alpha_base * outlier_factor
    fused = alpha * sgm_disp + (1.0 - alpha) * da2_aligned
    return fused.clip(0.0, None).astype(np.float32)


def fuse_two_threshold(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    theta_low: float = 0.3,
    theta_high: float = 0.7,
) -> np.ndarray:
    """
    Dead-zone cascade with two confidence thresholds.

        conf < theta_low            -> pure DA2  (alpha = 0)
        theta_low <= conf < theta_high   -> linear blend:  alpha = (conf - theta_low) / (theta_high - theta_low)
        conf >= theta_high           -> pure SGM  (alpha = 1)

    Prevents very-low-confidence SGM from leaking into the output.
    FPGA cost: 2 comparators + 1 DSP per pixel.  Reciprocal
    1 / (theta_high - theta_low) is a compile-time constant.
    """
    sgm_valid = (sgm_disp > 0).astype(np.float32)
    span = max(theta_high - theta_low, 1e-6)
    alpha = np.clip((confidence_map - theta_low) / span, 0.0, 1.0)
    alpha = alpha * sgm_valid
    fused = alpha * sgm_disp + (1.0 - alpha) * da2_aligned
    return fused.clip(0.0, None).astype(np.float32)


def fuse_edge_aware_residual(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    theta_low: float = 0.15,
    theta_high: float = 0.55,
    detail_suppression: float = 0.75,
    residual_gain: float = 0.90,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 4.0,
    bilateral_sigma_space: float = 3.0,
    lowpass_sigma: float = 2.0,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Edge-aware residual fusion.

    SGM supplies stable low-frequency geometry, while DA2 restores
    edge/detail components in regions where SGM is likely to oversmooth.
    """
    sgm_disp = np.asarray(sgm_disp, dtype=np.float32)
    da2_aligned = np.asarray(da2_aligned, dtype=np.float32)
    confidence_map = np.asarray(confidence_map, dtype=np.float32)

    sgm_valid = (sgm_disp > 0).astype(np.float32)
    sgm_smooth = cv2.bilateralFilter(
        sgm_disp,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    ).astype(np.float32)
    sgm_smooth = np.where(sgm_valid > 0, sgm_smooth, 0.0).astype(np.float32)

    span = max(theta_high - theta_low, 1e-6)
    alpha_conf = sgm_valid * np.clip((confidence_map - theta_low) / span, 0.0, 1.0)

    detail_debug = _compute_detail_guidance(
        sgm_smooth=sgm_smooth,
        da2_aligned=da2_aligned,
        image_bgr=image_bgr,
    )
    detail_score = detail_debug["detail_score"]
    alpha_eff = alpha_conf * (1.0 - detail_suppression * detail_score)

    da2_low = cv2.GaussianBlur(
        da2_aligned.astype(np.float32),
        (0, 0),
        sigmaX=lowpass_sigma,
        sigmaY=lowpass_sigma,
    ).astype(np.float32)
    da2_high = (da2_aligned - da2_low).astype(np.float32)

    fused_base = alpha_eff * sgm_smooth + (1.0 - alpha_eff) * da2_low
    fused = np.clip(fused_base + residual_gain * detail_score * da2_high, 0.0, None).astype(np.float32)

    if not return_debug:
        return fused

    debug = {
        "sgm_smooth": sgm_smooth,
        "alpha_conf": alpha_conf.astype(np.float32),
        "alpha_eff": alpha_eff.astype(np.float32),
        "detail_score": detail_score.astype(np.float32),
        "da2_low": da2_low.astype(np.float32),
        "da2_high": da2_high.astype(np.float32),
        "fused_base": fused_base.astype(np.float32),
        **detail_debug,
    }
    return fused, debug


# ---------------------------------------------------------------------------
# Fusion dispatcher
# ---------------------------------------------------------------------------


def compute_detail_guidance(
    da2_aligned: np.ndarray,
    sgm_disp: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    detail_suppression: float = 0.75,
) -> np.ndarray:
    """Compute detail/edge guidance score [0,1]."""
    g_da2 = _compute_gradient_magnitude(da2_aligned.astype(np.float32))
    g_da2 = g_da2 / max(np.percentile(g_da2, 95), 1e-6)
    g_da2 = np.clip(g_da2, 0, 1)

    sgm_valid = (sgm_disp > 0)
    g_sgm = _compute_gradient_magnitude(sgm_disp.astype(np.float32))
    g_sgm = g_sgm / max(np.percentile(g_sgm[sgm_valid], 95) if sgm_valid.any() else 1.0, 1e-6)
    g_sgm = np.clip(g_sgm, 0, 1)

    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) if image_bgr.ndim == 3 else image_bgr.astype(np.float32)
        g_img = _compute_gradient_magnitude(gray)
        g_img = g_img / max(np.percentile(g_img, 95), 1e-6)
        g_img = np.clip(g_img, 0, 1)
    else:
        g_img = np.zeros_like(g_da2)

    edge_adv = np.clip(g_da2 - g_sgm, 0, 1)
    disp_disagree = np.abs(da2_aligned.astype(np.float32) - sgm_disp.astype(np.float32))
    disp_disagree = disp_disagree / max(np.percentile(disp_disagree[sgm_valid], 95) if sgm_valid.any() else 1.0, 1e-6)
    disp_disagree = np.clip(disp_disagree, 0, 1) * sgm_valid.astype(np.float32)

    detail_score = 0.45 * g_da2 + 0.20 * g_img + 0.20 * edge_adv + 0.15 * disp_disagree
    return np.clip(detail_score, 0, 1).astype(np.float32)



# ---- Region-Calibrated Fusion (RCF) ----

def _compute_gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude of a 2D array using Sobel."""
    gx = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def segment_mono_regions(
    mono_disp_aligned: np.ndarray,
    image_bgr: np.ndarray | None = None,
    lowpass_sigma: float = 4.0,
    grad_threshold_quantile: float = 0.65,
    morph_close_size: int = 11,
    morph_open_size: int = 5,
    min_region_pixels: int = 500,
    image_edge_weight: float = 0.3,
) -> np.ndarray:
    """Segment mono disparity into coherent regions using gradient-based boundaries.

    Instead of quantizing disparity (which creates striping on slopes), this uses
    the gradient magnitude of smoothed mono as the boundary signal. Low-gradient
    areas are interior; high-gradient areas are boundaries. Morphological operations
    create large coherent regions even on slanted surfaces like ground planes.

    Returns region_labels (H, W) int32.
    """
    from scipy.ndimage import distance_transform_edt, label as scipy_label

    h, w = mono_disp_aligned.shape
    mono_f = mono_disp_aligned.astype(np.float32)

    # Smooth mono to suppress noise while preserving depth edges
    mono_smooth = cv2.GaussianBlur(mono_f, (0, 0), lowpass_sigma)

    # Gradient magnitude — high = depth boundary, low = interior of surface
    grad = _compute_gradient_magnitude(mono_smooth)

    # Combine with image gradient to respect color edges (often = depth edges)
    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) if image_bgr.ndim == 3 else image_bgr.astype(np.float32)
        grad_img = _compute_gradient_magnitude(gray)
        # Normalize both to [0,1]
        grad_norm = grad / max(np.percentile(grad, 98), 1e-6)
        grad_img_norm = grad_img / max(np.percentile(grad_img, 98), 1e-6)
        combined_grad = (1.0 - image_edge_weight) * grad_norm + image_edge_weight * grad_img_norm
    else:
        combined_grad = grad / max(np.percentile(grad, 98), 1e-6)

    combined_grad = np.clip(combined_grad, 0, 1)

    # Threshold: low gradient = interior
    tau = np.quantile(combined_grad[combined_grad > 0], grad_threshold_quantile)
    interior = (combined_grad < tau).astype(np.uint8)

    # Morphological close: fill small gaps in interior (bridge across texture noise)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size))
    interior = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, kernel_close)

    # Morphological open: remove small isolated interior fragments
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size))
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel_open)

    # Connected components on interior
    n_labels, labels = cv2.connectedComponents(interior, connectivity=8)

    # Remove small regions
    for lbl in range(1, n_labels):
        if (labels == lbl).sum() < min_region_pixels:
            labels[labels == lbl] = 0

    # Re-number labels compactly
    unique = np.unique(labels)
    unique = unique[unique > 0]
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    for i, lbl in enumerate(unique, 1):
        remap[lbl] = i
    labels = remap[labels]

    # Assign unlabeled pixels to nearest region
    unlabeled = (labels == 0)
    if unlabeled.any() and (labels > 0).any():
        _, nearest_idx = distance_transform_edt(unlabeled, return_distances=True, return_indices=True)
        labels[unlabeled] = labels[nearest_idx[0][unlabeled], nearest_idx[1][unlabeled]]

    return labels.astype(np.int32)


def calibrate_regions(
    mono_disp_aligned: np.ndarray,
    sgm_disp: np.ndarray,
    confidence_map: np.ndarray,
    region_labels: np.ndarray,
    conf_threshold: float = 0.5,
    n_min_robust: int = 30,
    n_min_few: int = 5,
    lowpass_sigma: float = 2.0,
    huber_scale: float = 3.0,
    outlier_sigma: float = 3.0,
    rmse_tau: float = 2.0,
    max_offset_abs: float = 6.0,
    max_offset_rel: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-region affine calibration of mono disparity using robust Huber regression.

    For each region:
      1. Fit s_k * mono_low + t_k ~= sgm using Huber-loss least-squares (resistant
         to biased SGM samples from occlusions passing confidence threshold)
      2. Reject outliers beyond outlier_sigma * MAD, refit on clean subset
      3. Gate region confidence by calibration RMSE: regions with rmse > rmse_tau
         get proportionally reduced region_confidence

    This avoids the previous failure mode where a handful of biased SGM samples
    (e.g., occluded pixels with high apparent confidence) skewed the entire
    region's affine parameters, causing large structural errors like the
    ETH3D box-top dark spot.

    Returns (calibrated_mono, offset_map, region_confidence_map).
    """
    h, w = mono_disp_aligned.shape
    mono_low = cv2.GaussianBlur(mono_disp_aligned.astype(np.float32), (0, 0), lowpass_sigma)
    mono_high = mono_disp_aligned.astype(np.float32) - mono_low

    sgm_valid = (sgm_disp > 0).astype(bool)
    conf_valid = (confidence_map >= conf_threshold) & sgm_valid

    calibrated_low = mono_low.copy()
    offset_map = np.zeros((h, w), dtype=np.float32)
    region_conf_map = np.zeros((h, w), dtype=np.float32)

    unique_labels = np.unique(region_labels)
    unique_labels = unique_labels[unique_labels > 0]

    region_params = {}  # lbl -> (scale, shift, confidence)

    def _huber_affine_fit(m_vals: np.ndarray, s_vals: np.ndarray,
                          c_vals: np.ndarray) -> tuple[float, float, float]:
        """Huber-robust affine fit with outlier rejection. Returns (scale, shift, rmse)."""
        if m_vals.size < 3:
            return 1.0, 0.0, float("inf")

        # Warm start via weighted L2
        A = np.column_stack([m_vals, np.ones_like(m_vals)]).astype(np.float64)
        w_sqrt = np.sqrt(np.clip(c_vals, 1e-3, None))
        Aw = A * w_sqrt[:, None]
        sw = s_vals.astype(np.float64) * w_sqrt
        try:
            coef0, *_ = np.linalg.lstsq(Aw, sw, rcond=None)
        except np.linalg.LinAlgError:
            coef0 = np.array([1.0, 0.0])

        # Huber refinement (weighted residuals)
        def _residuals(coef):
            return (coef[0] * m_vals + coef[1] - s_vals) * np.clip(c_vals, 1e-3, None)

        try:
            result = _scipy_least_squares(_residuals, coef0, loss="huber", f_scale=huber_scale)
            scale, shift = float(result.x[0]), float(result.x[1])
        except Exception:
            scale, shift = float(coef0[0]), float(coef0[1])

        # Outlier rejection via MAD on unweighted residuals
        unweighted = scale * m_vals + shift - s_vals
        mad = np.median(np.abs(unweighted - np.median(unweighted))) * 1.4826
        if mad > 1e-6 and m_vals.size >= 10:
            inlier = np.abs(unweighted) <= outlier_sigma * max(mad, 0.5)
            if inlier.sum() >= max(5, int(0.5 * m_vals.size)):
                # Refit on inliers
                m2, s2, c2 = m_vals[inlier], s_vals[inlier], c_vals[inlier]
                Aw2 = np.column_stack([m2, np.ones_like(m2)]).astype(np.float64) * np.sqrt(c2)[:, None]
                sw2 = s2.astype(np.float64) * np.sqrt(c2)
                try:
                    coef1, *_ = np.linalg.lstsq(Aw2, sw2, rcond=None)
                    result2 = _scipy_least_squares(
                        lambda cc: (cc[0] * m2 + cc[1] - s2) * np.sqrt(c2),
                        coef1,
                        loss="huber",
                        f_scale=huber_scale,
                    )
                    scale, shift = float(result2.x[0]), float(result2.x[1])
                    unweighted = scale * m2 + shift - s2
                except Exception:
                    pass

        # Clamp scale to reasonable range (global alignment already handles bulk scale)
        scale = float(np.clip(scale, 0.8, 1.2))
        rmse = float(np.sqrt(np.mean(unweighted ** 2))) if unweighted.size > 0 else float("inf")
        return scale, shift, rmse

    for lbl in unique_labels:
        region_mask = (region_labels == lbl)
        overlap = region_mask & conf_valid
        n_k = int(overlap.sum())

        if n_k >= n_min_robust:
            m_vals = mono_low[overlap].astype(np.float32)
            s_vals = sgm_disp[overlap].astype(np.float32)
            c_vals = confidence_map[overlap].astype(np.float32)

            scale_k, shift_k, rmse = _huber_affine_fit(m_vals, s_vals, c_vals)

            # Gate region confidence by RMSE — high-residual fits are untrusted
            rmse_conf = float(np.clip(rmse_tau / max(rmse, 1e-3), 0.0, 1.0))
            sample_conf = min(n_k / float(n_min_robust), 1.0)
            w_k = rmse_conf * sample_conf

            # If gate is near-zero, skip calibration entirely (use mono as-is)
            if w_k < 0.1:
                region_conf_map[region_mask] = 0.0
                region_params[lbl] = (1.0, 0.0, 0.0)
                continue

            mono_region = mono_low[region_mask]
            fitted = scale_k * mono_region + shift_k

            # Sanity check: reject calibrations that shift disparity too much.
            # Consistent bias scenario (e.g., ETH3D box top where all SGM samples
            # are uniformly wrong due to cross-surface mismatches) produces a fit
            # that passes Huber + outlier rejection but shifts the whole region
            # to a wrong depth. Cap the allowed offset by both absolute and
            # relative bounds tied to the region's own mono scale.
            mono_scale = max(float(mono_region.mean()), 1.0)
            max_offset = max(max_offset_abs, max_offset_rel * mono_scale)
            offset_peak = float(np.max(np.abs(fitted - mono_region)))
            if offset_peak > max_offset:
                # Scale the fit to respect the bound: compute safe (s', t') that
                # keeps |s'*m + t' - m| <= max_offset at the peak. Simplest:
                # downweight the correction, but keep sample confidence.
                # If way beyond bound, fall back to mono but keep region marked
                # so downstream detail path is still gated.
                shrink = max_offset / offset_peak
                if shrink < 0.15:
                    # Calibration untrustworthy: use mono, but mark region_conf low
                    region_conf_map[region_mask] = 0.1
                    region_params[lbl] = (1.0, 0.0, 0.1)
                    continue
                # Blend: w_k is unchanged but fitted shrinks toward mono
                fitted = shrink * fitted + (1.0 - shrink) * mono_region
                offset_peak = offset_peak * shrink

            # Apply partial calibration: blend between identity and fitted affine
            # according to confidence (preserve mono when gate is weak)
            calibrated_low[region_mask] = w_k * fitted + (1.0 - w_k) * mono_region
            offset_map[region_mask] = calibrated_low[region_mask] - mono_region
            region_conf_map[region_mask] = w_k
            region_params[lbl] = (scale_k, shift_k, w_k)

        elif n_k >= n_min_few:
            # Shift-only with Huber-robust median residual
            m_vals = mono_low[overlap].astype(np.float32)
            s_vals = sgm_disp[overlap].astype(np.float32)
            c_vals = confidence_map[overlap].astype(np.float32)
            residuals = s_vals - m_vals
            # Weighted median via sorting
            order = np.argsort(residuals)
            cw = np.cumsum(c_vals[order])
            cw_norm = cw / max(cw[-1], 1e-6)
            med_idx = np.searchsorted(cw_norm, 0.5)
            med_idx = min(med_idx, len(order) - 1)
            shift_k = float(residuals[order[med_idx]])

            mad = np.median(np.abs(residuals - shift_k)) * 1.4826
            rmse = float(mad)
            rmse_conf = float(np.clip(rmse_tau / max(rmse, 1e-3), 0.0, 1.0))
            sample_conf = n_k / float(n_min_robust)
            w_k = rmse_conf * sample_conf

            if w_k < 0.1:
                region_conf_map[region_mask] = 0.0
                region_params[lbl] = (1.0, 0.0, 0.0)
                continue

            # Sanity check on shift magnitude
            mono_scale = max(float(mono_low[region_mask].mean()), 1.0)
            max_offset = max(max_offset_abs, max_offset_rel * mono_scale)
            if abs(w_k * shift_k) > max_offset:
                overshoot_ratio = max_offset / abs(w_k * shift_k)
                if overshoot_ratio < 0.3:
                    region_conf_map[region_mask] = 0.0
                    region_params[lbl] = (1.0, 0.0, 0.0)
                    continue
                w_k = w_k * overshoot_ratio

            calibrated_low[region_mask] = mono_low[region_mask] + w_k * shift_k
            offset_map[region_mask] = w_k * shift_k
            region_conf_map[region_mask] = w_k
            region_params[lbl] = (1.0, w_k * shift_k, w_k)

        else:
            region_conf_map[region_mask] = 0.0
            region_params[lbl] = (1.0, 0.0, 0.0)

    # Propagate calibration to uncalibrated regions from neighbors
    for lbl in unique_labels:
        if region_params[lbl][2] > 0:
            continue
        region_mask = (region_labels == lbl)
        dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        neighbor_labels = np.unique(region_labels[dilated.astype(bool) & ~region_mask])
        neighbor_labels = neighbor_labels[neighbor_labels > 0]

        best_conf = 0.0
        best_params = (1.0, 0.0)
        mean_mono = mono_low[region_mask].mean()
        for nb in neighbor_labels:
            if nb in region_params and region_params[nb][2] > best_conf:
                nb_mean = mono_low[region_labels == nb].mean()
                if abs(nb_mean - mean_mono) < 5.0:
                    best_conf = region_params[nb][2]
                    best_params = (region_params[nb][0], region_params[nb][1])

        if best_conf > 0:
            s, t = best_params
            fitted = s * mono_low[region_mask] + t
            # Propagation is at half-confidence to avoid over-extension
            w_prop = 0.5 * best_conf
            calibrated_low[region_mask] = w_prop * fitted + (1.0 - w_prop) * mono_low[region_mask]
            offset_map[region_mask] = calibrated_low[region_mask] - mono_low[region_mask]
            region_conf_map[region_mask] = w_prop

    # Reconstruct: calibrated_low + original mono_high
    calibrated_mono = np.clip(calibrated_low + mono_high, 0, None).astype(np.float32)
    return calibrated_mono, offset_map, region_conf_map


def blend_region_boundaries(
    calibrated_mono: np.ndarray,
    region_labels: np.ndarray,
    detail_score: np.ndarray | None = None,
    image_bgr: np.ndarray | None = None,
    feather_radius: int = 5,
) -> np.ndarray:
    """Smooth seams at region boundaries using guided filter + feathering."""
    result = calibrated_mono.copy()

    # Apply guided filter if image available (edge-preserving smoothing)
    if image_bgr is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        try:
            result = cv2.ximgproc.guidedFilter(
                guide=gray.astype(np.float32),
                src=result.astype(np.float32),
                radius=4,
                eps=1.0,
            )
        except AttributeError:
            # cv2.ximgproc not available, use bilateral as fallback
            result = cv2.bilateralFilter(result.astype(np.float32), d=5, sigmaColor=4.0, sigmaSpace=3.0)

    # At true depth edges (high detail_score), restore original calibrated values
    if detail_score is not None:
        edge_mask = (detail_score > 0.3).astype(np.float32)
        result = edge_mask * calibrated_mono + (1.0 - edge_mask) * result

    return np.clip(result, 0, None).astype(np.float32)


def fuse_region_calibrated(
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    theta_low: float = 0.15,
    theta_high: float = 0.55,
    detail_suppression: float = 0.75,
    residual_gain: float = 0.90,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 4.0,
    bilateral_sigma_space: float = 3.0,
    lowpass_sigma: float = 2.0,
    conf_threshold: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Region-Calibrated Fusion: calibrate mono regions using SGM, then restore detail.

    Stage A: Segment mono into coherent disparity regions
    Stage B: Per-region affine calibration using confident SGM pixels
    Stage C: Blend region boundaries (guided filter + feathering)
    Stage D: Restore mono high-frequency detail
    """
    # Compute detail guidance (reuse existing function)
    detail_score = compute_detail_guidance(
        da2_aligned, sgm_disp, confidence_map, image_bgr,
        detail_suppression=detail_suppression,
    )

    # Stage A: Segment
    region_labels = segment_mono_regions(da2_aligned, image_bgr)

    # Stage B: Calibrate
    calibrated_mono, offset_map, region_conf = calibrate_regions(
        da2_aligned, sgm_disp, confidence_map, region_labels,
        conf_threshold=conf_threshold,
        lowpass_sigma=lowpass_sigma,
    )

    # Stage C: Blend boundaries
    calibrated_blended = blend_region_boundaries(
        calibrated_mono, region_labels, detail_score, image_bgr,
    )

    # Stage D: Restore high-frequency detail from mono
    mono_low = cv2.GaussianBlur(da2_aligned.astype(np.float32), (0, 0), lowpass_sigma)
    mono_high = da2_aligned.astype(np.float32) - mono_low

    # Detail restoration gated by detail_score AND region_confidence.
    # Simple multiplicative gate: when region_conf is low, detail restoration
    # is skipped (mono_high may carry noise from uncalibrated regions).
    detail_gate = np.clip(detail_score, 0, 1) * np.clip(region_conf, 0.3, 1.0)
    fused = calibrated_blended + residual_gain * detail_gate * mono_high

    return np.clip(fused, 0, None).astype(np.float32)


FUSION_STRATEGIES = {
    'soft_blend':    fuse_sgm_da2,
    'hard_switch':   fuse_hard_switch,
    'outlier_aware': fuse_outlier_aware,
    'two_threshold': fuse_two_threshold,
    'edge_aware_residual': fuse_edge_aware_residual,
    'region_stable_detail': fuse_region_stable_detail,
    'region_calibrated': fuse_region_calibrated,
}


def fuse_dispatch(
    strategy: str,
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    conf_threshold: float = 0.5,
    outlier_threshold: float = 10.0,
    theta_low: float = 0.3,
    theta_high: float = 0.7,
    detail_suppression: float = 0.75,
    residual_gain: float = 0.90,
) -> np.ndarray:
    """
    Route to the selected fusion function with the appropriate kwargs.
    """
    if strategy not in FUSION_STRATEGIES:
        raise ValueError(f"Unknown fusion strategy '{strategy}'. "
                         f"Choose from: {list(FUSION_STRATEGIES.keys())}")

    if strategy == 'outlier_aware':
        return fuse_outlier_aware(
            sgm_disp, da2_aligned, confidence_map,
            conf_threshold=conf_threshold,
            outlier_threshold=outlier_threshold,
        )
    elif strategy == 'two_threshold':
        return fuse_two_threshold(
            sgm_disp, da2_aligned, confidence_map,
            theta_low=theta_low,
            theta_high=theta_high,
        )
    elif strategy == 'edge_aware_residual':
        return fuse_edge_aware_residual(
            sgm_disp,
            da2_aligned,
            confidence_map,
            image_bgr=image_bgr,
            theta_low=theta_low,
            theta_high=theta_high,
            detail_suppression=detail_suppression,
            residual_gain=residual_gain,
        )
    elif strategy == 'region_stable_detail':
        return fuse_region_stable_detail(
            sgm_disp,
            da2_aligned,
            confidence_map,
            image_bgr=image_bgr,
            theta_low=theta_low,
            theta_high=theta_high,
            residual_gain=residual_gain,
        )
    else:
        # soft_blend and hard_switch both take conf_threshold
        return FUSION_STRATEGIES[strategy](
            sgm_disp, da2_aligned, confidence_map,
            conf_threshold=conf_threshold,
        )
