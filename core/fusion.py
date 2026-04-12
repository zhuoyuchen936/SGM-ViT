"""Confidence-guided SGM + DA2 fusion strategies.

Four pixel-level fusion functions that combine SGM stereo disparity with
DepthAnythingV2 monocular depth, gated by the SGM confidence map.
A dispatcher (``fuse_dispatch``) routes to the correct function by name.
"""
from __future__ import annotations

import cv2
import numpy as np


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

    if image_bgr is None:
        gray = np.zeros_like(sgm_disp, dtype=np.float32)
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
        "g_img": g_img.astype(np.float32),
        "g_da2": g_da2.astype(np.float32),
        "g_sgm": g_sgm.astype(np.float32),
        "edge_adv": edge_adv.astype(np.float32),
        "disp_disagree": disp_disagree.astype(np.float32),
        "da2_low": da2_low.astype(np.float32),
        "da2_high": da2_high.astype(np.float32),
        "fused_base": fused_base.astype(np.float32),
    }
    return fused, debug


# ---------------------------------------------------------------------------
# Fusion dispatcher
# ---------------------------------------------------------------------------

FUSION_STRATEGIES = {
    'soft_blend':    fuse_sgm_da2,
    'hard_switch':   fuse_hard_switch,
    'outlier_aware': fuse_outlier_aware,
    'two_threshold': fuse_two_threshold,
    'edge_aware_residual': fuse_edge_aware_residual,
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
    else:
        # soft_blend and hard_switch both take conf_threshold
        return FUSION_STRATEGIES[strategy](
            sgm_disp, da2_aligned, confidence_map,
            conf_threshold=conf_threshold,
        )
