"""Core SGM-ViT pipeline functions.

This module contains the core inference and alignment functions for the
SGM-ViT hybrid depth-estimation pipeline:

- ``load_da2_model``          — load a DepthAnythingV2 checkpoint
- ``run_masked_sparse_da2``   — sparse DA2 inference via Gather-Attend-Scatter (GAS)
- ``run_twopass_sparse_da2``  — two-pass GAS variant with cross-attention recovery
- ``align_depth_to_sgm``      — align relative mono depth to SGM disparity space

These were originally defined in ``demo.py`` and are extracted here for
reuse across evaluation scripts without pulling in visualisation helpers.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import core._paths  # noqa: F401
from depth_anything_v2.dpt import DepthAnythingV2
from scipy.optimize import least_squares as _scipy_least_squares

from core.decoder_adaptive_precision import (
    build_decoder_sensitivity_map,
    get_weight_quantized_depth_head,
    run_dpt_decoder_with_adaptive_precision,
    run_dpt_decoder_with_weight_adaptive_precision,
)
from core.eval_utils import compute_token_grid_size
from core.sparse_attention import (
    gas_get_intermediate_layers_caps_merge,
    gas_get_intermediate_layers,
    gas_get_intermediate_layers_merge,
    gas_get_intermediate_layers_twopass,
)
from core.token_merge import build_caps_merge_plan, build_token_merge_groups
from core.token_reassembly import reassemble_token_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DA2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

TOKEN_GRID_SIZE = compute_token_grid_size(518, 14)  # 518 // 14 = 37
EMBED_DIM_MAP   = {"vits": 384, "vitb": 768, "vitl": 1024}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_da2_model(encoder: str, weights_path: str, device: torch.device) -> DepthAnythingV2:
    cfg   = DA2_MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # DepthAnythingV2.image2tensor() picks a device from global availability,
    # which breaks explicit CPU runs on GPU hosts. Patch it to follow the model.
    if not hasattr(model, "_sgmvit_image2tensor_patched"):
        original_image2tensor = model.image2tensor

        def _image2tensor_on_model_device(raw_image, input_size=518):
            image, hw = original_image2tensor(raw_image, input_size)
            return image.to(device), hw

        model.image2tensor = _image2tensor_on_model_device  # type: ignore[method-assign]
        model._sgmvit_image2tensor_patched = True

    return model


# ---------------------------------------------------------------------------
# Core inference — sparse DepthAnythingV2
# ---------------------------------------------------------------------------

def run_masked_sparse_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    prune_mask: torch.Tensor,
    input_size: int = 518,
    prune_layer: int = 0,
    do_reassembly: bool = True,
) -> np.ndarray:
    """
    Run DepthAnythingV2 with **Gather-Attend-Scatter (GAS)** sparse attention.

    Physically excludes pruned tokens from the attention computation.
    For each ViT block >= ``prune_layer``:

      1. **Gather** — extract ``[CLS, kept_patches]`` into a compact sequence
      2. **Attend** — standard attention on the short sequence
      3. **Scatter** — write outputs back; pruned tokens retain previous features
      4. **FFN** — MLP on all tokens (per-token, no cross-token dependency)

    Pretrained weights work unchanged because DINOv2 attention is content-based
    (no relative positional encoding), positional encoding is added once
    before all blocks, and all linear layers are per-token operations.

    Parameters
    ----------
    model         : DepthAnythingV2, eval mode
    image_bgr     : (H, W, 3) uint8 BGR
    prune_mask    : (N,) bool CPU Tensor, True = pruned
    input_size    : int
    prune_layer   : int — first ViT block that uses GAS (0-based)
    do_reassembly : bool — fill pruned positions via Gaussian interpolation

    Returns
    -------
    depth : (H_orig, W_orig) float32
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14
    N_actual = patch_h * patch_w

    # ---- Adapt prune_mask to actual token grid (handles non-square images) ----
    if prune_mask.shape[0] != N_actual:
        G = int(round(prune_mask.shape[0] ** 0.5))
        pm_np = prune_mask.reshape(G, G).float().numpy()
        pm_np = cv2.resize(pm_np, (patch_w, patch_h),
                           interpolation=cv2.INTER_NEAREST)
        prune_mask_2d = torch.from_numpy(pm_np > 0.5)
    else:
        prune_mask_2d = prune_mask.reshape(patch_h, patch_w)

    prune_mask_1d = prune_mask_2d.reshape(-1)
    keep_indices = torch.where(~prune_mask_1d)[0].to(image_tensor.device)

    # ---- GAS backbone: gather-attend-scatter on intermediate layers ----
    layer_idx = model.intermediate_layer_idx[model.encoder]

    with torch.no_grad():
        features = gas_get_intermediate_layers(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            keep_indices=keep_indices,
            prune_layer=prune_layer,
        )

        # Optional: Gaussian reassembly at pruned positions
        if do_reassembly:
            features = reassemble_token_features(
                features, prune_mask_2d, patch_h, patch_w,
            )

        # DPT decoder
        depth_tensor = model.depth_head(features, patch_h, patch_w)
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None], (h, w),
            mode="bilinear", align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


def run_twopass_sparse_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    prune_mask: torch.Tensor,
    input_size: int = 518,
    prune_layer: int = 0,
    do_reassembly: bool = True,
) -> np.ndarray:
    """
    Run DepthAnythingV2 with **Two-Pass GAS** sparse attention.

    Pass 1: kept tokens do full self-attention (same as GAS).
    Pass 2: pruned tokens do cross-attention to kept tokens (Q=pruned, KV=kept).

    This restores global context for pruned tokens, significantly improving
    depth quality compared to standard GAS where pruned tokens are "dead".
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14
    N_actual = patch_h * patch_w

    if prune_mask.shape[0] != N_actual:
        G = int(round(prune_mask.shape[0] ** 0.5))
        pm_np = prune_mask.reshape(G, G).float().numpy()
        pm_np = cv2.resize(pm_np, (patch_w, patch_h),
                           interpolation=cv2.INTER_NEAREST)
        prune_mask_2d = torch.from_numpy(pm_np > 0.5)
    else:
        prune_mask_2d = prune_mask.reshape(patch_h, patch_w)

    prune_mask_1d = prune_mask_2d.reshape(-1)
    keep_indices = torch.where(~prune_mask_1d)[0].to(image_tensor.device)
    prune_indices = torch.where(prune_mask_1d)[0].to(image_tensor.device)

    layer_idx = model.intermediate_layer_idx[model.encoder]

    with torch.no_grad():
        features = gas_get_intermediate_layers_twopass(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            keep_indices=keep_indices,
            prune_indices=prune_indices,
            prune_layer=prune_layer,
        )

        if do_reassembly:
            features = reassemble_token_features(
                features, prune_mask_2d, patch_h, patch_w,
            )

        depth_tensor = model.depth_head(features, patch_h, patch_w)
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None], (h, w),
            mode="bilinear", align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


def run_token_merged_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    confidence_map: np.ndarray,
    keep_ratio: float,
    input_size: int = 518,
    merge_layer: int = 0,
) -> np.ndarray:
    """
    Run DepthAnythingV2 with confidence-guided token merge.

    Attention is computed only on a compact sequence of representative tokens,
    then the representative outputs are scattered back to the full spatial grid
    so the DPT decoder still receives a dense token map.
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14

    merge_plan = build_token_merge_groups(
        conf_map=confidence_map,
        token_grid_size=(patch_h, patch_w),
        keep_ratio=keep_ratio,
    )
    rep_patch_indices = merge_plan["rep_idx"].to(image_tensor.device)
    member_to_rep_local = merge_plan["member_to_rep_local"].to(image_tensor.device)

    layer_idx = model.intermediate_layer_idx[model.encoder]

    with torch.no_grad():
        features = gas_get_intermediate_layers_merge(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            rep_patch_indices=rep_patch_indices,
            member_to_rep_local=member_to_rep_local,
            merge_layer=merge_layer,
        )

        depth_tensor = model.depth_head(features, patch_h, patch_w)
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None],
            (h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


def run_caps_merged_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    confidence_map: np.ndarray,
    keep_ratio: float,
    input_size: int = 518,
    merge_layer: int = 0,
    high_precision_ratio: float = 0.25,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
    conf_weight: float = 1.0,
    var_weight: float = 0.5,
    range_weight: float = 0.5,
    radius_weight: float = 0.25,
) -> np.ndarray:
    """
    Run DepthAnythingV2 with merge + confidence-aware adaptive precision.

    This is a prototype path: representative tokens are assigned different
    fake-quantized activation precisions according to group sensitivity.
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14

    merge_plan = build_caps_merge_plan(
        conf_map=confidence_map,
        token_grid_size=(patch_h, patch_w),
        keep_ratio=keep_ratio,
        high_precision_ratio=high_precision_ratio,
        conf_weight=conf_weight,
        var_weight=var_weight,
        range_weight=range_weight,
        radius_weight=radius_weight,
    )
    rep_patch_indices = merge_plan["rep_idx"].to(image_tensor.device)
    member_to_rep_local = merge_plan["member_to_rep_local"].to(image_tensor.device)
    high_precision_local_mask = merge_plan["high_precision_local_mask"].to(image_tensor.device)

    layer_idx = model.intermediate_layer_idx[model.encoder]

    with torch.no_grad():
        features = gas_get_intermediate_layers_caps_merge(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            rep_patch_indices=rep_patch_indices,
            member_to_rep_local=member_to_rep_local,
            high_precision_local_mask=high_precision_local_mask,
            merge_layer=merge_layer,
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )

        depth_tensor = model.depth_head(features, patch_h, patch_w)
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None],
            (h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


def run_decoder_caps_merged_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    confidence_map: np.ndarray,
    keep_ratio: float,
    input_size: int = 518,
    merge_layer: int = 0,
    decoder_high_precision_ratio: float = 0.5,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
    decoder_conf_weight: float = 1.0,
    decoder_texture_weight: float = 0.0,
    decoder_variance_weight: float = 0.0,
    decoder_stage_policy: str = "all",
) -> np.ndarray:
    """
    Run token-merged DA2 with decoder-aware adaptive precision.

    Encoder attention stays in merge FP32 mode. The DPT decoder then applies
    spatially adaptive precision on feature maps according to a sensitivity map
    derived from SGM confidence and optional image texture cues.
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14

    merge_plan = build_token_merge_groups(
        conf_map=confidence_map,
        token_grid_size=(patch_h, patch_w),
        keep_ratio=keep_ratio,
    )
    rep_patch_indices = merge_plan["rep_idx"].to(image_tensor.device)
    member_to_rep_local = merge_plan["member_to_rep_local"].to(image_tensor.device)
    layer_idx = model.intermediate_layer_idx[model.encoder]

    sensitivity_map = build_decoder_sensitivity_map(
        conf_map=confidence_map,
        image_bgr=image_bgr,
        conf_weight=decoder_conf_weight,
        texture_weight=decoder_texture_weight,
        variance_weight=decoder_variance_weight,
    )

    with torch.no_grad():
        features = gas_get_intermediate_layers_merge(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            rep_patch_indices=rep_patch_indices,
            member_to_rep_local=member_to_rep_local,
            merge_layer=merge_layer,
        )

        depth_tensor = run_dpt_decoder_with_adaptive_precision(
            depth_head=model.depth_head,
            out_features=features,
            patch_h=patch_h,
            patch_w=patch_w,
            sensitivity_map=sensitivity_map,
            high_precision_ratio=decoder_high_precision_ratio,
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
            stage_policy=decoder_stage_policy,
        )
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None],
            (h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


def run_decoder_weight_caps_merged_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    confidence_map: np.ndarray,
    keep_ratio: float,
    input_size: int = 518,
    merge_layer: int = 0,
    decoder_high_precision_ratio: float = 0.75,
    high_precision_bits: int | None = None,
    low_precision_bits: int = 4,
    decoder_conf_weight: float = 1.0,
    decoder_texture_weight: float = 0.0,
    decoder_variance_weight: float = 0.0,
    decoder_stage_policy: str = "coarse_only",
) -> np.ndarray:
    """
    Run token-merged DA2 with a weight-aware decoder dual-path prototype.

    The encoder stays in merge FP32 mode. Selected decoder stages then run both
    high-precision kernels (FP32 by default, or a quantized high-precision
    branch when ``high_precision_bits`` is set) and a weight-quantized
    low-precision kernel, then spatially mix their outputs according to the
    high-precision mask.
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h = image_tensor.shape[-2] // 14
    patch_w = image_tensor.shape[-1] // 14

    merge_plan = build_token_merge_groups(
        conf_map=confidence_map,
        token_grid_size=(patch_h, patch_w),
        keep_ratio=keep_ratio,
    )
    rep_patch_indices = merge_plan["rep_idx"].to(image_tensor.device)
    member_to_rep_local = merge_plan["member_to_rep_local"].to(image_tensor.device)
    layer_idx = model.intermediate_layer_idx[model.encoder]

    sensitivity_map = build_decoder_sensitivity_map(
        conf_map=confidence_map,
        image_bgr=image_bgr,
        conf_weight=decoder_conf_weight,
        texture_weight=decoder_texture_weight,
        variance_weight=decoder_variance_weight,
    )
    high_precision_depth_head = None
    if high_precision_bits is not None and high_precision_bits < 16:
        high_precision_depth_head = get_weight_quantized_depth_head(model.depth_head, high_precision_bits)
    quantized_depth_head = get_weight_quantized_depth_head(model.depth_head, low_precision_bits)

    with torch.no_grad():
        features = gas_get_intermediate_layers_merge(
            backbone=model.pretrained,
            x_input=image_tensor,
            layer_indices=layer_idx,
            rep_patch_indices=rep_patch_indices,
            member_to_rep_local=member_to_rep_local,
            merge_layer=merge_layer,
        )

        depth_tensor = run_dpt_decoder_with_weight_adaptive_precision(
            depth_head=model.depth_head,
            quantized_depth_head=quantized_depth_head,
            out_features=features,
            patch_h=patch_h,
            patch_w=patch_w,
            sensitivity_map=sensitivity_map,
            high_precision_ratio=decoder_high_precision_ratio,
            high_precision_depth_head=high_precision_depth_head,
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
            stage_policy=decoder_stage_policy,
        )
        depth_tensor = F.relu(depth_tensor).squeeze(1)

        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        depth = F.interpolate(
            depth_tensor[:, None],
            (h, w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return depth.cpu().numpy()


# ---------------------------------------------------------------------------
# Metric depth alignment
# ---------------------------------------------------------------------------

def align_depth_to_sgm(
    depth_mono: np.ndarray,
    disparity_raw: np.ndarray,
    confidence_map: np.ndarray,
    min_disparity: float = 1.0,
    conf_threshold: float = 0.7,
) -> tuple[np.ndarray, float, float]:
    """
    Pull relative monocular depth into the physical SGM disparity space.

    Formulation
    -----------
    DAv2 outputs affine-invariant depth where larger values correspond to
    closer objects — the same monotonicity as stereo disparity.  We
    therefore align directly in disparity units (pixels) without
    converting to metric depth:

        argmin_{s,t}  Σ_i ρ_H( s·d_i + t − disp_i )

    where  d_i    = depth_mono at reliable pixel i
           disp_i = raw SGM disparity at the same pixel
           ρ_H    = Huber loss (f_scale = 3.0 px)

    Huber loss down-weights SGM outliers (wrong matches in textured or
    thin-object regions that pass the LR-check but still carry large
    disparity errors).  This suppresses the systematic bias that plain
    L2 regression exhibits at near/far depth extremes, reducing both
    EPE and the number of D1-threshold crossings.  No camera intrinsics
    are required.

    Parameters
    ----------
    depth_mono     : (H, W) float32  relative monocular depth (arbitrary scale)
    disparity_raw  : (H, W) float32  raw SGM disparity in pixels
    confidence_map : (H, W) float32  SGM per-pixel confidence [0, 1]
    min_disparity  : float  minimum valid SGM disparity
    conf_threshold : float  minimum SGM confidence to include a pixel

    Returns
    -------
    disp_aligned : (H, W) float32  monocular depth aligned to disparity space
                   (same units and scale as SGM disparity, pixels, ≥ 0)
    scale        : float
    shift        : float   disp_aligned ≈ scale·depth_mono + shift
    """
    if disparity_raw is None or float(disparity_raw.max()) < min_disparity:
        print("  [align] No valid SGM disparity — skipping alignment.")
        return depth_mono.astype(np.float32), 1.0, 0.0

    # Match spatial resolution to monocular depth map
    if disparity_raw.shape != depth_mono.shape:
        disparity_raw  = cv2.resize(disparity_raw,
                                    (depth_mono.shape[1], depth_mono.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        confidence_map = cv2.resize(confidence_map,
                                    (depth_mono.shape[1], depth_mono.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

    valid   = (disparity_raw > min_disparity) & (confidence_map >= conf_threshold)
    n_valid = int(valid.sum())

    if n_valid < 50:
        print(f"  [align] Only {n_valid} valid SGM pixels — skipping alignment.")
        return depth_mono.astype(np.float32), 1.0, 0.0

    # Target: SGM disparity in pixels (physical disparity space)
    D = disparity_raw[valid].astype(np.float64)
    d = depth_mono[valid].astype(np.float64)

    # Robust least-squares with Huber loss:  s·d + t ≈ D
    # Huber down-weights SGM outliers (wrong matches in textured / thin-object
    # regions that pass the LR-check but still carry large disparity errors).
    # This suppresses the systematic bias that L2 regression has at near/far
    # extremes, reducing both EPE and the number of D1-threshold crossings.
    def _residuals(coef):
        return coef[0] * d + coef[1] - D

    # Warm-start from the L2 solution for fast convergence
    A_ls          = np.column_stack([d, np.ones_like(d)])
    coef0, _, _, _ = np.linalg.lstsq(A_ls, D, rcond=None)
    result        = _scipy_least_squares(_residuals, coef0, loss='huber', f_scale=3.0)
    scale, shift  = float(result.x[0]), float(result.x[1])

    disp_aligned = np.clip(scale * depth_mono + shift, 0.0, None).astype(np.float32)

    print(f"  [align] n_valid={n_valid:,}  scale={scale:.4f}  shift={shift:.4f}  "
          f"disp range: [{disp_aligned.min():.2f}, {disp_aligned.max():.2f}] px")

    return disp_aligned, scale, shift
