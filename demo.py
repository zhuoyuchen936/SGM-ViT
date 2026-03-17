"""
demo.py — SGM-ViT End-to-End Demo
===================================
Demonstrates the full SGM-ViT pipeline on a stereo image pair:

  1. SGM stereo matching  →  disparity map + confidence map
  2. DepthAnythingV2 (ViT-S)  →  dense monocular depth map (baseline)
  3. SGMConfidenceTokenRouter  →  keep / prune token split
  4. Sparse DA2 with:
       a. Progressive pruning  — token zeroing starts at blocks[prune_layer]
          (not necessarily block 0) so tokens accumulate low-level context
          before being suppressed, trading FLOPs for accuracy.
       b. Token re-assembly   — before the DPT decoder, pruned feature
          positions are filled via Gaussian-weighted interpolation from
          kept neighbours (restores spatial completeness at all decoder
          input levels).
  5. Metric depth alignment  →  least-squares scale & shift from SGM
       disparity for both dense and sparse DA2.
  6. SGM + DA2 fusion  →  use SGM where confident, aligned DA2 elsewhere;
       confidence-weighted soft blend avoids boundary discontinuities.
  7. Multi-panel visualisation  →  saved to results/demo/

Usage
-----
  python demo.py                          # default KITTI test pair
  python demo.py --no-sgm                # skip SGM (uniform confidence)
  python demo.py --threshold 0.5         # pruning threshold
  python demo.py --prune-layer 6         # prune at block 6 (not 0)
  python demo.py --no-reassembly         # disable token re-assembly
  python demo.py --no-align              # skip metric alignment
  python demo.py --fusion-strategy hard_switch   # binary SGM/DA2 selector
  python demo.py --fusion-strategy outlier_aware --outlier-threshold 10
  python demo.py --fusion-strategy two_threshold --theta-low 0.3 --theta-high 0.7
  python demo.py --conf-threshold 0.55   # fusion blend boundary
  python demo.py --help

Expected outputs (in results/demo/)
-------------------------------------
  01_left_image.png          — input left image
  02_sgm_disparity.png       — filled SGM disparity (plasma)
  03_sgm_confidence.png      — per-pixel SGM confidence (RdYlGn)
  04_da2_depth.png           — Dense DA2 baseline depth (Spectral_r)
  05_token_routing.png       — 37×37 token keep/prune grid
  06_routing_overlay.png     — routing decision overlaid on input image
  07_sparse_da2_depth.png    — Sparse DA2 output (pruning + re-assembly)
  08_aligned_disparity.png   — Mono depth aligned to SGM disparity space (pixels, plasma)
  09_fused_sgm_dense_da2.png — Fused SGM + Dense DA2 (SGM where confident, DA2 elsewhere)
  09b_fused_sgm_sparse_da2.png — Fused SGM + Sparse DA2 (GAS-accelerated)
  10_diff_map.png            — |Dense − Sparse| per-pixel diff (hot)
  11_comparison.png          — 4-panel paper figure
  00_summary.png             — composite panel

Runtime notes
-------------
  • On the FIRST run, Numba JIT compilation of the SGM kernels takes 30-90 s.
  • Subsequent runs use the cached compiled code and are much faster (~5 s).
  • Use --no-sgm to skip SGM and test only the DAv2 + router path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types as _types

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares as _scipy_least_squares

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DA2_DIR    = os.path.join(_SCRIPT_DIR, "Depth-Anything-V2")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _DA2_DIR not in sys.path:
    sys.path.insert(0, _DA2_DIR)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from core.sgm_wrapper        import run_sgm_with_confidence, confidence_to_token_grid
from core.token_router       import SGMConfidenceTokenRouter
from core.token_reassembly   import reassemble_token_features
from core.sparse_attention   import gas_get_intermediate_layers
from core.eval_utils         import compute_token_grid_size

from depth_anything_v2.dpt import DepthAnythingV2


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
# Helpers — visualisation
# ---------------------------------------------------------------------------

def load_da2_model(encoder: str, weights_path: str, device: torch.device) -> DepthAnythingV2:
    cfg   = DA2_MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def colorize(arr: np.ndarray, cmap: str = "plasma", vmin=None, vmax=None) -> np.ndarray:
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    cm   = plt.get_cmap(cmap)
    rgba = (cm(norm) * 255).astype(np.uint8)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def resize_to_match(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    th, tw = target.shape[:2]
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def draw_token_grid(
    token_conf: np.ndarray,
    threshold: float,
    cell_px: int = 14,
) -> np.ndarray:
    G     = token_conf.shape[0]
    canvas = np.zeros((G * cell_px, G * cell_px, 3), dtype=np.uint8)
    prune_mask = token_conf > threshold
    for r in range(G):
        for c in range(G):
            y0, y1 = r * cell_px, (r + 1) * cell_px
            x0, x1 = c * cell_px, (c + 1) * cell_px
            colour = (220, 60, 60) if prune_mask[r, c] else (60, 180, 60)
            canvas[y0:y1, x0:x1] = colour
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (30, 30, 30), 1)
    legend = np.full((28, G * cell_px, 3), 40, dtype=np.uint8)
    cv2.putText(legend, f"GREEN=keep  RED=prune  theta={threshold:.2f}",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return np.vstack([canvas, legend])


def save_panel(img_bgr: np.ndarray, path: str, title: str | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if title:
        bar = np.full((30, img_bgr.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        img_bgr = np.vstack([bar, img_bgr])
    cv2.imwrite(path, img_bgr)
    print(f"  [saved] {path}")


def build_summary_figure(
    panels: list[tuple[str, np.ndarray]],
    out_path: str,
    ncol: int = 4,
) -> None:
    n    = len(panels)
    nrow = (n + ncol - 1) // ncol
    fig  = plt.figure(figsize=(ncol * 6, nrow * 5.5), constrained_layout=True)
    fig.suptitle(
        "SGM-ViT Demo: SGM Confidence-Guided Token Routing + DepthAnythingV2",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    for idx, (title, bgr) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // ncol, idx % ncol])
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(n, nrow * ncol):
        fig.add_subplot(gs[idx // ncol, idx % ncol]).axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def build_comparison_figure(
    da2_bgr: np.ndarray,
    sparse_bgr: np.ndarray,
    aligned_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    prune_ratio: float,
    attn_reduction: float,
    threshold: float,
    scale: float,
    shift: float,
    out_path: str,
    aligned_is_disparity: bool = False,
) -> None:
    """
    4-panel paper figure:
      Dense DA2  |  Sparse DA2 (SGM-ViT)  |  Aligned to SGM disp  |  |Dense−Sparse|
    """
    fig, axes = plt.subplots(1, 4, figsize=(28, 5.5), constrained_layout=True)
    aligned_subtitle = (
        f"s={scale:.4f}  t={shift:.4f}  [mono depth → SGM disp space, px]"
        if aligned_is_disparity else "no alignment"
    )
    panels = [
        (da2_bgr,
         "Dense DA2 (Baseline)",
         "All tokens attend — no pruning"),
        (sparse_bgr,
         "Sparse DA2 (SGM-ViT)",
         f"{100*prune_ratio:.1f}% pruned  |  Attn FLOPs↓{100*attn_reduction:.1f}%  |  θ={threshold:.2f}"),
        (aligned_bgr,
         "Mono Depth → SGM Disparity Space",
         aligned_subtitle),
        (diff_bgr,
         "|Dense − Sparse| Difference",
         "Brighter = larger prediction change from pruning"),
    ]
    for ax, (bgr, title, subtitle) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
        ax.set_xlabel(subtitle, fontsize=9, color="#555555")
        ax.axis("off")
    fig.suptitle(
        "SGM-ViT: Dense DA2 vs. Sparse DA2 (pruning + re-assembly) vs. SGM-Disparity-Aligned",
        fontsize=12, fontweight="bold", y=1.01,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Core inference — sparse DepthAnythingV2
# ---------------------------------------------------------------------------

def run_sparse_da2(
    model: DepthAnythingV2,
    image_bgr: np.ndarray,
    prune_mask: torch.Tensor,
    input_size: int = 518,
    prune_layer: int = 0,
    do_reassembly: bool = True,
) -> np.ndarray:
    """
    Run DepthAnythingV2 with two SGM-guided optimisations:

    1. **Progressive pruning** (``prune_layer``)
       Zero pruned spatial tokens at the INPUT of transformer block
       ``prune_layer`` (0-indexed, range 0 .. num_blocks-1).  Tokens
       participate normally in all blocks *before* ``prune_layer``,
       accumulating low-level context before their contributions are
       suppressed.  Higher values trade fewer FLOPs savings for better
       depth accuracy.

       Implementation: ``register_forward_pre_hook`` on
       ``pretrained.blocks[prune_layer]``.  The hook fires after
       ``prepare_tokens_with_masks`` has produced the full sequence
       [CLS | register_tokens | patch_tokens]; pruned spatial positions
       are multiplied by 0.

    2. **Token re-assembly** (``do_reassembly``)
       Before the DPT decoder, pruned positions in each of the 4
       intermediate feature maps (layers 2, 5, 8, 11 for ViT-S) are
       filled via Gaussian-weighted interpolation from kept neighbours
       (see ``core.token_reassembly.reassemble_token_features``).  This
       restores spatial completeness so the decoder receives a full
       feature grid at every scale.

       Implementation: ``model.forward`` is temporarily monkey-patched
       to intercept between ``pretrained.get_intermediate_layers()`` and
       ``depth_head()``.  The patch is removed in the ``finally`` block.

    Parameters
    ----------
    model         : DepthAnythingV2, eval mode
    image_bgr     : (H, W, 3) uint8 BGR
    prune_mask    : (N,) bool CPU Tensor, True = pruned.
                    N may differ from the actual token count for non-square
                    images; the mask is resized (nearest-neighbour) to match.
    input_size    : int
    prune_layer   : int — which ViT block receives the zeroing hook (0-based)
    do_reassembly : bool — fill pruned feature positions before DPT decoder

    Returns
    -------
    depth : (H_orig, W_orig) float32
    """
    image_tensor, (h, w) = model.image2tensor(image_bgr, input_size)
    patch_h  = image_tensor.shape[-2] // 14
    patch_w  = image_tensor.shape[-1] // 14
    N_actual = patch_h * patch_w

    # ----------------------------------------------------------------
    # Adapt prune_mask to actual token grid (handles non-square images)
    # ----------------------------------------------------------------
    if prune_mask.shape[0] != N_actual:
        G       = int(round(prune_mask.shape[0] ** 0.5))
        pm_np   = prune_mask.reshape(G, G).float().numpy()
        pm_np   = cv2.resize(pm_np, (patch_w, patch_h),
                             interpolation=cv2.INTER_NEAREST)
        prune_mask_2d = torch.from_numpy(pm_np > 0.5)
    else:
        prune_mask_2d = prune_mask.reshape(patch_h, patch_w)

    nr          = getattr(model.pretrained, 'num_register_tokens', 0)
    keep_weight = (~prune_mask_2d.reshape(-1)).float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

    # ----------------------------------------------------------------
    # Hook 1: zero pruned tokens at blocks[prune_layer] input
    # ----------------------------------------------------------------
    def _pre_hook(_module, args):
        x = args[0].clone()                                     # (B, 1+nr+N, D)
        x[:, 1 + nr : 1 + nr + N_actual, :] *= keep_weight.to(x.device)
        return (x,) + args[1:]

    if getattr(model.pretrained, 'chunked_blocks', False):
        all_blocks = [b for chunk in model.pretrained.blocks for b in chunk]
    else:
        all_blocks = list(model.pretrained.blocks)

    layer_idx   = min(prune_layer, len(all_blocks) - 1)
    hook_handle = all_blocks[layer_idx].register_forward_pre_hook(_pre_hook)

    # ----------------------------------------------------------------
    # Optional: patch model.forward to inject re-assembly between
    #           backbone (get_intermediate_layers) and DPT decoder
    # ----------------------------------------------------------------
    original_forward = model.forward
    _prune_2d        = prune_mask_2d   # captured by closure

    if do_reassembly:
        def _patched_forward(self_m, x_in):
            ph = x_in.shape[-2] // 14
            pw = x_in.shape[-1] // 14
            feats = self_m.pretrained.get_intermediate_layers(
                x_in,
                self_m.intermediate_layer_idx[self_m.encoder],
                return_class_token=True,
            )
            feats = reassemble_token_features(feats, _prune_2d, ph, pw)
            depth = self_m.depth_head(feats, ph, pw)
            return F.relu(depth).squeeze(1)

        model.forward = _types.MethodType(_patched_forward, model)

    try:
        with torch.no_grad():
            depth_tensor = model.forward(image_tensor)         # (B, H', W')
            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0)
            depth = F.interpolate(
                depth_tensor[:, None], (h, w),
                mode="bilinear", align_corners=True,
            )[0, 0]
        return depth.cpu().numpy()
    finally:
        hook_handle.remove()
        model.forward = original_forward


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

    Unlike :func:`run_sparse_da2` which *zeros* pruned tokens (causing
    attention pollution), this function physically excludes pruned tokens
    from the attention computation.  For each ViT block ≥ ``prune_layer``:

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
    * SGM confident  (sgm_disp > 0  AND  conf ≥ threshold) → use SGM directly.
      SGM is geometrically accurate in these regions (sv D1 ≈ 8%).
    * SGM uncertain / hole (sgm_disp = 0  OR  conf < threshold) → use DA2.
      DA2 provides semantic coverage where SGM fails.

    A confidence-weighted soft blend is applied instead of a hard threshold
    to avoid discontinuities at region boundaries:

        alpha  = clip(conf / threshold, 0, 1)  if sgm_disp > 0  else 0
        fused  = alpha · sgm_disp + (1 − alpha) · da2_aligned

    When conf ≥ threshold the alpha saturates at 1 and SGM is used exclusively.
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
    # alpha ∈ [0,1]: 1 = full SGM, 0 = full DA2
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
        if sgm_disp > 0 AND conf >= conf_threshold → use SGM
        else → use DA2

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
    the LR-check but have high |SGM − DA2| disagreement:

        alpha_base      = clip(conf / θ, 0, 1) * sgm_valid
        discrepancy     = |sgm_disp − da2_aligned|
        outlier_factor  = clip(1 − discrepancy / outlier_threshold, 0, 1)
        alpha           = alpha_base * outlier_factor
        fused           = alpha * SGM + (1 − alpha) * DA2

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

        conf < θ_low            → pure DA2  (alpha = 0)
        θ_low ≤ conf < θ_high   → linear blend:  alpha = (conf − θ_low) / (θ_high − θ_low)
        conf ≥ θ_high           → pure SGM  (alpha = 1)

    Prevents very-low-confidence SGM from leaking into the output.
    FPGA cost: 2 comparators + 1 DSP per pixel.  Reciprocal
    1 / (θ_high − θ_low) is a compile-time constant.
    """
    sgm_valid = (sgm_disp > 0).astype(np.float32)
    span = max(theta_high - theta_low, 1e-6)
    alpha = np.clip((confidence_map - theta_low) / span, 0.0, 1.0)
    alpha = alpha * sgm_valid
    fused = alpha * sgm_disp + (1.0 - alpha) * da2_aligned
    return fused.clip(0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion dispatcher
# ---------------------------------------------------------------------------

FUSION_STRATEGIES = {
    'soft_blend':    fuse_sgm_da2,
    'hard_switch':   fuse_hard_switch,
    'outlier_aware': fuse_outlier_aware,
    'two_threshold': fuse_two_threshold,
}


def fuse_dispatch(
    strategy: str,
    sgm_disp: np.ndarray,
    da2_aligned: np.ndarray,
    confidence_map: np.ndarray,
    conf_threshold: float = 0.5,
    outlier_threshold: float = 10.0,
    theta_low: float = 0.3,
    theta_high: float = 0.7,
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
    else:
        # soft_blend and hard_switch both take conf_threshold
        return FUSION_STRATEGIES[strategy](
            sgm_disp, da2_aligned, confidence_map,
            conf_threshold=conf_threshold,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"\n{'='*62}")
    print(f"  SGM-ViT Demo")
    print(f"{'='*62}")
    print(f"  Left      : {args.left}")
    print(f"  Right     : {args.right}")
    print(f"  Weights   : {args.weights}")
    print(f"  Device    : {device}")
    print(f"  Output    : {out_dir}")
    print(f"  Threshold : {args.threshold}   Prune-layer: {args.prune_layer}")
    print(f"  Sparse mode : {args.sparse_mode}")
    print(f"  Re-assembly: {'yes' if not args.no_reassembly else 'no'}"
          f"   Alignment: {'yes' if not args.no_align else 'no'}")
    print(f"  Fusion      : {args.fusion_strategy}")
    print(f"{'='*62}\n")

    # ------------------------------------------------------------------
    # Step 1: Load left image
    # ------------------------------------------------------------------
    left_bgr = cv2.imread(args.left)
    if left_bgr is None:
        raise FileNotFoundError(f"Cannot load left image: {args.left}")
    H_orig, W_orig = left_bgr.shape[:2]
    print(f"[1/7] Left image: {W_orig}×{H_orig} px")

    # ------------------------------------------------------------------
    # Step 2: SGM stereo matching
    # ------------------------------------------------------------------
    disp_raw = None   # kept for metric alignment
    if args.no_sgm:
        print("[2/7] Skipping SGM (--no-sgm).  Uniform confidence = 0.5")
        disparity_norm = np.zeros((H_orig, W_orig), dtype=np.float32)
        confidence_map = np.full((H_orig, W_orig), 0.5, dtype=np.float32)
    else:
        print("[2/7] Running SGM stereo matching ...")
        print("      (Numba JIT on first run: ~30-90 s)")
        t_sgm = time.perf_counter()
        disparity_norm, confidence_map, disp_raw = run_sgm_with_confidence(
            left_path       = args.left,
            right_path      = args.right,
            disparity_range = args.disparity_range,
            smooth_sigma    = args.conf_sigma,
            verbose         = True,
        )
        print(f"      SGM time: {time.perf_counter()-t_sgm:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Dense DepthAnythingV2 (baseline)
    # ------------------------------------------------------------------
    print(f"[3/7] Loading DepthAnythingV2 ({args.encoder}) ...")
    model = load_da2_model(args.encoder, args.weights, device)

    print("[3/7] Dense depth inference ...")
    t_da2   = time.perf_counter()
    depth_map = model.infer_image(left_bgr, input_size=518)
    print(f"      DAv2 time: {time.perf_counter()-t_da2:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 4: Token routing
    # ------------------------------------------------------------------
    print(f"[4/7] Token routing (θ={args.threshold}) ...")
    router     = SGMConfidenceTokenRouter(
        token_grid_size      = TOKEN_GRID_SIZE,
        confidence_threshold = args.threshold,
        learnable_threshold  = False,
    )
    embed_dim    = EMBED_DIM_MAP[args.encoder]
    N            = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE   # 1369 (square 518-px input)
    conf_tensor  = torch.from_numpy(confidence_map).unsqueeze(0).unsqueeze(0)
    dummy_tokens = torch.zeros(1, N, embed_dim)

    with torch.no_grad():
        routing = router(conf_tensor, dummy_tokens)

    n_keep       = len(routing["keep_idx"][0])
    n_prune      = len(routing["prune_idx"][0])
    prune_ratio  = routing["prune_ratio"]
    attn_reduction = 1.0 - (n_keep / N) ** 2

    print(f"      Total tokens  : {N}")
    print(f"      Kept          : {n_keep}  ({100*(1-prune_ratio):.1f}%)")
    print(f"      Pruned        : {n_prune}  ({100*prune_ratio:.1f}%)")
    print(f"      Attn FLOPs ↓  : {100*attn_reduction:.1f}%\n")

    token_conf = confidence_to_token_grid(confidence_map, TOKEN_GRID_SIZE)

    # ------------------------------------------------------------------
    # Step 5: Sparse DA2 — progressive pruning + token re-assembly
    # ------------------------------------------------------------------
    prune_mask_1d = torch.zeros(N, dtype=torch.bool)
    prune_mask_1d[routing["prune_idx"][0]] = True

    n_blocks = len(list(model.pretrained.blocks))
    print(f"[5/7] Sparse DA2 inference  (mode: {args.sparse_mode}) ...")
    print(f"      Prune at block: {args.prune_layer}/{n_blocks-1}")
    print(f"      Re-assembly   : {'enabled' if not args.no_reassembly else 'disabled'}")

    t_sparse = time.perf_counter()
    if args.sparse_mode == "mask":
        depth_sparse = run_masked_sparse_da2(
            model         = model,
            image_bgr     = left_bgr,
            prune_mask    = prune_mask_1d,
            input_size    = 518,
            prune_layer   = args.prune_layer,
            do_reassembly = not args.no_reassembly,
        )
    else:
        depth_sparse = run_sparse_da2(
            model         = model,
            image_bgr     = left_bgr,
            prune_mask    = prune_mask_1d,
            input_size    = 518,
            prune_layer   = args.prune_layer,
            do_reassembly = not args.no_reassembly,
        )
    print(f"      Sparse time: {time.perf_counter()-t_sparse:.2f}s\n")

    # Dense/sparse normalised to [0,1] for diff computation
    dense_norm  = (depth_map    - depth_map.min())    / (depth_map.max()    - depth_map.min()    + 1e-8)
    sparse_norm = (depth_sparse - depth_sparse.min()) / (depth_sparse.max() - depth_sparse.min() + 1e-8)
    diff_map    = np.abs(dense_norm - sparse_norm).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 5.5: Metric depth alignment via SGM least-squares
    # ------------------------------------------------------------------
    scale_dense, shift_dense = 1.0, 0.0
    scale, shift = 1.0, 0.0
    if args.no_align or args.no_sgm or disp_raw is None:
        print("[5.5/7] Skipping disparity-space alignment (--no-align or no SGM).")
        disp_dense_aligned = depth_map
        disp_aligned = depth_sparse
    else:
        print("[5.5/7] Disparity-space alignment (pull monocular depth → SGM disp) ...")
        print("      Aligning dense DA2 ...")
        disp_dense_aligned, scale_dense, shift_dense = align_depth_to_sgm(
            depth_mono     = depth_map,
            disparity_raw  = disp_raw,
            confidence_map = confidence_map,
            conf_threshold = args.conf_threshold,
        )
        print("      Aligning sparse DA2 ...")
        disp_aligned, scale, shift = align_depth_to_sgm(
            depth_mono     = depth_sparse,
            disparity_raw  = disp_raw,
            confidence_map = confidence_map,
            conf_threshold = args.conf_threshold,
        )

    # ------------------------------------------------------------------
    # Step 6: SGM + DA2 Fusion (Dense and Sparse)
    # ------------------------------------------------------------------
    if args.no_align or args.no_sgm or disp_raw is None:
        print("[6/7] Skipping fusion (no alignment available).")
        disp_fused = disp_dense_aligned
        disp_fused_sparse = disp_aligned
    else:
        print(f"[6/7] Fusing SGM + aligned DA2  (strategy: {args.fusion_strategy}) ...")
        sgm_for_fuse  = disp_raw
        conf_for_fuse = confidence_map
        if disp_raw.shape != disp_dense_aligned.shape:
            sgm_for_fuse = cv2.resize(disp_raw,
                (disp_dense_aligned.shape[1], disp_dense_aligned.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            conf_for_fuse = cv2.resize(confidence_map,
                (disp_dense_aligned.shape[1], disp_dense_aligned.shape[0]),
                interpolation=cv2.INTER_LINEAR)
        # Fused SGM + Dense DA2
        disp_fused = fuse_dispatch(
            strategy          = args.fusion_strategy,
            sgm_disp          = sgm_for_fuse,
            da2_aligned       = disp_dense_aligned,
            confidence_map    = conf_for_fuse,
            conf_threshold    = args.conf_threshold,
            outlier_threshold = args.outlier_threshold,
            theta_low         = args.theta_low,
            theta_high        = args.theta_high,
        )
        print(f"      Fused SGM+Dense  disp range: [{disp_fused.min():.2f}, {disp_fused.max():.2f}] px")

        # Fused SGM + Sparse DA2
        sgm_for_fuse_sp  = sgm_for_fuse
        conf_for_fuse_sp = conf_for_fuse
        if disp_aligned.shape != disp_dense_aligned.shape:
            disp_aligned_resized = cv2.resize(disp_aligned,
                (disp_dense_aligned.shape[1], disp_dense_aligned.shape[0]),
                interpolation=cv2.INTER_LINEAR)
        else:
            disp_aligned_resized = disp_aligned
        disp_fused_sparse = fuse_dispatch(
            strategy          = args.fusion_strategy,
            sgm_disp          = sgm_for_fuse_sp,
            da2_aligned       = disp_aligned_resized,
            confidence_map    = conf_for_fuse_sp,
            conf_threshold    = args.conf_threshold,
            outlier_threshold = args.outlier_threshold,
            theta_low         = args.theta_low,
            theta_high        = args.theta_high,
        )
        print(f"      Fused SGM+Sparse disp range: [{disp_fused_sparse.min():.2f}, {disp_fused_sparse.max():.2f}] px\n")

    # ------------------------------------------------------------------
    # Step 7: Save visualisations
    # ------------------------------------------------------------------
    print("[7/7] Saving visualisations ...")

    # Panel 1
    p1 = left_bgr.copy()
    save_panel(p1, os.path.join(out_dir, "01_left_image.png"), "Input Left Image")

    # Panel 2
    p2 = colorize(disparity_norm, cmap="plasma")
    p2 = resize_to_match(p2, left_bgr)
    save_panel(p2, os.path.join(out_dir, "02_sgm_disparity.png"), "SGM Disparity Map")

    # Panel 3
    p3 = colorize(confidence_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    p3 = resize_to_match(p3, left_bgr)
    save_panel(p3, os.path.join(out_dir, "03_sgm_confidence.png"),
               f"SGM Confidence (mean={confidence_map.mean():.2f})")

    # Panel 4 — dense DA2 baseline
    p4 = colorize(depth_map, cmap="Spectral_r")
    p4 = resize_to_match(p4, left_bgr)
    save_panel(p4, os.path.join(out_dir, "04_da2_depth.png"),
               f"Dense DA2 ({args.encoder}) — Baseline")

    # Panel 5 — token routing grid
    grid_img = draw_token_grid(token_conf, threshold=args.threshold, cell_px=14)
    save_panel(grid_img, os.path.join(out_dir, "05_token_routing.png"),
               f"Token Grid (θ={args.threshold:.2f}, prune={100*prune_ratio:.1f}%)")

    # Panel 6 — routing overlay
    prune_mask_full = cv2.resize(
        (token_conf > args.threshold).astype(np.uint8) * 255,
        (W_orig, H_orig), interpolation=cv2.INTER_NEAREST,
    )
    overlay = left_bgr.copy()
    r_ch = overlay[:, :, 2]
    g_ch = overlay[:, :, 1]
    r_ch[prune_mask_full > 0]  = np.clip(r_ch[prune_mask_full > 0].astype(int) + 60,   0, 255).astype(np.uint8)
    g_ch[prune_mask_full == 0] = np.clip(g_ch[prune_mask_full == 0].astype(int) + 40, 0, 255).astype(np.uint8)
    overlay[:, :, 2] = r_ch
    overlay[:, :, 1] = g_ch
    p6 = overlay
    save_panel(p6, os.path.join(out_dir, "06_routing_overlay.png"),
               "Routing Overlay (green=keep, red=prune)")

    # Panel 7 — sparse DA2 (after pruning + re-assembly)
    p7 = colorize(sparse_norm, cmap="Spectral_r", vmin=0.0, vmax=1.0)
    p7 = resize_to_match(p7, left_bgr)
    ras_label = "+reassembly" if not args.no_reassembly else ""
    save_panel(p7, os.path.join(out_dir, "07_sparse_da2_depth.png"),
               f"Sparse DA2 — SGM-ViT ({100*prune_ratio:.1f}% pruned, "
               f"block={args.prune_layer}{ras_label})")

    # Panel 8 — disparity-aligned monocular depth
    p8 = colorize(disp_aligned, cmap="plasma")
    p8 = resize_to_match(p8, left_bgr)
    align_label = f"s={scale:.4f} t={shift:.4f}" if not (args.no_align or args.no_sgm) else "no alignment"
    save_panel(p8, os.path.join(out_dir, "08_aligned_disparity.png"),
               f"Mono Depth Aligned to SGM Disparity Space [{align_label}] (px)")

    # Panel 9 — fused SGM + Dense DA2
    p9_fused = colorize(disp_fused, cmap="plasma")
    p9_fused = resize_to_match(p9_fused, left_bgr)
    fuse_label = (f"θ_conf={args.conf_threshold:.2f}"
                  if not (args.no_align or args.no_sgm) else "no fusion")
    save_panel(p9_fused, os.path.join(out_dir, "09_fused_sgm_dense_da2.png"),
               f"Fused SGM + Dense DA2 [{fuse_label}]")

    # Panel 9b — fused SGM + Sparse DA2
    p9b_fused = colorize(disp_fused_sparse, cmap="plasma")
    p9b_fused = resize_to_match(p9b_fused, left_bgr)
    save_panel(p9b_fused, os.path.join(out_dir, "09b_fused_sgm_sparse_da2.png"),
               f"Fused SGM + Sparse DA2 [{fuse_label}]")

    # Panel 10 — diff map |Dense − Sparse|
    p10 = colorize(diff_map, cmap="hot", vmin=0.0, vmax=diff_map.max())
    p10 = resize_to_match(p10, left_bgr)
    save_panel(p10, os.path.join(out_dir, "10_diff_map.png"),
               f"|Dense − Sparse|  mean={diff_map.mean():.4f}  (hot)")

    # Panel 11 — 4-panel paper comparison figure
    p4_normed = colorize(dense_norm, cmap="Spectral_r", vmin=0.0, vmax=1.0)
    p4_normed = resize_to_match(p4_normed, left_bgr)

    build_comparison_figure(
        da2_bgr        = p4_normed,
        sparse_bgr     = p7,
        aligned_bgr    = p8,
        diff_bgr       = p10,
        prune_ratio    = prune_ratio,
        attn_reduction = attn_reduction,
        threshold      = args.threshold,
        scale          = scale,
        shift          = shift,
        out_path       = os.path.join(out_dir, "11_comparison.png"),
        aligned_is_disparity = not (args.no_align or args.no_sgm),
    )

    # Summary 3×4 composite
    summary_panels = [
        ("(1) Input",                                        p1),
        ("(2) SGM Disparity",                               p2),
        (f"(3) SGM Confidence (μ={confidence_map.mean():.2f})", p3),
        (f"(4) Dense DA2 — Baseline ({args.encoder})",      p4),
        (f"(5) Token Grid (prune {100*prune_ratio:.1f}%)",  grid_img),
        ("(6) Routing Overlay",                             p6),
        (f"(7) Sparse DA2 (block={args.prune_layer}{ras_label})", p7),
        (f"(8) Aligned Disparity [{align_label}]",           p8),
        (f"(9) Fused SGM + Dense DA2 [{fuse_label}]",       p9_fused),
        (f"(9b) Fused SGM + Sparse DA2 [{fuse_label}]",    p9b_fused),
        (f"(10) |Dense−Sparse| (attn↓{100*attn_reduction:.1f}%)", p10),
    ]
    build_summary_figure(summary_panels, os.path.join(out_dir, "00_summary.png"), ncol=4)

    # ------------------------------------------------------------------
    # Final stats
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Demo complete — results in: {out_dir}/")
    print(f"{'='*62}")
    print(f"  Encoder           : DepthAnythingV2-{args.encoder.upper()}")
    print(f"  Token grid        : {TOKEN_GRID_SIZE}×{TOKEN_GRID_SIZE} = {N} tokens")
    print(f"  Threshold θ       : {args.threshold}")
    print(f"  Prune-layer       : block {args.prune_layer} / {n_blocks-1}")
    print(f"  Re-assembly       : {'enabled' if not args.no_reassembly else 'disabled'}")
    print(f"  Pruning ratio     : {100*prune_ratio:.1f}%  ({n_prune}/{N} tokens)")
    print(f"  Attn FLOPs ↓      : ~{100*attn_reduction:.1f}%")
    print(f"  Mean |Dense−Sparse|: {diff_map.mean():.4f}")
    if not (args.no_align or args.no_sgm):
        print(f"  Dense  align (s,t): ({scale_dense:.4f}, {shift_dense:.4f})")
        print(f"  Sparse align (s,t): ({scale:.4f}, {shift:.4f})")
        print(f"  Aligned disp range: [{disp_aligned.min():.2f}, {disp_aligned.max():.2f}] px")
        print(f"  Fused(Dense) range: [{disp_fused.min():.2f}, {disp_fused.max():.2f}] px")
        print(f"  Fused(Sparse)range: [{disp_fused_sparse.min():.2f}, {disp_fused_sparse.max():.2f}] px")
        print(f"  Fusion strategy   : {args.fusion_strategy}")
        print(f"  Fusion conf thresh: {args.conf_threshold}")
    print(f"{'='*62}")
    print(f"  Key outputs:")
    print(f"    04  — Dense DA2 baseline")
    print(f"    07  — Sparse DA2 (SGM-ViT) with re-assembly")
    print(f"    08  — Mono depth aligned to SGM disparity space (px)")
    print(f"    09  — Fused SGM + Dense DA2 (SGM where confident, DA2 elsewhere)")
    print(f"    09b — Fused SGM + Sparse DA2 (GAS-accelerated fusion)")
    print(f"    11  — 4-panel paper comparison figure")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SGM-ViT demo: SGM + DepthAnythingV2 + token routing "
                    "+ progressive pruning + re-assembly + metric alignment."
    )
    p.add_argument("--left",
        default=os.path.join(_SCRIPT_DIR, "asserts", "left",  "000005_10.png"))
    p.add_argument("--right",
        default=os.path.join(_SCRIPT_DIR, "asserts", "right", "000005_10.png"))
    p.add_argument("--weights",
        default=os.path.join(_SCRIPT_DIR, "Depth-Anything-V2", "checkpoints",
                             "depth_anything_v2_vits.pth"))
    p.add_argument("--encoder", default="vits",
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument("--threshold", type=float, default=0.65,
        help="SGM confidence threshold θ for token pruning (default: 0.65).")
    p.add_argument("--prune-layer", type=int, default=0,
        help="ViT block index (0-based) at which pruning starts. "
             "0 = earliest (max FLOPs savings); higher = better quality. "
             "For ViT-S: 0..11. (default: 0)")
    p.add_argument("--sparse-mode", default="mask",
        choices=["zero", "mask"],
        help="Sparse attention mode: 'mask' = GAS (Gather-Attend-Scatter, "
             "physically excludes pruned tokens from attention), "
             "'zero' = legacy token zeroing. (default: mask)")
    p.add_argument("--no-reassembly", action="store_true",
        help="Disable token re-assembly before the DPT decoder.")
    p.add_argument("--disparity-range", type=int, default=128)
    p.add_argument("--conf-sigma", type=float, default=5.0,
        help="Gaussian σ for SGM confidence map smoothing.")
    p.add_argument("--no-align", action="store_true",
        help="Skip disparity-space alignment of monocular depth.")
    p.add_argument("--conf-threshold", type=float, default=0.5,
        help="Confidence threshold for depth alignment and SGM/DA2 fusion "
             "blend boundary (default: 0.5).")
    p.add_argument("--fusion-strategy", default="soft_blend", #choices=hard_switch
        choices=list(FUSION_STRATEGIES.keys()),
        help="Fusion strategy for SGM + DA2 combination (default: soft_blend).")
    p.add_argument("--outlier-threshold", type=float, default=10.0,
        help="Disparity disagreement threshold for outlier_aware fusion "
             "(default: 10.0 px).")
    p.add_argument("--theta-low", type=float, default=0.3,
        help="Lower confidence threshold for two_threshold fusion (default: 0.3).")
    p.add_argument("--theta-high", type=float, default=0.7,
        help="Upper confidence threshold for two_threshold fusion (default: 0.7).")
    p.add_argument("--out-dir",
        default=os.path.join(_SCRIPT_DIR, "results", "demo"))
    p.add_argument("--no-sgm", action="store_true",
        help="Skip SGM (uniform confidence=0.5). Fast path for testing DAv2.")
    p.add_argument("--cpu", action="store_true",
        help="Force CPU even if CUDA is available.")
    return p.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
