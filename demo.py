"""
demo.py — SGM-ViT End-to-End Demo
===================================
Demonstrates the full SGM-ViT pipeline on a stereo image pair:

  1. SGM stereo matching  →  disparity map + confidence map
  2. DepthAnythingV2 (ViT-S)  →  monocular depth map
  3. SGMConfidenceTokenRouter  →  keep / prune token split + statistics
  4. Multi-panel visualisation  →  saved to results/demo/

Usage
-----
  python demo.py                         # use default KITTI test pair
  python demo.py --no-sgm               # skip SGM (use uniform confidence)
  python demo.py --threshold 0.5        # change pruning threshold
  python demo.py --help

Expected outputs (in results/demo/)
-------------------------------------
  01_left_image.png          — input left image (BGR)
  02_sgm_disparity.png       — filled disparity map (plasma colourmap)
  03_sgm_confidence.png      — per-pixel SGM confidence (RdYlGn colourmap)
  04_da2_depth.png           — DepthAnythingV2 raw depth (Spectral_r colourmap)
  05_token_routing.png       — 37×37 token keep/prune grid
  06_routing_overlay.png     — routing decision overlaid on input image
  07_fusion_depth.png        — SGM-ViT fused depth (SGM fills pruned, DA2 keeps rest)
  08_diff_map.png            — per-pixel absolute diff |DA2 - fused| (hot colourmap)
  09_comparison_da2_fused.png— side-by-side paper figure: raw DA2 vs SGM-ViT fused
  00_summary.png             — 3×3 composite panel (paper-ready figure)

Runtime notes
-------------
  • On the FIRST run, Numba JIT compilation of the SGM kernels takes 30-90 s.
  • Subsequent runs use the cached compiled code and are much faster (~5 s).
  • Use --no-sgm to skip SGM entirely and test only the DAv2 + router path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless backend — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ---------------------------------------------------------------------------
# Path setup — run from project root or any location
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DA2_DIR     = os.path.join(_SCRIPT_DIR, "Depth-Anything-V2")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _DA2_DIR not in sys.path:
    sys.path.insert(0, _DA2_DIR)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from core.sgm_wrapper import run_sgm_with_confidence, confidence_to_token_grid
from core.token_router import SGMConfidenceTokenRouter

# DepthAnythingV2 — imported after sys.path update above
from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ViT-S architecture config (must match the checkpoint)
DA2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# ViT token grid for 518-px input with patch_size=14: 518 // 14 = 37
TOKEN_GRID_SIZE = 37

# ViT-S embed dim
EMBED_DIM_MAP = {"vits": 384, "vitb": 768, "vitl": 1024}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_da2_model(encoder: str, weights_path: str, device: torch.device) -> DepthAnythingV2:
    """Load a DepthAnythingV2 model from a checkpoint file."""
    cfg = DA2_MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def colorize(arr: np.ndarray, cmap: str = "plasma", vmin=None, vmax=None) -> np.ndarray:
    """Map a 2-D float array to an 8-bit BGR image via a matplotlib colourmap."""
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    cm   = plt.get_cmap(cmap)
    rgba = (cm(norm) * 255).astype(np.uint8)           # (H, W, 4)
    bgr  = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr


def resize_to_match(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize img (H, W, C) or (H, W) to match target's spatial dimensions."""
    th, tw = target.shape[:2]
    if img.ndim == 2:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def draw_token_grid(
    token_conf: np.ndarray,
    threshold: float,
    cell_px: int = 14,
) -> np.ndarray:
    """
    Render the 37×37 token routing grid as an RGB image.

    Colour coding:
      GREEN   — token is KEPT   (conf ≤ threshold → needs ViT attention)
      RED     — token is PRUNED (conf > threshold → SGM already reliable)
    """
    G     = token_conf.shape[0]
    H_img = G * cell_px
    W_img = G * cell_px
    canvas = np.zeros((H_img, W_img, 3), dtype=np.uint8)

    prune_mask = token_conf > threshold  # True → pruned

    for r in range(G):
        for c in range(G):
            y0, y1 = r * cell_px, (r + 1) * cell_px
            x0, x1 = c * cell_px, (c + 1) * cell_px
            if prune_mask[r, c]:
                colour = (220, 60, 60)     # Red  → pruned
            else:
                colour = (60, 180, 60)     # Green → kept
            canvas[y0:y1, x0:x1] = colour
            # Cell border
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (30, 30, 30), 1)

    # Legend
    legend_h = 28
    legend   = np.full((legend_h, W_img, 3), 40, dtype=np.uint8)
    cv2.putText(legend, f"GREEN=keep  RED=prune  theta={threshold:.2f}",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return np.vstack([canvas, legend])


def save_panel(img_bgr: np.ndarray, path: str, title: str | None = None) -> None:
    """Save a BGR image with an optional white title bar."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if title:
        bar = np.full((30, img_bgr.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(bar, title, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        img_bgr = np.vstack([bar, img_bgr])
    cv2.imwrite(path, img_bgr)
    print(f"  [saved] {path}")


def build_summary_figure(
    panels: list[tuple[str, np.ndarray]],
    out_path: str,
    ncol: int = 3,
) -> None:
    """Create an N×ncol Matplotlib figure from BGR panels and save as PNG."""
    n    = len(panels)
    nrow = (n + ncol - 1) // ncol

    fig = plt.figure(figsize=(ncol * 6, nrow * 5.5), constrained_layout=True)
    fig.suptitle(
        "SGM-ViT Demo: SGM Confidence-Guided Token Routing + DepthAnythingV2",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)

    for idx, (title, bgr) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // ncol, idx % ncol])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for idx in range(n, nrow * ncol):
        fig.add_subplot(gs[idx // ncol, idx % ncol]).axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def fuse_depth_maps(
    da2_depth: np.ndarray,
    sgm_disp_norm: np.ndarray,
    confidence_map: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse DA2 depth and SGM disparity using the token routing decision.

    Strategy
    --------
    SGM disparity (larger = closer) is converted to a depth-like representation
    by inversion: sgm_depth = 1 - disp_norm.  Both maps are independently
    normalised to [0, 1] before blending.

    The per-pixel fusion gate is the confidence map itself:
        fused = (1 - conf) * da2_norm + conf * sgm_depth_norm
                ─────────────────────   ───────────────────────
                DA2 dominates when       SGM dominates when
                conf is low (uncertain)  conf is high (reliable)

    This matches the token-routing logic exactly:
      • conf > θ  →  "prune"  →  SGM weight ≈ 1 (SGM fills the depth)
      • conf ≤ θ  →  "keep"   →  DA2 weight ≈ 1 (ViT attention used)

    Returns
    -------
    fused_norm : (H, W) float32 [0, 1]
        Fused depth map in display space (higher = farther).
    diff_map   : (H, W) float32 [0, 1]
        Per-pixel absolute difference |da2_norm - fused_norm|,
        normalised to [0, 1].  Highlights regions altered by SGM fusion.
    """
    # -- Step 1: normalise DA2 depth to [0, 1] (higher = farther) --
    d_min, d_max = da2_depth.min(), da2_depth.max()
    da2_norm = (da2_depth - d_min) / (d_max - d_min + 1e-8)

    # -- Step 2: convert SGM disparity → depth-like (invert) --
    # Disparity is inversely proportional to depth:
    #   larger disparity ↔ closer   →   invert so higher value = farther
    sgm_depth_norm = 1.0 - np.clip(sgm_disp_norm, 0.0, 1.0)

    # -- Step 3: resize SGM to match DA2 resolution if needed --
    if sgm_depth_norm.shape != da2_norm.shape:
        sgm_depth_norm = cv2.resize(
            sgm_depth_norm, (da2_norm.shape[1], da2_norm.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    conf = cv2.resize(
        confidence_map, (da2_norm.shape[1], da2_norm.shape[0]),
        interpolation=cv2.INTER_AREA,
    ) if confidence_map.shape != da2_norm.shape else confidence_map.copy()

    # -- Step 4: soft confidence-weighted blend --
    fused = (1.0 - conf) * da2_norm + conf * sgm_depth_norm
    fused = fused.astype(np.float32)

    # -- Step 5: absolute difference map (shows SGM contribution) --
    diff = np.abs(da2_norm - fused).astype(np.float32)
    d_max_diff = diff.max()
    diff_norm = diff / (d_max_diff + 1e-8)

    return fused, diff_norm


def build_comparison_figure(
    da2_bgr: np.ndarray,
    fused_bgr: np.ndarray,
    diff_bgr: np.ndarray,
    prune_ratio: float,
    attn_reduction: float,
    threshold: float,
    out_path: str,
) -> None:
    """
    Build a paper-ready 1×3 comparison figure:
      [Raw DA2 depth]  |  [SGM-ViT Fused]  |  [Diff map]

    The figure is annotated with pruning statistics and saved at high DPI.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5), constrained_layout=True)

    panels = [
        (da2_bgr,    "Raw DepthAnythingV2",             "No SGM prior"),
        (fused_bgr,  "SGM-ViT Fused Depth",
         f"SGM fills {100*prune_ratio:.1f}% tokens  |  Attn FLOPs↓{100*attn_reduction:.1f}%"),
        (diff_bgr,   "|DA2 − Fused| Difference Map",    f"θ = {threshold:.2f}"),
    ]

    for ax, (bgr, title, subtitle) in zip(axes, panels):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=4)
        ax.set_xlabel(subtitle, fontsize=10, color="#555555")
        ax.axis("off")

    fig.suptitle(
        "SGM-ViT: Confidence-Guided Depth Fusion — Raw DA2 vs. SGM-ViT Output",
        fontsize=13, fontweight="bold", y=1.01,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  SGM-ViT Demo")
    print(f"{'='*60}")
    print(f"  Left  : {args.left}")
    print(f"  Right : {args.right}")
    print(f"  Weights: {args.weights}")
    print(f"  Device : {device}")
    print(f"  Output : {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Step 1: Load left image (kept as BGR for OpenCV + RGB for matplotlib)
    # ------------------------------------------------------------------
    left_bgr = cv2.imread(args.left)
    if left_bgr is None:
        raise FileNotFoundError(f"Cannot load left image: {args.left}")
    H_orig, W_orig = left_bgr.shape[:2]
    print(f"[1/5] Left image loaded: {W_orig}×{H_orig} px")

    # ------------------------------------------------------------------
    # Step 2: SGM stereo matching  →  disparity + confidence
    # ------------------------------------------------------------------
    if args.no_sgm:
        print("[2/5] Skipping SGM (--no-sgm).  Using uniform confidence = 0.5")
        disparity_norm = np.zeros((H_orig, W_orig), dtype=np.float32)
        confidence_map = np.full((H_orig, W_orig), 0.5, dtype=np.float32)
    else:
        print("[2/5] Running SGM stereo matching ...")
        print("      (Numba JIT compilation on first run: ~30-90 s)")
        t_sgm = time.perf_counter()
        disparity_norm, confidence_map, _ = run_sgm_with_confidence(
            left_path       = args.left,
            right_path      = args.right,
            disparity_range = args.disparity_range,
            smooth_sigma    = args.conf_sigma,
            verbose         = True,
        )
        print(f"      SGM total time: {time.perf_counter()-t_sgm:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: DepthAnythingV2 monocular depth estimation
    # ------------------------------------------------------------------
    print(f"[3/5] Loading DepthAnythingV2 ({args.encoder}) from {args.weights} ...")
    model = load_da2_model(args.encoder, args.weights, device)

    print("[3/5] Running depth inference ...")
    t_da2 = time.perf_counter()
    depth_map = model.infer_image(left_bgr, input_size=518)   # (H, W) float32
    print(f"      DAv2 inference time: {time.perf_counter()-t_da2:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 4: Token routing analysis
    # ------------------------------------------------------------------
    print(f"[4/5] Token routing (threshold θ={args.threshold}) ...")
    router = SGMConfidenceTokenRouter(
        token_grid_size      = TOKEN_GRID_SIZE,
        confidence_threshold = args.threshold,
        learnable_threshold  = False,
    )

    embed_dim  = EMBED_DIM_MAP[args.encoder]
    N          = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE    # 1369 tokens

    # Build a (1, 1, H, W) confidence tensor for the router
    conf_tensor = torch.from_numpy(confidence_map).unsqueeze(0).unsqueeze(0)

    # Use zero tokens for this routing-only analysis pass
    # (in the full model, these would be real ViT patch embeddings)
    dummy_tokens = torch.zeros(1, N, embed_dim)

    with torch.no_grad():
        routing = router(conf_tensor, dummy_tokens)

    n_keep  = len(routing["keep_idx"][0])
    n_prune = len(routing["prune_idx"][0])
    prune_ratio = routing["prune_ratio"]

    print(f"      Total tokens  : {N}")
    print(f"      Kept (attend) : {n_keep}  ({100*(1-prune_ratio):.1f}%)")
    print(f"      Pruned (skip) : {n_prune}  ({100*prune_ratio:.1f}%)")

    # FLOPs reduction estimate (attention scales as O(N²))
    attn_reduction = 1.0 - (n_keep / N) ** 2
    print(f"      Attention FLOPs reduction : {100*attn_reduction:.1f}%\n")

    # Token-grid confidence (for visualisation)
    token_conf = confidence_to_token_grid(confidence_map, TOKEN_GRID_SIZE)

    # ------------------------------------------------------------------
    # Step 4.5: Compute SGM-ViT fused depth map
    # ------------------------------------------------------------------
    # Fuse DA2 depth (full-res float) and SGM disparity (image-res float)
    # using the per-pixel confidence map as a soft blend gate.
    # High confidence (conf > θ) → SGM dominates (replaces ViT depth)
    # Low confidence (conf ≤ θ) → DA2 dominates (ViT attention preserved)
    depth_fused, diff_map = fuse_depth_maps(
        da2_depth      = depth_map,
        sgm_disp_norm  = disparity_norm,
        confidence_map = confidence_map,
        threshold      = args.threshold,
    )

    # ------------------------------------------------------------------
    # Step 5: Build & save visualisations
    # ------------------------------------------------------------------
    print("[5/5] Saving visualisations ...")

    # --- Panel 1: Left image ---
    p1 = left_bgr.copy()
    save_panel(p1, os.path.join(out_dir, "01_left_image.png"), "Input Left Image")

    # --- Panel 2: SGM disparity ---
    p2 = colorize(disparity_norm, cmap="plasma")
    p2 = resize_to_match(p2, left_bgr)
    save_panel(p2, os.path.join(out_dir, "02_sgm_disparity.png"), "SGM Disparity Map")

    # --- Panel 3: SGM confidence ---
    p3 = colorize(confidence_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    p3 = resize_to_match(p3, left_bgr)
    save_panel(p3, os.path.join(out_dir, "03_sgm_confidence.png"),
               f"SGM Confidence Map (μ={confidence_map.mean():.2f})")

    # --- Panel 4: DAv2 depth ---
    p4 = colorize(depth_map, cmap="Spectral_r")
    p4 = resize_to_match(p4, left_bgr)
    save_panel(p4, os.path.join(out_dir, "04_da2_depth.png"),
               f"DepthAnythingV2 ({args.encoder}) Depth Map")

    # --- Panel 5: Token routing grid ---
    grid_img = draw_token_grid(token_conf, threshold=args.threshold, cell_px=14)
    save_panel(grid_img, os.path.join(out_dir, "05_token_routing.png"),
               f"Token Routing (θ={args.threshold:.2f}, "
               f"prune={100*prune_ratio:.1f}%, FLOPs↓{100*attn_reduction:.1f}%)")

    # --- Panel 6: Confidence + routing overlay on left image ---
    # Resize token routing mask to image resolution for overlay
    prune_mask_full = cv2.resize(
        (token_conf > args.threshold).astype(np.uint8) * 255,
        (W_orig, H_orig), interpolation=cv2.INTER_NEAREST,
    )
    overlay = left_bgr.copy()
    red_channel  = overlay[:, :, 2]
    green_channel = overlay[:, :, 1]
    # Tint pruned regions red, kept regions green (subtle blend)
    red_channel[prune_mask_full > 0]   = np.clip(
        red_channel[prune_mask_full > 0].astype(int) + 60, 0, 255).astype(np.uint8)
    green_channel[prune_mask_full == 0] = np.clip(
        green_channel[prune_mask_full == 0].astype(int) + 40, 0, 255).astype(np.uint8)
    overlay[:, :, 2] = red_channel
    overlay[:, :, 1] = green_channel
    p6 = overlay
    save_panel(p6, os.path.join(out_dir, "06_routing_overlay.png"),
               "Routing Overlay (green=keep, red=prune)")

    # --- Panel 7: SGM-ViT fused depth ---
    # Use the same Spectral_r colourmap as DA2 for direct visual comparison
    p7 = colorize(depth_fused, cmap="Spectral_r", vmin=0.0, vmax=1.0)
    p7 = resize_to_match(p7, left_bgr)
    save_panel(p7, os.path.join(out_dir, "07_fusion_depth.png"),
               f"SGM-ViT Fused Depth (conf-weighted, θ={args.threshold:.2f})")

    # --- Panel 8: Difference map |DA2 - fused| ---
    # Hot colourmap: brighter = larger deviation introduced by SGM fusion.
    # These are the regions where SGM meaningfully changed the DA2 output.
    p8 = colorize(diff_map, cmap="hot", vmin=0.0, vmax=1.0)
    p8 = resize_to_match(p8, left_bgr)
    save_panel(p8, os.path.join(out_dir, "08_diff_map.png"),
               "|DA2 raw - SGM-ViT fused|  (brighter = larger SGM contribution)")

    # --- Panel 9: Paper comparison strip: raw DA2 vs fused ---
    # Re-render DA2 on the same [0,1] normalised scale as fused for fair comparison
    da2_norm_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    p4_normed = colorize(da2_norm_vis, cmap="Spectral_r", vmin=0.0, vmax=1.0)
    p4_normed = resize_to_match(p4_normed, left_bgr)

    build_comparison_figure(
        da2_bgr        = p4_normed,
        fused_bgr      = p7,
        diff_bgr       = p8,
        prune_ratio    = prune_ratio,
        attn_reduction = attn_reduction,
        threshold      = args.threshold,
        out_path       = os.path.join(out_dir, "09_comparison_da2_fused.png"),
    )

    # --- Summary figure (3×3 grid) ---
    panels = [
        ("(1) Input Left Image",                         p1),
        ("(2) SGM Disparity",                            p2),
        (f"(3) SGM Confidence (μ={confidence_map.mean():.2f})", p3),
        (f"(4) DA2 Raw Depth ({args.encoder})",          p4),
        (f"(5) Token Grid (prune {100*prune_ratio:.1f}%)", grid_img),
        ("(6) Routing Overlay",                          p6),
        (f"(7) SGM-ViT Fused Depth (θ={args.threshold:.2f})", p7),
        (f"(8) |DA2 − Fused| Diff (attn↓{100*attn_reduction:.1f}%)", p8),
    ]
    build_summary_figure(panels, os.path.join(out_dir, "00_summary.png"), ncol=4)

    # ------------------------------------------------------------------
    # Final stats
    # ------------------------------------------------------------------
    sgm_contribution = float(diff_map.mean())
    print(f"\n{'='*60}")
    print(f"  Demo complete.  Results saved to: {out_dir}/")
    print(f"{'='*60}")
    print(f"  Encoder         : DepthAnythingV2-{args.encoder.upper()}")
    print(f"  Token grid      : {TOKEN_GRID_SIZE}×{TOKEN_GRID_SIZE} = {N} tokens")
    print(f"  Threshold θ     : {args.threshold}")
    print(f"  Pruning ratio   : {100*prune_ratio:.1f}%  ({n_prune}/{N} tokens skipped)")
    print(f"  Attn FLOPs ↓    : ~{100*attn_reduction:.1f}%  (quadratic attention)")
    print(f"  SGM contribution: mean |DA2-fused| = {sgm_contribution:.4f}  "
          f"(higher = SGM changed more pixels)")
    print(f"{'='*60}")
    print(f"  Key outputs:")
    print(f"    04_da2_depth.png           — raw DepthAnythingV2")
    print(f"    07_fusion_depth.png        — SGM-ViT fused depth")
    print(f"    09_comparison_da2_fused.png— side-by-side comparison (paper figure)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SGM-ViT end-to-end demo: SGM + DepthAnythingV2 + token routing."
    )
    p.add_argument(
        "--left",
        default=os.path.join(_SCRIPT_DIR, "asserts", "left",  "000005_10.png"),
        help="Path to left stereo image.",
    )
    p.add_argument(
        "--right",
        default=os.path.join(_SCRIPT_DIR, "asserts", "right", "000005_10.png"),
        help="Path to right stereo image.",
    )
    p.add_argument(
        "--weights",
        default=os.path.join(_SCRIPT_DIR, "Depth-Anything-V2", "checkpoints",
                             "depth_anything_v2_vits.pth"),
        help="Path to DepthAnythingV2 checkpoint (.pth).",
    )
    p.add_argument(
        "--encoder", default="vits", choices=list(DA2_MODEL_CONFIGS.keys()),
        help="ViT encoder variant (must match weights file).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.65,
        help="SGM confidence threshold θ for token pruning (default: 0.65).",
    )
    p.add_argument(
        "--disparity-range", type=int, default=128,
        help="SGM max disparity (multiple of 4, default: 128).",
    )
    p.add_argument(
        "--conf-sigma", type=float, default=5.0,
        help="Gaussian smoothing σ for confidence map (0 = no smoothing).",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "results", "demo"),
        help="Output directory for saved images.",
    )
    p.add_argument(
        "--no-sgm", action="store_true",
        help="Skip SGM (use uniform confidence=0.5). Fast path for testing DAv2.",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
