#!/usr/bin/env python3
"""Merge-centric demo for SGM-ViT."""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import core._paths  # noqa: F401, E402

from core.decoder_adaptive_precision import (  # noqa: E402
    build_decoder_sensitivity_map,
    build_stage_high_precision_mask,
)
from core.fusion import fuse_dispatch, fuse_edge_aware_residual  # noqa: E402
from core.pipeline import (  # noqa: E402
    DA2_MODEL_CONFIGS,
    align_depth_to_sgm,
    load_da2_model,
    run_decoder_weight_caps_merged_da2,
    run_token_merged_da2,
)
from core.sgm_wrapper import run_sgm_with_confidence  # noqa: E402
from core.viz import build_summary_figure, colorize, resize_to_match, save_panel  # noqa: E402
from scripts.common_config import (  # noqa: E402
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_EDGE_DETAIL_SUPPRESSION,
    DEFAULT_EDGE_RESIDUAL_GAIN,
    DEFAULT_EDGE_THETA_HIGH,
    DEFAULT_EDGE_THETA_LOW,
    DEFAULT_ENCODER,
    DEFAULT_FUSION_BACKEND,
    DEFAULT_FUSION_NET_WEIGHTS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_WEIGHTS,
    default_results_dir,
)

try:  # noqa: E402
    from core.fusion_net import (
        build_fusion_inputs,
        get_fusion_net_runtime_config,
        load_fusion_net,
        run_fusion_net_refinement,
    )
except ImportError:  # pragma: no cover - optional module on some machines
    build_fusion_inputs = None
    get_fusion_net_runtime_config = None
    load_fusion_net = None
    run_fusion_net_refinement = None


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


LEGACY_OUTPUT_FILES = [
    "00_summary.png",
    "01_left_image.png",
    "02_sgm_disparity.png",
    "03_confidence.png",
    "04_dense_aligned.png",
    "04_gt_disparity.png",
    "05_dense_aligned.png",
    "05_merge_aligned.png",
    "06_merge_aligned.png",
    "06_wcaps_aligned.png",
    "07_fused_sgm_merge.png",
    "07_wcaps_aligned.png",
    "08_fused_sgm_merge.png",
    "08_fused_sgm_wcaps.png",
    "09_fused_sgm_wcaps.png",
    "09_heuristic_fused_wcaps.png",
    "09_wcaps_hp_mask.png",
    "10_diff_merge_vs_wcaps.png",
    "10_fused_sgm_wcaps.png",
    "10_wcaps_hp_mask.png",
    "11_detail_score.png",
    "11_diff_merge_vs_wcaps.png",
    "11_fusion_net_residual.png",
    "12_alpha_conf.png",
    "12_detail_score.png",
    "12_heuristic_vs_net_diff.png",
    "13_alpha_conf.png",
    "13_alpha_eff.png",
    "13_wcaps_hp_mask.png",
    "14_alpha_eff.png",
    "14_diff_merge_vs_wcaps.png",
    "15_detail_score.png",
    "16_alpha_conf.png",
    "17_alpha_eff.png",
]


def disparity_valid_mask(disp: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_mask is not None:
        return valid_mask.astype(bool) & np.isfinite(disp) & (disp > 0)
    return np.isfinite(disp) & (disp > 0)


def compute_disparity_display_range(
    disparity_maps: list[np.ndarray],
    gt_disp: np.ndarray | None = None,
) -> tuple[float, float]:
    if gt_disp is not None:
        valid_vals = gt_disp[disparity_valid_mask(gt_disp)]
    else:
        valid_chunks = [disp[disparity_valid_mask(disp)] for disp in disparity_maps if disp is not None]
        valid_chunks = [vals for vals in valid_chunks if vals.size > 0]
        valid_vals = np.concatenate(valid_chunks, axis=0) if valid_chunks else np.array([], dtype=np.float32)

    if valid_vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(valid_vals, 1.0))
    vmax = float(np.percentile(valid_vals, 99.0))
    if vmax <= vmin + 1e-6:
        vmin = float(valid_vals.min())
        vmax = float(valid_vals.max())
    if vmax <= vmin + 1e-6:
        vmax = vmin + 1.0
    return vmin, vmax


def colorize_disparity_shared(
    disp: np.ndarray,
    vmin: float,
    vmax: float,
    valid_mask: np.ndarray | None = None,
    cmap: str = "plasma",
) -> np.ndarray:
    disp = disp.astype(np.float32)
    valid = disparity_valid_mask(disp, valid_mask)
    if not np.any(valid):
        return colorize(np.zeros_like(disp, dtype=np.float32), cmap=cmap, vmin=0.0, vmax=1.0)

    colored = colorize(np.where(valid, disp, vmin).astype(np.float32), cmap=cmap, vmin=vmin, vmax=vmax)
    colored[~valid] = 0
    return colored


def cleanup_demo_outputs(out_dir: str) -> None:
    for name in LEGACY_OUTPUT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        assert header in (b"PF", b"Pf"), f"Not a PFM file: {path}"
        w, h = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f").reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


def load_gt_disparity(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot load GT disparity: {path}")
    if path.lower().endswith(".pfm"):
        return read_pfm(path)
    if path.lower().endswith(".png"):
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise FileNotFoundError(f"Cannot load GT disparity: {path}")
        return raw.astype(np.float32) / 256.0
    raise ValueError(f"Unsupported GT disparity format: {path}")


def maybe_align(
    depth_map: np.ndarray,
    sgm_disp: np.ndarray | None,
    conf_map: np.ndarray | None,
    do_align: bool,
    conf_threshold: float,
) -> np.ndarray:
    if not do_align or sgm_disp is None or conf_map is None:
        return depth_map.astype(np.float32)
    disp_aligned, _, _ = align_depth_to_sgm(
        depth_mono=depth_map,
        disparity_raw=sgm_disp,
        confidence_map=conf_map,
        conf_threshold=conf_threshold,
    )
    return disp_aligned.astype(np.float32)


def run_demo(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    cleanup_demo_outputs(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    left_bgr = cv2.imread(args.left)
    if left_bgr is None:
        raise FileNotFoundError(f"Cannot load left image: {args.left}")
    h_orig, w_orig = left_bgr.shape[:2]

    print(f"Demo input: {args.left}")
    print(f"Output dir: {args.out_dir}")
    print(f"Device: {device}")

    gt_disp = load_gt_disparity(args.gt_disparity)

    disp_raw = None
    confidence_map = None
    if args.no_sgm:
        sgm_disp_display = np.zeros((h_orig, w_orig), dtype=np.float32)
        confidence_map = np.full((h_orig, w_orig), 0.5, dtype=np.float32)
    else:
        t0 = time.perf_counter()
        _, confidence_map, disp_raw = run_sgm_with_confidence(
            left_path=args.left,
            right_path=args.right,
            disparity_range=args.disparity_range,
            smooth_sigma=args.conf_sigma,
            verbose=True,
        )
        sgm_disp_display = disp_raw.astype(np.float32)
        print(f"SGM time: {time.perf_counter() - t0:.2f}s")

    model = load_da2_model(args.encoder, args.weights, device)

    t_dense = time.perf_counter()
    depth_dense = model.infer_image(left_bgr, input_size=args.input_size)
    print(f"Dense time: {time.perf_counter() - t_dense:.2f}s")

    t_merge = time.perf_counter()
    depth_merge = run_token_merged_da2(
        model=model,
        image_bgr=left_bgr,
        confidence_map=confidence_map,
        keep_ratio=args.merge_keep_ratio,
        input_size=args.input_size,
        merge_layer=args.merge_layer,
    )
    print(f"Merge time: {time.perf_counter() - t_merge:.2f}s")

    t_wcaps = time.perf_counter()
    depth_wcaps = run_decoder_weight_caps_merged_da2(
        model=model,
        image_bgr=left_bgr,
        confidence_map=confidence_map,
        keep_ratio=args.adaptive_keep_ratio,
        input_size=args.input_size,
        merge_layer=args.merge_layer,
        decoder_high_precision_ratio=args.decoder_high_precision_ratio,
        low_precision_bits=args.decoder_low_bits,
        decoder_conf_weight=args.decoder_conf_weight,
        decoder_texture_weight=0.0,
        decoder_variance_weight=0.0,
        decoder_stage_policy="coarse_only",
    )
    print(f"W-CAPS time: {time.perf_counter() - t_wcaps:.2f}s")

    do_align = (not args.no_align) and (disp_raw is not None) and (confidence_map is not None)
    dense_aligned = maybe_align(depth_dense, disp_raw, confidence_map, do_align, args.align_conf_threshold)
    merge_aligned = maybe_align(depth_merge, disp_raw, confidence_map, do_align, args.align_conf_threshold)
    wcaps_aligned = maybe_align(depth_wcaps, disp_raw, confidence_map, do_align, args.align_conf_threshold)

    if do_align:
        sgm_for_fuse = resize_to_match(disp_raw.astype(np.float32), merge_aligned)
        conf_for_fuse = resize_to_match(confidence_map.astype(np.float32), merge_aligned)
        fused_merge = fuse_dispatch(
            strategy=args.fusion_strategy,
            sgm_disp=sgm_for_fuse,
            da2_aligned=merge_aligned,
            confidence_map=conf_for_fuse,
            image_bgr=left_bgr,
            theta_low=args.theta_low,
            theta_high=args.theta_high,
            detail_suppression=args.detail_suppression,
            residual_gain=args.residual_gain,
        )
        fused_wcaps, fusion_debug = fuse_edge_aware_residual(
            sgm_disp=sgm_for_fuse,
            da2_aligned=wcaps_aligned,
            confidence_map=conf_for_fuse,
            image_bgr=left_bgr,
            theta_low=args.theta_low,
            theta_high=args.theta_high,
            detail_suppression=args.detail_suppression,
            residual_gain=args.residual_gain,
            return_debug=True,
        )
    else:
        sgm_for_fuse = np.zeros_like(wcaps_aligned, dtype=np.float32)
        conf_for_fuse = np.full_like(wcaps_aligned, 0.5, dtype=np.float32)
        fused_merge = merge_aligned
        fused_wcaps = wcaps_aligned
        zeros = np.zeros_like(fused_wcaps, dtype=np.float32)
        fusion_debug = {
            "detail_score": zeros,
            "alpha_conf": zeros,
            "alpha_eff": zeros,
        }

    fused_refined = None
    fusion_residual = None
    heuristic_vs_net_diff = None
    if args.fusion_backend == "net":
        if (
            load_fusion_net is None
            or build_fusion_inputs is None
            or get_fusion_net_runtime_config is None
            or run_fusion_net_refinement is None
        ):
            raise ImportError(
                "fusion_backend='net' requires core.fusion_net, which is not available in this workspace."
            )
        if not args.fusion_net_weights or not os.path.isfile(args.fusion_net_weights):
            raise FileNotFoundError(
                f"FusionNet weights not found: {args.fusion_net_weights}\n"
                "Pass --fusion-net-weights or keep --fusion-backend heuristic."
            )
        fusion_net_cfg = get_fusion_net_runtime_config(args.fusion_net_weights)
        fusion_net = load_fusion_net(
            args.fusion_net_weights,
            device,
            max_residual_scale=float(fusion_net_cfg["max_residual_scale"]),
        )
        fusion_inputs = build_fusion_inputs(
            image_bgr=left_bgr,
            mono_disp_aligned=wcaps_aligned,
            sgm_disp=sgm_for_fuse,
            confidence_map=conf_for_fuse,
            fused_base=fused_wcaps,
            detail_score=fusion_debug["detail_score"],
        )
        fused_refined, fusion_residual = run_fusion_net_refinement(
            fusion_net,
            fusion_inputs,
            device=device,
            apply_anchor_gate=bool(fusion_net_cfg["apply_anchor_gate"]),
            anchor_gate_strength=float(fusion_net_cfg["anchor_gate_strength"]),
        )
        heuristic_vs_net_diff = np.abs(fused_refined - fused_wcaps).astype(np.float32)

    sensitivity_map = build_decoder_sensitivity_map(
        conf_map=confidence_map,
        image_bgr=left_bgr,
        conf_weight=args.decoder_conf_weight,
        texture_weight=0.0,
        variance_weight=0.0,
    )
    hp_mask = build_stage_high_precision_mask(
        sensitivity_map=sensitivity_map,
        target_hw=(h_orig, w_orig),
        high_precision_ratio=args.decoder_high_precision_ratio,
    ).squeeze().cpu().numpy().astype(np.float32)
    diff_merge_wcaps = np.abs(normalize_map(merge_aligned) - normalize_map(wcaps_aligned)).astype(np.float32)

    disparity_display_maps = [
        sgm_disp_display,
        dense_aligned,
        merge_aligned,
        wcaps_aligned,
        fused_merge,
        fused_wcaps,
    ]
    if fused_refined is not None:
        disparity_display_maps.append(fused_refined)
    display_vmin, display_vmax = compute_disparity_display_range(disparity_display_maps, gt_disp=gt_disp)
    print(f"Display disparity range: [{display_vmin:.2f}, {display_vmax:.2f}] px")

    panels = []
    p1 = left_bgr.copy()
    save_panel(p1, os.path.join(args.out_dir, "01_left_image.png"), "Input Left Image")
    panels.append(("Input", p1))

    p2 = resize_to_match(colorize_disparity_shared(sgm_disp_display, display_vmin, display_vmax), left_bgr)
    save_panel(p2, os.path.join(args.out_dir, "02_sgm_disparity.png"), "SGM disparity (shared scale)")
    panels.append(("SGM", p2))

    p3 = resize_to_match(colorize(confidence_map, cmap="RdYlGn", vmin=0.0, vmax=1.0), left_bgr)
    save_panel(p3, os.path.join(args.out_dir, "03_confidence.png"), "Confidence map")
    panels.append(("Confidence", p3))

    if gt_disp is not None:
        p4 = resize_to_match(colorize_disparity_shared(gt_disp.astype(np.float32), display_vmin, display_vmax), left_bgr)
        save_panel(p4, os.path.join(args.out_dir, "04_gt_disparity.png"), "GT disparity (shared scale)")
        panels.append(("GT disparity", p4))

    p4b = resize_to_match(colorize_disparity_shared(dense_aligned, display_vmin, display_vmax), left_bgr)
    save_panel(p4b, os.path.join(args.out_dir, "05_dense_aligned.png"), "Dense DA2 aligned (shared scale)")
    panels.append(("Dense aligned", p4b))

    p5 = resize_to_match(colorize_disparity_shared(merge_aligned, display_vmin, display_vmax), left_bgr)
    save_panel(p5, os.path.join(args.out_dir, "06_merge_aligned.png"), f"Merge FP32 aligned (shared scale) keep={args.merge_keep_ratio:.3f}")
    panels.append(("Merge aligned", p5))

    p6 = resize_to_match(colorize_disparity_shared(wcaps_aligned, display_vmin, display_vmax), left_bgr)
    save_panel(
        p6,
        os.path.join(args.out_dir, "07_wcaps_aligned.png"),
        f"W-CAPS aligned (shared scale) keep={args.adaptive_keep_ratio:.3f} hp={args.decoder_high_precision_ratio:.2f}",
    )
    panels.append(("W-CAPS aligned", p6))

    p7 = resize_to_match(colorize_disparity_shared(fused_merge, display_vmin, display_vmax), left_bgr)
    save_panel(p7, os.path.join(args.out_dir, "08_fused_sgm_merge.png"), "Fused SGM + Merge (shared scale)")
    panels.append(("Fused merge", p7))

    p8 = resize_to_match(colorize_disparity_shared(fused_wcaps, display_vmin, display_vmax), left_bgr)
    save_panel(p8, os.path.join(args.out_dir, "09_heuristic_fused_wcaps.png"), "Fused SGM + W-CAPS (shared scale)")
    panels.append(("Fused W-CAPS", p8))

    if fused_refined is not None and fusion_residual is not None and heuristic_vs_net_diff is not None:
        p8b = resize_to_match(colorize_disparity_shared(fused_refined, display_vmin, display_vmax), left_bgr)
        save_panel(p8b, os.path.join(args.out_dir, "10_fusion_net_refined.png"), "FusionNet refined (shared scale)")
        panels.append(("FusionNet refined", p8b))

        residual_vis = np.abs(fusion_residual).astype(np.float32)
        p8c = resize_to_match(
            colorize(residual_vis, cmap="hot", vmin=0.0, vmax=float(np.percentile(residual_vis, 99.0)) + 1e-8),
            left_bgr,
        )
        save_panel(p8c, os.path.join(args.out_dir, "11_fusion_net_residual.png"), "FusionNet residual |r|")
        panels.append(("Residual map", p8c))

        p8d = resize_to_match(
            colorize(heuristic_vs_net_diff, cmap="hot", vmin=0.0, vmax=float(np.percentile(heuristic_vs_net_diff, 99.0)) + 1e-8),
            left_bgr,
        )
        save_panel(p8d, os.path.join(args.out_dir, "12_heuristic_vs_net_diff.png"), "|heuristic - net|")
        panels.append(("Heuristic vs net", p8d))

    p9 = resize_to_match(colorize(hp_mask, cmap="viridis", vmin=0.0, vmax=1.0), left_bgr)
    save_panel(p9, os.path.join(args.out_dir, "13_wcaps_hp_mask.png"), "W-CAPS high-precision mask")
    panels.append(("HP mask", p9))

    p10 = resize_to_match(colorize(diff_merge_wcaps, cmap="hot", vmin=0.0, vmax=diff_merge_wcaps.max() + 1e-8), left_bgr)
    save_panel(p10, os.path.join(args.out_dir, "14_diff_merge_vs_wcaps.png"), "|Merge - W-CAPS|")
    panels.append(("|Merge-W-CAPS|", p10))

    p11 = resize_to_match(colorize(fusion_debug["detail_score"], cmap="viridis", vmin=0.0, vmax=1.0), left_bgr)
    save_panel(p11, os.path.join(args.out_dir, "15_detail_score.png"), "edge_aware_residual detail_score")
    panels.append(("detail_score", p11))

    p12 = resize_to_match(colorize(fusion_debug["alpha_conf"], cmap="magma", vmin=0.0, vmax=1.0), left_bgr)
    save_panel(p12, os.path.join(args.out_dir, "16_alpha_conf.png"), "edge_aware_residual alpha_conf")
    panels.append(("alpha_conf", p12))

    p13 = resize_to_match(colorize(fusion_debug["alpha_eff"], cmap="magma", vmin=0.0, vmax=1.0), left_bgr)
    save_panel(p13, os.path.join(args.out_dir, "17_alpha_eff.png"), "edge_aware_residual alpha_eff")
    panels.append(("alpha_eff", p13))

    build_summary_figure(panels, os.path.join(args.out_dir, "00_summary.png"), ncol=4)

    print("Demo complete.")
    print(f"  merge keep ratio    : {args.merge_keep_ratio:.3f}")
    print(f"  adaptive keep ratio : {args.adaptive_keep_ratio:.3f}")
    print(f"  adaptive hp ratio   : {args.decoder_high_precision_ratio:.2f}")
    print(f"  fusion strategy     : {args.fusion_strategy}")
    print(f"  fusion backend      : {args.fusion_backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge + decoder W-CAPS demo for SGM-ViT.")
    parser.add_argument("--left", default=os.path.join(_SCRIPT_DIR, "asserts", "left", "000005_10.png"))
    parser.add_argument("--right", default=os.path.join(_SCRIPT_DIR, "asserts", "right", "000005_10.png"))
    parser.add_argument("--gt-disparity", default=None)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(DA2_MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--merge-layer", dest="merge_layer", type=int, default=0)
    parser.add_argument("--prune-layer", dest="merge_layer", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--merge-keep-ratio", type=float, default=0.814)
    parser.add_argument("--adaptive-keep-ratio", type=float, default=0.814)
    parser.add_argument("--decoder-high-precision-ratio", type=float, default=0.75)
    parser.add_argument("--decoder-low-bits", type=int, default=4)
    parser.add_argument("--decoder-conf-weight", type=float, default=1.0)
    parser.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    parser.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    parser.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--fusion-strategy", default="edge_aware_residual", choices=["edge_aware_residual"])
    parser.add_argument("--theta-low", type=float, default=DEFAULT_EDGE_THETA_LOW)
    parser.add_argument("--theta-high", type=float, default=DEFAULT_EDGE_THETA_HIGH)
    parser.add_argument("--detail-suppression", type=float, default=DEFAULT_EDGE_DETAIL_SUPPRESSION)
    parser.add_argument("--residual-gain", type=float, default=DEFAULT_EDGE_RESIDUAL_GAIN)
    parser.add_argument("--fusion-backend", choices=["heuristic", "net"], default=DEFAULT_FUSION_BACKEND)
    parser.add_argument("--fusion-net-weights", default=DEFAULT_FUSION_NET_WEIGHTS)
    parser.add_argument("--out-dir", default=default_results_dir("demo_merge_adaptive"))
    parser.add_argument("--no-sgm", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
