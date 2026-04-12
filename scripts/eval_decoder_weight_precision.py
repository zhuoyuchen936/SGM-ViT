#!/usr/bin/env python3
"""
scripts/eval_decoder_weight_precision.py
========================================
Evaluate coarse-stage decoder weight-aware adaptive precision.

This script compares the earlier activation proxy against a more realistic
weight-aware dual-path decoder prototype:

- coarse INT4 activation proxy
- coarse INT4 activation CAPS hp=75%
- coarse INT4 weight-aware proxy
- coarse INT4 weight-aware CAPS hp=50% / 75%
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import core._paths  # noqa: F401
from core.fusion import fuse_sgm_da2
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    align_depth_to_sgm,
    load_da2_model,
    run_decoder_caps_merged_da2,
    run_decoder_weight_caps_merged_da2,
    run_token_merged_da2,
)
from scripts.common_config import (
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_ENCODER,
    DEFAULT_FUSION_CONF_THRESHOLD,
    DEFAULT_INPUT_SIZE,
    DEFAULT_KITTI_ROOT,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_PRUNE_LAYER,
    DEFAULT_WEIGHTS,
    default_results_dir,
)
from scripts.eval_kitti import (
    aggregate,
    build_sample_list,
    compute_metrics,
    load_protocol_confidence_maps,
    read_kitti_gt,
    read_pfm,
)


CONFIGS = [
    {"key": "dense_fp32", "label": "Dense DA2 + align", "kind": "dense"},
    {"key": "merge_fp32", "label": "Merge FP32", "kind": "merge"},
    {
        "key": "act_int4_uniform",
        "label": "Decoder coarse INT4 activation proxy",
        "kind": "act",
        "hp_ratio": 0.0,
    },
    {
        "key": "act_int4_caps75",
        "label": "Decoder coarse INT4 activation CAPS hp=75%",
        "kind": "act",
        "hp_ratio": 0.75,
    },
    {
        "key": "weight_int4_uniform",
        "label": "Decoder coarse INT4 weight-aware proxy",
        "kind": "weight",
        "hp_ratio": 0.0,
    },
    {
        "key": "weight_int4_caps50",
        "label": "Decoder coarse INT4 weight-aware CAPS hp=50%",
        "kind": "weight",
        "hp_ratio": 0.50,
    },
    {
        "key": "weight_int4_caps75",
        "label": "Decoder coarse INT4 weight-aware CAPS hp=75%",
        "kind": "weight",
        "hp_ratio": 0.75,
    },
]


def precision_proxy(hp_ratio: float, high_bits: int = 8, low_bits: int = 4) -> float:
    active_fraction = 0.5
    low_fraction = active_fraction * (1.0 - hp_ratio)
    return 1.0 - low_fraction * (1.0 - low_bits / float(high_bits))


def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "eval_decoder_weight_precision.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info("=" * 96)
    logging.info("SGM-ViT Decoder Weight-Aware Precision Evaluation")
    logging.info("=" * 96)
    logging.info(f"  KITTI root          : {args.kitti_root}")
    logging.info(f"  Encoder             : {args.encoder}")
    logging.info(f"  Keep ratio          : {args.keep_ratio}")
    logging.info(f"  Merge layer         : {args.merge_layer}")
    logging.info(f"  Stage policy        : coarse_only")
    logging.info(f"  Low precision bits  : {args.low_precision_bits}")
    logging.info(f"  Max samples         : {args.max_samples}")
    logging.info(f"  Output              : {args.out_dir}")
    logging.info("=" * 96)

    model = load_da2_model(args.encoder, args.weights, device)
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error("No samples found; check --kitti-root.")
        return
    samples = samples[:args.max_samples]
    logging.info(f"Loaded {len(samples)} samples.\n")

    records = {cfg["key"]: {"all": [], "kitti15": [], "kitti12": []} for cfg in CONFIGS}
    fused_records = {cfg["key"]: {"all": [], "kitti15": [], "kitti12": []} for cfg in CONFIGS}
    csv_rows: list[dict[str, object]] = []

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating", unit="img")):
        left_bgr = cv2.imread(sample["left"])
        if left_bgr is None:
            logging.warning(f"Cannot read image: {sample['left']}")
            continue

        gt_disp, valid_gt = read_kitti_gt(sample["gt_disp"])
        sgm_disp = read_pfm(sample["sgm_pfm"])
        conf_align, conf_route = load_protocol_confidence_maps(
            sample,
            disparity_range=args.disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )

        sgm_for_eval = sgm_disp
        if sgm_for_eval.shape != gt_disp.shape:
            sgm_for_eval = cv2.resize(
                sgm_for_eval,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        conf_fuse = conf_route
        if conf_fuse.shape != gt_disp.shape:
            conf_fuse = cv2.resize(
                conf_fuse,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        for cfg in CONFIGS:
            if cfg["kind"] == "dense":
                depth_pred = model.infer_image(left_bgr, input_size=args.input_size)
            elif cfg["kind"] == "merge":
                depth_pred = run_token_merged_da2(
                    model=model,
                    image_bgr=left_bgr,
                    confidence_map=conf_route,
                    keep_ratio=args.keep_ratio,
                    input_size=args.input_size,
                    merge_layer=args.merge_layer,
                )
            elif cfg["kind"] == "act":
                depth_pred = run_decoder_caps_merged_da2(
                    model=model,
                    image_bgr=left_bgr,
                    confidence_map=conf_route,
                    keep_ratio=args.keep_ratio,
                    input_size=args.input_size,
                    merge_layer=args.merge_layer,
                    decoder_high_precision_ratio=float(cfg["hp_ratio"]),
                    high_precision_bits=8,
                    low_precision_bits=args.low_precision_bits,
                    decoder_conf_weight=args.decoder_conf_weight,
                    decoder_texture_weight=0.0,
                    decoder_variance_weight=0.0,
                    decoder_stage_policy="coarse_only",
                )
            else:
                depth_pred = run_decoder_weight_caps_merged_da2(
                    model=model,
                    image_bgr=left_bgr,
                    confidence_map=conf_route,
                    keep_ratio=args.keep_ratio,
                    input_size=args.input_size,
                    merge_layer=args.merge_layer,
                    decoder_high_precision_ratio=float(cfg["hp_ratio"]),
                    low_precision_bits=args.low_precision_bits,
                    decoder_conf_weight=args.decoder_conf_weight,
                    decoder_texture_weight=0.0,
                    decoder_variance_weight=0.0,
                    decoder_stage_policy="coarse_only",
                )

            disp_pred, _, _ = align_depth_to_sgm(
                depth_mono=depth_pred,
                disparity_raw=sgm_disp,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )
            if disp_pred.shape != gt_disp.shape:
                disp_pred = cv2.resize(
                    disp_pred,
                    (gt_disp.shape[1], gt_disp.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            dense_metrics = compute_metrics(disp_pred, gt_disp, valid_gt)
            disp_fused = fuse_sgm_da2(
                sgm_disp=sgm_for_eval,
                da2_aligned=disp_pred,
                confidence_map=conf_fuse,
                conf_threshold=args.fusion_conf_threshold,
            )
            fused_metrics = compute_metrics(disp_fused, gt_disp, valid_gt)

            for split_key in ("all", sample["split"]):
                records[cfg["key"]][split_key].append(dense_metrics)
                fused_records[cfg["key"]][split_key].append(fused_metrics)

            csv_rows.append(
                {
                    "idx": idx,
                    "split": sample["split"],
                    "image": os.path.basename(sample["left"]),
                    "config": cfg["key"],
                    "hp_ratio": f"{float(cfg.get('hp_ratio', 1.0)):.2f}",
                    "precision_proxy": f"{precision_proxy(float(cfg.get('hp_ratio', 1.0))):.4f}",
                    "dense_epe": f"{dense_metrics['epe']:.4f}",
                    "dense_d1": f"{dense_metrics['d1']:.4f}",
                    "fused_epe": f"{fused_metrics['epe']:.4f}",
                    "fused_d1": f"{fused_metrics['d1']:.4f}",
                }
            )

    lines = [
        "=" * 96,
        "SGM-ViT Decoder Weight-Aware Precision Results",
        "=" * 96,
        f"Fixed keep ratio: {args.keep_ratio:.3f}",
        "Fixed stage policy: coarse_only",
        "Primary metric: dense-only EPE/D1. Fused metrics are secondary.",
        "",
        "  Config                                      PrecProxy  Dense EPE  Dense D1   Fused EPE  Fused D1",
        "  -----------------------------------------------------------------------------------------------------",
    ]

    for cfg in CONFIGS:
        dense_agg = aggregate(records[cfg["key"]]["all"])
        fused_agg = aggregate(fused_records[cfg["key"]]["all"])
        proxy = precision_proxy(float(cfg.get("hp_ratio", 1.0)))
        lines.append(
            f"  {cfg['label']:<42s}"
            f"  {proxy:>10.4f}"
            f"  {dense_agg['epe']:>10.4f}"
            f"  {dense_agg['d1']:>10.4f}"
            f"  {fused_agg['epe']:>10.4f}"
            f"  {fused_agg['d1']:>10.4f}"
        )

    lines.extend(
        [
            "",
            "Readout:",
            "  - Compare activation proxy vs weight-aware at the same coarse INT4 budget.",
            "  - If weight-aware CAPS remains competitive, coarse decoder quantization is not just an activation artifact.",
            "  - This is still a prototype; Python dual-path wall time is not a hardware speedup number.",
            "",
        ]
    )

    summary_path = os.path.join(args.out_dir, "decoder_weight_precision_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logging.info("\n" + "\n".join(lines))
    logging.info(f"Summary: {summary_path}")

    csv_path = os.path.join(args.out_dir, "decoder_weight_precision_per_image.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-image CSV: {csv_path}")

    logging.info("=" * 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoder weight-aware coarse-stage precision evaluation for SGM-ViT.",
    )
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(DA2_MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--keep-ratio", type=float, default=0.814)
    parser.add_argument("--merge-layer", type=int, default=DEFAULT_PRUNE_LAYER)
    parser.add_argument("--low-precision-bits", type=int, default=4)
    parser.add_argument("--decoder-conf-weight", type=float, default=1.0)
    parser.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    parser.add_argument("--fusion-conf-threshold", type=float, default=DEFAULT_FUSION_CONF_THRESHOLD)
    parser.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    parser.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    parser.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        default=default_results_dir("eval_decoder_weight_precision"),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
