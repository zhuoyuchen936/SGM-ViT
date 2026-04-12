#!/usr/bin/env python3
"""
scripts/eval_token_merge.py
===========================
20-image pilot comparing dense DA2, pruning baselines, and token merge.

Primary comparison is dense-only accuracy after alignment to SGM disparity
space. Fused SGM+DA2 metrics are reported as a secondary endpoint.
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import core._paths  # noqa: F401
from core.eval_utils import compute_attn_reduction
from core.fusion import fuse_sgm_da2
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    TOKEN_GRID_SIZE,
    align_depth_to_sgm,
    load_da2_model,
    run_masked_sparse_da2,
    run_token_merged_da2,
)
from core.pruning_strategies import topk_confidence_mask
from core.token_merge import build_token_merge_groups
from core.token_router import SGMConfidenceTokenRouter
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
    DEFAULT_PRUNE_THRESHOLD,
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


KEEP_RATIOS = [0.60, 0.70, 0.80, 0.814, 0.90]


def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "eval_token_merge.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info("=" * 76)
    logging.info("SGM-ViT Token-Merge Pilot")
    logging.info("=" * 76)
    logging.info(f"  KITTI root        : {args.kitti_root}")
    logging.info(f"  Encoder           : {args.encoder}")
    logging.info(f"  Input size        : {args.input_size}")
    logging.info(f"  Merge layer       : {args.merge_layer}")
    logging.info(f"  Keep ratios       : {KEEP_RATIOS}")
    logging.info(f"  Anchor theta      : {args.threshold}")
    logging.info(f"  Samples           : {args.max_samples}")
    logging.info(f"  Re-assembly       : {'on' if not args.no_reassembly else 'off'}")
    logging.info(f"  Device            : {device}")
    logging.info(f"  Output            : {args.out_dir}")
    logging.info("=" * 76)

    model = load_da2_model(args.encoder, args.weights, device)
    n_blocks = len(list(model.pretrained.blocks))
    total_tokens = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE

    router = SGMConfidenceTokenRouter(
        token_grid_size=TOKEN_GRID_SIZE,
        confidence_threshold=args.threshold,
        learnable_threshold=False,
    )

    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error("No samples found; check --kitti-root.")
        return
    samples = samples[:args.max_samples]
    logging.info(f"Loaded {len(samples)} pilot samples.\n")

    dense_records = {"all": [], "kitti15": [], "kitti12": []}
    fused_dense_records = {"all": [], "kitti15": [], "kitti12": []}

    topk_records: dict[float, dict[str, list]] = {}
    fused_topk_records: dict[float, dict[str, list]] = {}
    topk_keep_counts: dict[float, list[int]] = {}

    merge_records: dict[float, dict[str, list]] = {}
    fused_merge_records: dict[float, dict[str, list]] = {}
    merge_rep_counts: dict[float, list[int]] = {}

    anchor_records = {"all": [], "kitti15": [], "kitti12": []}
    fused_anchor_records = {"all": [], "kitti15": [], "kitti12": []}
    anchor_keep_counts: list[int] = []

    csv_rows: list[dict[str, object]] = []

    def _init_record_map(dst: dict[float, dict[str, list]], keep_ratio: float) -> None:
        if keep_ratio not in dst:
            dst[keep_ratio] = {"all": [], "kitti15": [], "kitti12": []}

    for keep_ratio in KEEP_RATIOS:
        _init_record_map(topk_records, keep_ratio)
        _init_record_map(fused_topk_records, keep_ratio)
        _init_record_map(merge_records, keep_ratio)
        _init_record_map(fused_merge_records, keep_ratio)
        topk_keep_counts[keep_ratio] = []
        merge_rep_counts[keep_ratio] = []

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

        depth_dense = model.infer_image(left_bgr, input_size=args.input_size)
        disp_dense, _, _ = align_depth_to_sgm(
            depth_mono=depth_dense,
            disparity_raw=sgm_disp,
            confidence_map=conf_align,
            conf_threshold=args.align_conf_threshold,
        )
        if disp_dense.shape != gt_disp.shape:
            disp_dense = cv2.resize(
                disp_dense,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        dense_metrics = compute_metrics(disp_dense, gt_disp, valid_gt)
        fused_dense = fuse_sgm_da2(
            sgm_disp=sgm_for_eval,
            da2_aligned=disp_dense,
            confidence_map=conf_fuse,
            conf_threshold=args.fusion_conf_threshold,
        )
        fused_dense_metrics = compute_metrics(fused_dense, gt_disp, valid_gt)

        for key in ("all", sample["split"]):
            dense_records[key].append(dense_metrics)
            fused_dense_records[key].append(fused_dense_metrics)

        csv_rows.append({
            "idx": idx,
            "split": sample["split"],
            "image": os.path.basename(sample["left"]),
            "method": "dense",
            "variant": "dense_da2_align",
            "keep_ratio": "1.000",
            "rep_count": total_tokens,
            "attn_reduction": "0.0000",
            "dense_epe": f"{dense_metrics['epe']:.4f}",
            "dense_d1": f"{dense_metrics['d1']:.4f}",
            "fused_epe": f"{fused_dense_metrics['epe']:.4f}",
            "fused_d1": f"{fused_dense_metrics['d1']:.4f}",
        })

        conf_tensor = torch.from_numpy(conf_route).unsqueeze(0).unsqueeze(0)
        dummy_tokens = torch.zeros(1, total_tokens, 1)
        with torch.no_grad():
            routing = router(conf_tensor, dummy_tokens)
        anchor_prune_mask = torch.zeros(total_tokens, dtype=torch.bool)
        anchor_prune_mask[routing["prune_idx"][0]] = True
        anchor_keep = int((~anchor_prune_mask).sum().item())

        depth_anchor = run_masked_sparse_da2(
            model=model,
            image_bgr=left_bgr,
            prune_mask=anchor_prune_mask,
            input_size=args.input_size,
            prune_layer=args.merge_layer,
            do_reassembly=not args.no_reassembly,
        )
        disp_anchor, _, _ = align_depth_to_sgm(
            depth_mono=depth_anchor,
            disparity_raw=sgm_disp,
            confidence_map=conf_align,
            conf_threshold=args.align_conf_threshold,
        )
        if disp_anchor.shape != gt_disp.shape:
            disp_anchor = cv2.resize(
                disp_anchor,
                (gt_disp.shape[1], gt_disp.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        anchor_metrics = compute_metrics(disp_anchor, gt_disp, valid_gt)
        fused_anchor = fuse_sgm_da2(
            sgm_disp=sgm_for_eval,
            da2_aligned=disp_anchor,
            confidence_map=conf_fuse,
            conf_threshold=args.fusion_conf_threshold,
        )
        fused_anchor_metrics = compute_metrics(fused_anchor, gt_disp, valid_gt)
        anchor_attn_reduction = compute_attn_reduction(
            args.merge_layer,
            anchor_keep,
            total_tokens,
            n_blocks,
        )

        for key in ("all", sample["split"]):
            anchor_records[key].append(anchor_metrics)
            fused_anchor_records[key].append(fused_anchor_metrics)
        anchor_keep_counts.append(anchor_keep)

        csv_rows.append({
            "idx": idx,
            "split": sample["split"],
            "image": os.path.basename(sample["left"]),
            "method": "prune",
            "variant": f"threshold_theta_{args.threshold:.2f}",
            "keep_ratio": f"{anchor_keep / total_tokens:.3f}",
            "rep_count": anchor_keep,
            "attn_reduction": f"{anchor_attn_reduction:.4f}",
            "dense_epe": f"{anchor_metrics['epe']:.4f}",
            "dense_d1": f"{anchor_metrics['d1']:.4f}",
            "fused_epe": f"{fused_anchor_metrics['epe']:.4f}",
            "fused_d1": f"{fused_anchor_metrics['d1']:.4f}",
        })

        for keep_ratio in KEEP_RATIOS:
            prune_mask = topk_confidence_mask(conf_route, TOKEN_GRID_SIZE, keep_ratio)
            keep_count = int((~prune_mask).sum().item())
            depth_topk = run_masked_sparse_da2(
                model=model,
                image_bgr=left_bgr,
                prune_mask=prune_mask,
                input_size=args.input_size,
                prune_layer=args.merge_layer,
                do_reassembly=not args.no_reassembly,
            )
            disp_topk, _, _ = align_depth_to_sgm(
                depth_mono=depth_topk,
                disparity_raw=sgm_disp,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )
            if disp_topk.shape != gt_disp.shape:
                disp_topk = cv2.resize(
                    disp_topk,
                    (gt_disp.shape[1], gt_disp.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            topk_metrics = compute_metrics(disp_topk, gt_disp, valid_gt)
            fused_topk = fuse_sgm_da2(
                sgm_disp=sgm_for_eval,
                da2_aligned=disp_topk,
                confidence_map=conf_fuse,
                conf_threshold=args.fusion_conf_threshold,
            )
            fused_topk_metrics = compute_metrics(fused_topk, gt_disp, valid_gt)
            topk_attn_reduction = compute_attn_reduction(
                args.merge_layer,
                keep_count,
                total_tokens,
                n_blocks,
            )

            for key in ("all", sample["split"]):
                topk_records[keep_ratio][key].append(topk_metrics)
                fused_topk_records[keep_ratio][key].append(fused_topk_metrics)
            topk_keep_counts[keep_ratio].append(keep_count)

            csv_rows.append({
                "idx": idx,
                "split": sample["split"],
                "image": os.path.basename(sample["left"]),
                "method": "prune",
                "variant": f"topk_keep_{keep_ratio:.3f}",
                "keep_ratio": f"{keep_count / total_tokens:.3f}",
                "rep_count": keep_count,
                "attn_reduction": f"{topk_attn_reduction:.4f}",
                "dense_epe": f"{topk_metrics['epe']:.4f}",
                "dense_d1": f"{topk_metrics['d1']:.4f}",
                "fused_epe": f"{fused_topk_metrics['epe']:.4f}",
                "fused_d1": f"{fused_topk_metrics['d1']:.4f}",
            })

            merge_plan = build_token_merge_groups(
                conf_map=conf_route,
                token_grid_size=TOKEN_GRID_SIZE,
                keep_ratio=keep_ratio,
            )
            rep_count = int(merge_plan["rep_count"])
            depth_merge = run_token_merged_da2(
                model=model,
                image_bgr=left_bgr,
                confidence_map=conf_route,
                keep_ratio=keep_ratio,
                input_size=args.input_size,
                merge_layer=args.merge_layer,
            )
            disp_merge, _, _ = align_depth_to_sgm(
                depth_mono=depth_merge,
                disparity_raw=sgm_disp,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )
            if disp_merge.shape != gt_disp.shape:
                disp_merge = cv2.resize(
                    disp_merge,
                    (gt_disp.shape[1], gt_disp.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            merge_metrics = compute_metrics(disp_merge, gt_disp, valid_gt)
            fused_merge = fuse_sgm_da2(
                sgm_disp=sgm_for_eval,
                da2_aligned=disp_merge,
                confidence_map=conf_fuse,
                conf_threshold=args.fusion_conf_threshold,
            )
            fused_merge_metrics = compute_metrics(fused_merge, gt_disp, valid_gt)
            merge_attn_reduction = compute_attn_reduction(
                args.merge_layer,
                rep_count,
                total_tokens,
                n_blocks,
            )

            for key in ("all", sample["split"]):
                merge_records[keep_ratio][key].append(merge_metrics)
                fused_merge_records[keep_ratio][key].append(fused_merge_metrics)
            merge_rep_counts[keep_ratio].append(rep_count)

            csv_rows.append({
                "idx": idx,
                "split": sample["split"],
                "image": os.path.basename(sample["left"]),
                "method": "merge",
                "variant": f"confidence_guided_keep_{keep_ratio:.3f}",
                "keep_ratio": f"{rep_count / total_tokens:.3f}",
                "rep_count": rep_count,
                "attn_reduction": f"{merge_attn_reduction:.4f}",
                "dense_epe": f"{merge_metrics['epe']:.4f}",
                "dense_d1": f"{merge_metrics['d1']:.4f}",
                "fused_epe": f"{fused_merge_metrics['epe']:.4f}",
                "fused_d1": f"{fused_merge_metrics['d1']:.4f}",
            })

    def _fmt_row(
        label: str,
        keep_count: float,
        dense_agg: dict,
        fused_agg: dict,
    ) -> str:
        keep_ratio = keep_count / total_tokens
        attn_reduction = compute_attn_reduction(
            args.merge_layer,
            keep_count,
            total_tokens,
            n_blocks,
        ) * 100.0
        return (
            f"  {label:<34s}"
            f"  {keep_ratio:>6.3f}"
            f"  {int(round(keep_count)):>7d}"
            f"  {attn_reduction:>8.2f}%"
            f"  {dense_agg['epe']:>10.4f}"
            f"  {dense_agg['d1']:>10.4f}"
            f"  {fused_agg['epe']:>10.4f}"
            f"  {fused_agg['d1']:>10.4f}"
        )

    lines = [
        "=" * 76,
        "SGM-ViT Token-Merge Pilot Results",
        "=" * 76,
        "Primary metric: dense-only EPE/D1 after alignment. Fused metrics are secondary.",
        "",
        "  Method                              KeepR   RepCnt   AttnRed  Dense EPE  Dense D1   Fused EPE  Fused D1",
        "  ------------------------------------------------------------------------------------------------------------",
    ]

    dense_agg = aggregate(dense_records["all"])
    fused_dense_agg = aggregate(fused_dense_records["all"])
    lines.append(
        "  "
        f"{'Dense DA2 + align':<34s}"
        f"  {1.000:>6.3f}"
        f"  {total_tokens:>7d}"
        f"  {0.0:>8.2f}%"
        f"  {dense_agg['epe']:>10.4f}"
        f"  {dense_agg['d1']:>10.4f}"
        f"  {fused_dense_agg['epe']:>10.4f}"
        f"  {fused_dense_agg['d1']:>10.4f}"
    )

    anchor_dense_agg = aggregate(anchor_records["all"])
    fused_anchor_agg = aggregate(fused_anchor_records["all"])
    lines.append(
        _fmt_row(
            f"Prune anchor (theta={args.threshold:.2f})",
            float(np.mean(anchor_keep_counts)) if anchor_keep_counts else total_tokens,
            anchor_dense_agg,
            fused_anchor_agg,
        )
    )

    for keep_ratio in KEEP_RATIOS:
        lines.append(
            _fmt_row(
                f"Prune top-k (keep={keep_ratio:.3f})",
                float(np.mean(topk_keep_counts[keep_ratio])),
                aggregate(topk_records[keep_ratio]["all"]),
                aggregate(fused_topk_records[keep_ratio]["all"]),
            )
        )
        lines.append(
            _fmt_row(
                f"Merge conf-guided (keep={keep_ratio:.3f})",
                float(np.mean(merge_rep_counts[keep_ratio])),
                aggregate(merge_records[keep_ratio]["all"]),
                aggregate(fused_merge_records[keep_ratio]["all"]),
            )
        )

    lines.extend([
        "",
        "Direct paper note:",
        "  Token pruning harms dense monocular depth quality; confidence-guided fusion only partially",
        "  recovers end-task performance by replacing many damaged regions with SGM output.",
        "",
        "Token-merge table template for the paper:",
        "",
        "| Method | Keep Ratio | Rep Count | Attn Reduction (%) | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |",
        "|--------|------------|-----------|--------------------|-----------|--------------|-----------|--------------|",
        "| Dense DA2 + align | 1.000 | 1369 | 0.0 | fill from summary | fill from summary | fill from summary | fill from summary |",
        "| Prune top-k | 0.814 | 1114 | fill from summary | fill from summary | fill from summary | fill from summary | fill from summary |",
        "| Prune anchor theta=0.65 | variable | variable | fill from summary | fill from summary | fill from summary | fill from summary | fill from summary |",
        "| Merge conf-guided | 0.814 | 1114 | fill from summary | fill from summary | fill from summary | fill from summary | fill from summary |",
        "",
    ])

    summary_path = os.path.join(args.out_dir, "token_merge_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    logging.info("\n" + "\n".join(lines))
    logging.info(f"Summary: {summary_path}")

    csv_path = os.path.join(args.out_dir, "token_merge_per_image.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-image CSV: {csv_path}")

    logging.info("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot comparison of pruning and token merge on KITTI.",
    )
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(DA2_MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_PRUNE_THRESHOLD)
    parser.add_argument("--merge-layer", type=int, default=DEFAULT_PRUNE_LAYER)
    parser.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    parser.add_argument("--fusion-conf-threshold", type=float, default=DEFAULT_FUSION_CONF_THRESHOLD)
    parser.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    parser.add_argument("--disparity-range", type=int, default=DEFAULT_DISPARITY_RANGE)
    parser.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--no-reassembly", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=default_results_dir("eval_token_merge"),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
