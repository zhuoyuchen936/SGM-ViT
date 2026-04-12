#!/usr/bin/env python3
"""Build offline cache files for FusionNet-v1 training and evaluation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from scripts.common_config import (  # noqa: E402
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_EDGE_DETAIL_SUPPRESSION,
    DEFAULT_EDGE_RESIDUAL_GAIN,
    DEFAULT_EDGE_THETA_HIGH,
    DEFAULT_EDGE_THETA_LOW,
    DEFAULT_ENCODER,
    DEFAULT_ETH3D_ROOT,
    DEFAULT_INPUT_SIZE,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MIDDLEBURY_ROOT,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_SCENEFLOW_ROOT,
    DEFAULT_WEIGHTS,
)


DEFAULT_CACHE_ROOT = os.path.join(_PROJECT_DIR, "artifacts", "fusion_cache")
DEFAULT_MERGE_KEEP_RATIO = 0.814
DEFAULT_ADAPTIVE_KEEP_RATIO = 0.814
DEFAULT_DECODER_HIGH_PRECISION_RATIO = 0.75
DEFAULT_DECODER_LOW_BITS = 4
DEFAULT_DECODER_STAGE_POLICY = "coarse_only"
DATASET_ORDER = ("sceneflow", "kitti", "eth3d", "middlebury")


def _safe_name(text: str) -> str:
    return text.replace("\\", "__").replace("/", "__").replace(":", "_").replace(" ", "_")


def split_name(dataset: str, index: int) -> str:
    if dataset == "sceneflow":
        if index % 10 == 0:
            return "val"
        if index % 10 == 1:
            return "test"
        return "train"
    if dataset == "kitti":
        return "val" if index % 10 == 0 else "train"
    return "eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline FusionNet cache from the current merge+W-CAPS pipeline.")
    parser.add_argument("--dataset", default="all", choices=["all", "kitti", "eth3d", "sceneflow", "middlebury"])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--merge-layer", type=int, default=0)
    parser.add_argument("--merge-keep-ratio", type=float, default=DEFAULT_MERGE_KEEP_RATIO)
    parser.add_argument("--adaptive-keep-ratio", type=float, default=DEFAULT_ADAPTIVE_KEEP_RATIO)
    parser.add_argument("--decoder-high-precision-ratio", type=float, default=DEFAULT_DECODER_HIGH_PRECISION_RATIO)
    parser.add_argument("--decoder-low-bits", type=int, default=DEFAULT_DECODER_LOW_BITS)
    parser.add_argument("--decoder-stage-policy", default=DEFAULT_DECODER_STAGE_POLICY, choices=["all", "coarse_only", "fine_only"])
    parser.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    parser.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    parser.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    parser.add_argument("--theta-low", type=float, default=DEFAULT_EDGE_THETA_LOW)
    parser.add_argument("--theta-high", type=float, default=DEFAULT_EDGE_THETA_HIGH)
    parser.add_argument("--detail-suppression", type=float, default=DEFAULT_EDGE_DETAIL_SUPPRESSION)
    parser.add_argument("--residual-gain", type=float, default=DEFAULT_EDGE_RESIDUAL_GAIN)
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    parser.add_argument("--sceneflow-root", default=DEFAULT_SCENEFLOW_ROOT)
    parser.add_argument("--middlebury-root", default=DEFAULT_MIDDLEBURY_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from core.fusion import fuse_edge_aware_residual
    from core.fusion_net import compute_disp_scale
    from core.pipeline import align_depth_to_sgm, load_da2_model, run_decoder_weight_caps_merged_da2
    from scripts.eval_merge_adaptive import (
        DATASET_DISPARITY_RANGE,
        PRIMARY_PROTOCOL,
        build_sample_list_eth3d,
        build_sample_list_kitti,
        build_sample_list_middlebury,
        build_sample_list_sceneflow,
        load_gt_and_protocol_masks,
        load_protocol_confidence_maps,
        read_pfm,
        resize_map,
    )

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_da2_model(args.encoder, args.weights, device)

    def build_samples(dataset: str) -> list[dict]:
        if dataset == "kitti":
            samples = build_sample_list_kitti(args.kitti_root)
        elif dataset == "eth3d":
            samples = build_sample_list_eth3d(args.eth3d_root)
        elif dataset == "sceneflow":
            samples = build_sample_list_sceneflow(args.sceneflow_root)
        elif dataset == "middlebury":
            samples = build_sample_list_middlebury(args.middlebury_root)
        else:
            raise ValueError(dataset)
        return samples[:args.max_samples] if args.max_samples > 0 else samples

    datasets = DATASET_ORDER if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        samples = build_samples(dataset)
        disparity_range = DATASET_DISPARITY_RANGE[dataset]
        print(f"[{dataset}] caching {len(samples)} samples -> {os.path.join(args.out_root, dataset)}")
        for index, sample in enumerate(tqdm(samples, desc=f"cache:{dataset}", unit="img")):
            split = split_name(dataset, index)
            sample_name = _safe_name(sample["name"])
            out_dir = os.path.join(args.out_root, dataset, split)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{sample_name}.npz")
            if os.path.isfile(out_path) and not args.overwrite:
                continue

            left_bgr = cv2.imread(sample["left"])
            if left_bgr is None:
                raise FileNotFoundError(f"Cannot read left image: {sample['left']}")

            gt_disp, protocol_masks = load_gt_and_protocol_masks(dataset, sample)
            valid_mask = protocol_masks[PRIMARY_PROTOCOL[dataset]].astype(bool)

            sgm_disp_raw = read_pfm(sample["sgm_pfm"]).astype(np.float32)
            conf_align, conf_route = load_protocol_confidence_maps(
                sample,
                disparity_range=disparity_range,
                pkrn_min_dist=args.pkrn_min_dist,
                smooth_sigma=args.conf_sigma,
            )

            mono_depth = run_decoder_weight_caps_merged_da2(
                model=model,
                image_bgr=left_bgr,
                confidence_map=conf_route,
                keep_ratio=args.adaptive_keep_ratio,
                input_size=args.input_size,
                merge_layer=args.merge_layer,
                decoder_high_precision_ratio=args.decoder_high_precision_ratio,
                low_precision_bits=args.decoder_low_bits,
                decoder_conf_weight=1.0,
                decoder_texture_weight=0.0,
                decoder_variance_weight=0.0,
                decoder_stage_policy=args.decoder_stage_policy,
            )
            mono_disp_aligned, align_scale, align_shift = align_depth_to_sgm(
                depth_mono=mono_depth,
                disparity_raw=sgm_disp_raw,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )

            target_hw = mono_disp_aligned.shape[:2]
            sgm_disp = resize_map(sgm_disp_raw, target_hw, cv2.INTER_NEAREST)
            confidence_map = resize_map(conf_route, target_hw, cv2.INTER_LINEAR)
            gt_disp = resize_map(gt_disp, target_hw, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = resize_map(valid_mask.astype(np.float32), target_hw, cv2.INTER_NEAREST) > 0.5

            fused_base, debug = fuse_edge_aware_residual(
                sgm_disp=sgm_disp,
                da2_aligned=mono_disp_aligned,
                confidence_map=confidence_map,
                image_bgr=left_bgr,
                theta_low=args.theta_low,
                theta_high=args.theta_high,
                detail_suppression=args.detail_suppression,
                residual_gain=args.residual_gain,
                return_debug=True,
            )
            disp_scale = compute_disp_scale(fused_base, valid_mask)

            np.savez_compressed(
                out_path,
                rgb=cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8),
                sgm_disp=sgm_disp.astype(np.float32),
                confidence_map=np.clip(confidence_map, 0.0, 1.0).astype(np.float32),
                mono_disp_aligned=mono_disp_aligned.astype(np.float32),
                fused_base=fused_base.astype(np.float32),
                detail_score=debug["detail_score"].astype(np.float32),
                gt_disp=gt_disp.astype(np.float32),
                valid_mask=valid_mask.astype(np.uint8),
                dataset_name=np.array(dataset),
                sample_name=np.array(sample["name"]),
                disp_scale=np.array(float(disp_scale), dtype=np.float32),
                align_scale=np.array(float(align_scale), dtype=np.float32),
                align_shift=np.array(float(align_shift), dtype=np.float32),
            )

        print(f"[{dataset}] done")


if __name__ == "__main__":
    main()
