#!/usr/bin/env python3
"""
Unified four-dataset evaluation for the merge + decoder W-CAPS research line.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import core._paths  # noqa: F401
from core.fusion import fuse_dispatch
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    align_depth_to_sgm,
    load_da2_model,
    run_decoder_weight_caps_merged_da2,
    run_token_merged_da2,
)
from core.sgm_wrapper import run_sgm_with_confidence
from core.token_merge import build_token_merge_groups
from scripts.common_config import (
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_EDGE_DETAIL_SUPPRESSION,
    DEFAULT_EDGE_RESIDUAL_GAIN,
    DEFAULT_EDGE_THETA_HIGH,
    DEFAULT_EDGE_THETA_LOW,
    DEFAULT_ENCODER,
    DEFAULT_ETH3D_ROOT,
    DEFAULT_FUSION_STRATEGY,
    DEFAULT_INPUT_SIZE,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIDDLEBURY_ROOT,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_PRUNE_LAYER,
    DEFAULT_SCENEFLOW_ROOT,
    DEFAULT_WEIGHTS,
    default_results_dir,
)


DATASET_ORDER = ["kitti", "eth3d", "middlebury", "sceneflow"]
PRIMARY_PROTOCOL = {
    "kitti": "main",
    "eth3d": "main",
    "sceneflow": "main",
    "middlebury": "all_valid",
}
DATASET_METRIC_KEYS = {
    "kitti": ["epe", "d1"],
    "eth3d": ["epe", "bad1", "bad2", "d1", "delta1"],
    "sceneflow": ["epe", "bad1", "bad2", "bad3", "delta1"],
    "middlebury": ["epe", "bad2", "delta1"],
}
DATASET_DISPARITY_RANGE = {
    "kitti": DEFAULT_DISPARITY_RANGE,
    "eth3d": 128,
    "sceneflow": 256,
    "middlebury": 192,
}


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip()
        assert header in (b"PF", b"Pf"), f"Not a PFM file: {path}"
        w, h = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f").reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


def read_kitti_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    disp = raw.astype(np.float32) / 256.0
    valid = disp > 0.0
    return disp, valid


def normalize_middlebury_root(root: str) -> str:
    root = os.path.expanduser(root)
    if os.path.isdir(os.path.join(root, "MiddEval3", "trainingQ")):
        return os.path.join(root, "MiddEval3", "trainingQ")
    if os.path.isdir(os.path.join(root, "trainingQ")):
        return os.path.join(root, "trainingQ")
    return root


def resize_map(arr: np.ndarray, shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = shape_hw
    if arr.shape[:2] == (h, w):
        return arr.astype(np.float32)
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=interpolation)


def build_sample_list_kitti(kitti_root: str) -> list[dict]:
    def sg(pat: str) -> list[str]:
        return sorted(glob(pat))

    root_15 = kitti_root
    root_12 = os.path.join(kitti_root, "kitti2012")
    samples: list[dict] = []

    lefts = sg(os.path.join(root_15, "training", "image_2", "*_10.png"))
    rights = sg(os.path.join(root_15, "training", "image_3", "*_10.png"))
    gts = sg(os.path.join(root_15, "training", "disp_occ_0", "*_10.png"))
    sgms = sg(os.path.join(root_15, "training", "sgm_hole", "*_10.pfm"))
    mis = sg(os.path.join(root_15, "training", "sgm_hole", "*_10_mismatches.npy"))
    occ = sg(os.path.join(root_15, "training", "sgm_hole", "*_10_occlusion.npy"))
    for left, right, gt, sgm, mismatch, occlusion in zip(lefts, rights, gts, sgms, mis, occ):
        samples.append(
            {
                "left": left,
                "right": right,
                "gt_path": gt,
                "sgm_pfm": sgm,
                "mismatch": mismatch,
                "occlusion": occlusion,
                "pkrn_cache": sgm.replace(".pfm", "_pkrn.npy"),
                "name": f"kitti15/{Path(left).name}",
            }
        )

    lefts = sg(os.path.join(root_12, "training", "colored_0", "*_10.png"))
    rights = sg(os.path.join(root_12, "training", "colored_1", "*_10.png"))
    gts = sg(os.path.join(root_12, "training", "disp_occ", "*_10.png"))
    sgms = sg(os.path.join(root_12, "training", "sgm_hole", "*_10.pfm"))
    mis = sg(os.path.join(root_12, "training", "sgm_hole", "*_10_mismatches.npy"))
    occ = sg(os.path.join(root_12, "training", "sgm_hole", "*_10_occlusion.npy"))
    for left, right, gt, sgm, mismatch, occlusion in zip(lefts, rights, gts, sgms, mis, occ):
        samples.append(
            {
                "left": left,
                "right": right,
                "gt_path": gt,
                "sgm_pfm": sgm,
                "mismatch": mismatch,
                "occlusion": occlusion,
                "pkrn_cache": sgm.replace(".pfm", "_pkrn.npy"),
                "name": f"kitti12/{Path(left).name}",
            }
        )
    return samples


def build_sample_list_eth3d(eth3d_root: str) -> list[dict]:
    sgm_base = os.path.join(eth3d_root, "training", "sgm_hole")
    image_paths = sorted(glob(os.path.join(eth3d_root, "*/im0.png")))
    samples: list[dict] = []
    for img0 in image_paths:
        scene = Path(img0).parent.name
        if scene == "training":
            continue
        img1 = os.path.join(Path(img0).parent, "im1.png")
        gt_pfm = os.path.join(Path(img0).parent, "disp0GT.pfm")
        sgm_dir = os.path.join(sgm_base, scene)
        sgm_pfm = os.path.join(sgm_dir, f"{scene}.pfm")
        mismatch = os.path.join(sgm_dir, f"{scene}_mismatches.npy")
        occlusion = os.path.join(sgm_dir, f"{scene}_occlusion.npy")
        missing = [p for p in (img1, gt_pfm, sgm_pfm, mismatch, occlusion) if not os.path.isfile(p)]
        if missing:
            continue
        samples.append(
            {
                "left": str(img0),
                "right": str(img1),
                "gt_path": gt_pfm,
                "sgm_pfm": sgm_pfm,
                "mismatch": mismatch,
                "occlusion": occlusion,
                "pkrn_cache": sgm_pfm.replace(".pfm", "_pkrn.npy"),
                "name": scene,
            }
        )
    return samples


def build_sample_list_sceneflow(sceneflow_root: str) -> list[dict]:
    focal = "35mm_focallength"
    scene = "scene_forwards"
    speed = "fast"
    prefix = f"{focal}_{scene}_{speed}_left_"

    img_dir_left = os.path.join(sceneflow_root, "frames_cleanpass", focal, scene, speed, "left")
    img_dir_right = os.path.join(sceneflow_root, "frames_cleanpass", focal, scene, speed, "right")
    gt_dir = os.path.join(sceneflow_root, "disparity", focal, scene, speed, "left")
    sgm_dir = os.path.join(sceneflow_root, "sgm_hole", focal, scene, speed, "left")

    samples: list[dict] = []
    for right_img in sorted(glob(os.path.join(img_dir_right, "*.png"))):
        frame = Path(right_img).stem
        left_img = os.path.join(img_dir_left, f"{frame}.png")
        gt_pfm = os.path.join(gt_dir, f"{frame}.pfm")
        sgm_pfm = os.path.join(sgm_dir, f"{prefix}{frame}.pfm")
        mismatch = sgm_pfm.replace(".pfm", "_mismatches.npy")
        occlusion = sgm_pfm.replace(".pfm", "_occlusion.npy")
        missing = [p for p in (left_img, gt_pfm, sgm_pfm) if not os.path.isfile(p)]
        if missing:
            continue
        samples.append(
            {
                "left": left_img,
                "right": right_img,
                "gt_path": gt_pfm,
                "sgm_pfm": sgm_pfm,
                "mismatch": mismatch if os.path.isfile(mismatch) else None,
                "occlusion": occlusion if os.path.isfile(occlusion) else None,
                "pkrn_cache": sgm_pfm.replace(".pfm", "_pkrn.npy"),
                "name": frame,
            }
        )
    return samples


def build_sample_list_middlebury(middlebury_root: str) -> list[dict]:
    training_root = normalize_middlebury_root(middlebury_root)
    samples: list[dict] = []
    for scene_dir in sorted(Path(training_root).iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        left = scene_dir / "im0.png"
        right = scene_dir / "im1.png"
        gt = scene_dir / "disp0GT.pfm"
        nocc = scene_dir / "mask0nocc.png"
        sgm_dir = scene_dir / "sgm_hole"
        sgm_pfm = sgm_dir / f"{scene}.pfm"
        mismatch = sgm_dir / f"{scene}_mismatches.npy"
        occlusion = sgm_dir / f"{scene}_occlusion.npy"
        missing = [p for p in (left, right, gt, nocc, sgm_pfm, mismatch, occlusion) if not p.exists()]
        if missing:
            continue
        samples.append(
            {
                "left": str(left),
                "right": str(right),
                "gt_path": str(gt),
                "nocc_path": str(nocc),
                "sgm_pfm": str(sgm_pfm),
                "mismatch": str(mismatch),
                "occlusion": str(occlusion),
                "pkrn_cache": str(sgm_pfm).replace(".pfm", "_pkrn.npy"),
                "name": scene,
            }
        )
    return samples


def load_pkrn_confidence(
    sample: dict,
    disparity_range: int,
    pkrn_min_dist: int,
    smooth_sigma: float,
) -> np.ndarray:
    cache_path = sample["pkrn_cache"]
    if os.path.exists(cache_path):
        pkrn_raw = np.load(cache_path).astype(np.float32)
    else:
        _, pkrn_raw, _ = run_sgm_with_confidence(
            left_path=sample["left"],
            right_path=sample["right"],
            disparity_range=disparity_range,
            smooth_sigma=0.0,
            pkrn_min_dist=pkrn_min_dist,
            verbose=False,
        )
        np.save(cache_path, pkrn_raw.astype(np.float32))
    if smooth_sigma > 0:
        return gaussian_filter(pkrn_raw, sigma=smooth_sigma).clip(0.0, 1.0).astype(np.float32)
    return pkrn_raw.astype(np.float32)


def load_protocol_confidence_maps(
    sample: dict,
    disparity_range: int,
    pkrn_min_dist: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample.get("mismatch") and sample.get("occlusion"):
        mismatch = np.load(sample["mismatch"]).astype(bool)
        occlusion = np.load(sample["occlusion"]).astype(bool)
        conf_align = (~(mismatch | occlusion)).astype(np.float32)
        conf_align = gaussian_filter(conf_align, sigma=smooth_sigma).astype(np.float32)
    else:
        sgm_disp = read_pfm(sample["sgm_pfm"])
        conf_align = np.ones_like(sgm_disp, dtype=np.float32)
    conf_route = load_pkrn_confidence(sample, disparity_range, pkrn_min_dist, smooth_sigma)
    return conf_align, conf_route


def load_gt_and_protocol_masks(dataset: str, sample: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset == "kitti":
        gt_disp, valid = read_kitti_gt(sample["gt_path"])
        return gt_disp, {"main": valid}

    gt_disp = read_pfm(sample["gt_path"])
    valid = (gt_disp > 0) & np.isfinite(gt_disp)

    if dataset == "eth3d":
        return gt_disp.astype(np.float32), {"main": valid}

    if dataset == "sceneflow":
        valid = valid & (gt_disp < 512.0)
        return gt_disp.astype(np.float32), {"main": valid}

    if dataset == "middlebury":
        nocc_raw = cv2.imread(sample["nocc_path"], cv2.IMREAD_UNCHANGED)
        nocc_mask = nocc_raw == 255
        return gt_disp.astype(np.float32), {
            "all_valid": valid,
            "nocc": valid & nocc_mask,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def compute_metrics(dataset: str, pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    mask = valid_mask & np.isfinite(pred) & np.isfinite(gt) & (pred >= 0)
    n = int(mask.sum())
    if n == 0:
        out = {key: float("nan") for key in DATASET_METRIC_KEYS[dataset]}
        out["n"] = 0
        return out

    pred_m = pred[mask].astype(np.float32)
    gt_m = gt[mask].astype(np.float32)
    err = np.abs(pred_m - gt_m)
    ratio = np.maximum(pred_m.clip(1e-6) / gt_m.clip(1e-6), gt_m.clip(1e-6) / pred_m.clip(1e-6))

    if dataset == "kitti":
        return {
            "epe": float(err.mean()),
            "d1": float((err > np.maximum(3.0, 0.05 * gt_m)).mean() * 100.0),
            "n": n,
        }

    if dataset == "eth3d":
        return {
            "epe": float(err.mean()),
            "bad1": float((err > 1.0).mean() * 100.0),
            "bad2": float((err > 2.0).mean() * 100.0),
            "d1": float((err > np.maximum(1.0, 0.05 * gt_m)).mean() * 100.0),
            "delta1": float((ratio < 1.25).mean()),
            "n": n,
        }

    if dataset == "sceneflow":
        return {
            "epe": float(err.mean()),
            "bad1": float((err > 1.0).mean() * 100.0),
            "bad2": float((err > 2.0).mean() * 100.0),
            "bad3": float((err > 3.0).mean() * 100.0),
            "delta1": float((ratio < 1.25).mean()),
            "n": n,
        }

    if dataset == "middlebury":
        return {
            "epe": float(err.mean()),
            "bad2": float((err > 2.0).mean() * 100.0),
            "delta1": float((ratio < 1.25).mean()),
            "n": n,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def aggregate(records: list[dict[str, float]], metric_keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys:
        vals = [r[key] for r in records if not np.isnan(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["image_n"] = len([r for r in records if not np.isnan(r[metric_keys[0]])])
    out["pixel_n"] = int(sum(r["n"] for r in records if not np.isnan(r[metric_keys[0]])))
    return out


def precision_proxy(hp_ratio: float, high_bits: int = 8, low_bits: int = 4) -> float:
    active_fraction = 0.5
    low_fraction = active_fraction * (1.0 - hp_ratio)
    return 1.0 - low_fraction * (1.0 - low_bits / float(high_bits))


def compute_attn_reduction(merge_layer: int, rep_count: int, total_tokens: int, n_blocks: int) -> float:
    sparse_blocks = max(0, n_blocks - merge_layer)
    if total_tokens <= 0 or n_blocks <= 0:
        return 0.0
    return float(sparse_blocks / float(n_blocks) * (1.0 - (rep_count / float(total_tokens)) ** 2))


def build_dataset_samples(args: argparse.Namespace, dataset: str) -> list[dict]:
    if dataset == "kitti":
        samples = build_sample_list_kitti(args.kitti_root)
    elif dataset == "eth3d":
        samples = build_sample_list_eth3d(args.eth3d_root)
    elif dataset == "sceneflow":
        samples = build_sample_list_sceneflow(args.sceneflow_root)
    elif dataset == "middlebury":
        samples = build_sample_list_middlebury(args.middlebury_root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    limit = args.max_samples if args.max_samples > 0 else None
    if dataset == "sceneflow":
        scene_limit = args.scene_flow_subset if args.scene_flow_subset > 0 else None
        if scene_limit is not None:
            limit = scene_limit if limit is None else min(limit, scene_limit)
    if limit is not None:
        samples = samples[:limit]
    return samples


def dataset_disparity_range(args: argparse.Namespace, dataset: str) -> int:
    if dataset == "kitti":
        return args.kitti_range
    if dataset == "eth3d":
        return args.eth3d_range
    if dataset == "sceneflow":
        return args.sceneflow_range
    if dataset == "middlebury":
        return args.middlebury_range
    raise ValueError(dataset)


def format_metric_block(metrics: dict[str, float], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = metrics.get(key, float("nan"))
        parts.append(f"{key}={value:.4f}" if not np.isnan(value) else f"{key}=nan")
    return "  ".join(parts)


def select_best_method(
    method_keys: list[str],
    method_meta: dict[str, dict[str, object]],
    fused_summary: dict[str, dict[str, dict[str, float]]],
    primary_protocol: str,
) -> dict[str, object]:
    ranked = []
    for key in method_keys:
        score = fused_summary[key][primary_protocol]["epe"]
        ranked.append((float("inf") if np.isnan(score) else score, key))
    ranked.sort(key=lambda item: item[0])
    best_key = ranked[0][1]
    best = dict(method_meta[best_key])
    best["primary_protocol"] = primary_protocol
    best["fused_metrics"] = fused_summary[best_key][primary_protocol]
    return best


def evaluate_dataset(
    args: argparse.Namespace,
    dataset: str,
    model,
) -> dict[str, object]:
    out_dir = os.path.join(args.out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    samples = build_dataset_samples(args, dataset)
    if not samples:
        raise RuntimeError(f"No samples found for dataset={dataset}")

    protocol_names = ["all_valid", "nocc"] if dataset == "middlebury" else ["main"]
    metric_keys = DATASET_METRIC_KEYS[dataset]
    primary_protocol = PRIMARY_PROTOCOL[dataset]
    n_blocks = len(list(model.pretrained.blocks))
    disparity_range = dataset_disparity_range(args, dataset)

    method_meta: dict[str, dict[str, object]] = {}
    dense_records: dict[str, dict[str, list[dict[str, float]]]] = {}
    fused_records: dict[str, dict[str, list[dict[str, float]]]] = {}
    csv_rows: list[dict[str, object]] = []

    def register_method(key: str, label: str, category: str, **meta: object) -> None:
        method_meta[key] = {"key": key, "label": label, "category": category, **meta}
        dense_records[key] = {protocol: [] for protocol in protocol_names}
        fused_records[key] = {protocol: [] for protocol in protocol_names}

    register_method("sgm", "SGM", "baseline")
    register_method("dense_align", "Dense DA2 + align", "baseline")
    for keep_ratio in args.keep_ratios:
        register_method(
            f"merge_{keep_ratio:.3f}",
            f"Merge FP32 keep={keep_ratio:.3f}",
            "merge",
            keep_ratio=float(keep_ratio),
        )
    for hp_ratio in args.adaptive_hp_ratios:
        register_method(
            f"wcaps_hp_{int(round(hp_ratio * 100)):02d}",
            f"Merge + Decoder W-CAPS INT{args.low_precision_bits} hp={int(round(hp_ratio * 100))}%",
            "adaptive",
            keep_ratio=float(args.adaptive_keep_ratio),
            hp_ratio=float(hp_ratio),
            low_bits=int(args.low_precision_bits),
            precision_proxy=float(precision_proxy(float(hp_ratio), low_bits=args.low_precision_bits)),
        )

    print(f"\n[{dataset}] samples={len(samples)}  disparity_range={disparity_range}  out={out_dir}")

    for index, sample in enumerate(tqdm(samples, desc=f"{dataset}", unit="img")):
        left_bgr = cv2.imread(sample["left"])
        if left_bgr is None:
            raise FileNotFoundError(f"Cannot read left image: {sample['left']}")

        gt_disp, protocol_masks = load_gt_and_protocol_masks(dataset, sample)
        gt_shape = gt_disp.shape[:2]
        sgm_disp_raw = read_pfm(sample["sgm_pfm"])
        conf_align, conf_route = load_protocol_confidence_maps(
            sample,
            disparity_range=disparity_range,
            pkrn_min_dist=args.pkrn_min_dist,
            smooth_sigma=args.conf_sigma,
        )
        sgm_eval = resize_map(sgm_disp_raw, gt_shape, cv2.INTER_NEAREST)
        conf_fuse = resize_map(conf_route, gt_shape, cv2.INTER_LINEAR)

        image_tensor, _ = model.image2tensor(left_bgr, args.input_size)
        patch_h = image_tensor.shape[-2] // 14
        patch_w = image_tensor.shape[-1] // 14
        total_tokens = int(patch_h * patch_w)

        def append_result(
            method_key: str,
            pred_aligned: np.ndarray,
            fused_pred: np.ndarray,
            rep_count: int | None = None,
            effective_keep_ratio: float | None = None,
            attn_reduction: float | None = None,
            hp_ratio: float | None = None,
            precision_proxy_value: float | None = None,
        ) -> None:
            for protocol, valid_mask in protocol_masks.items():
                dense_metrics = compute_metrics(dataset, pred_aligned, gt_disp, valid_mask)
                fused_metrics = compute_metrics(dataset, fused_pred, gt_disp, valid_mask)
                dense_records[method_key][protocol].append(dense_metrics)
                fused_records[method_key][protocol].append(fused_metrics)

                row = {
                    "dataset": dataset,
                    "index": index,
                    "image": sample["name"],
                    "protocol": protocol,
                    "method_key": method_key,
                    "label": method_meta[method_key]["label"],
                    "category": method_meta[method_key]["category"],
                    "rep_count": rep_count if rep_count is not None else "",
                    "effective_keep_ratio": f"{effective_keep_ratio:.6f}" if effective_keep_ratio is not None else "",
                    "attn_reduction": f"{attn_reduction:.6f}" if attn_reduction is not None else "",
                    "hp_ratio": f"{hp_ratio:.4f}" if hp_ratio is not None else "",
                    "low_bits": method_meta[method_key].get("low_bits", ""),
                    "precision_proxy": (
                        f"{precision_proxy_value:.6f}" if precision_proxy_value is not None else ""
                    ),
                }
                for key in metric_keys:
                    row[f"dense_{key}"] = f"{dense_metrics[key]:.6f}"
                    row[f"fused_{key}"] = f"{fused_metrics[key]:.6f}"
                csv_rows.append(row)

        depth_dense = model.infer_image(left_bgr, input_size=args.input_size)
        disp_dense, _, _ = align_depth_to_sgm(
            depth_mono=depth_dense,
            disparity_raw=sgm_disp_raw,
            confidence_map=conf_align,
            conf_threshold=args.align_conf_threshold,
        )
        disp_dense = resize_map(disp_dense, gt_shape, cv2.INTER_LINEAR)
        fused_dense = fuse_dispatch(
            strategy=args.fusion_strategy,
            sgm_disp=sgm_eval,
            da2_aligned=disp_dense,
            confidence_map=conf_fuse,
            image_bgr=left_bgr,
            theta_low=args.theta_low,
            theta_high=args.theta_high,
            detail_suppression=args.detail_suppression,
            residual_gain=args.residual_gain,
        )

        append_result("sgm", sgm_eval, sgm_eval)
        append_result("dense_align", disp_dense, fused_dense)

        for keep_ratio in args.keep_ratios:
            merge_plan = build_token_merge_groups(
                conf_map=conf_route,
                token_grid_size=(patch_h, patch_w),
                keep_ratio=float(keep_ratio),
            )
            rep_count = int(merge_plan["rep_count"])
            eff_keep_ratio = float(merge_plan["effective_keep_ratio"])
            attn_reduction = compute_attn_reduction(
                args.merge_layer,
                rep_count,
                total_tokens,
                n_blocks,
            )

            depth_merge = run_token_merged_da2(
                model=model,
                image_bgr=left_bgr,
                confidence_map=conf_route,
                keep_ratio=float(keep_ratio),
                input_size=args.input_size,
                merge_layer=args.merge_layer,
            )
            disp_merge, _, _ = align_depth_to_sgm(
                depth_mono=depth_merge,
                disparity_raw=sgm_disp_raw,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )
            disp_merge = resize_map(disp_merge, gt_shape, cv2.INTER_LINEAR)
            fused_merge = fuse_dispatch(
                strategy=args.fusion_strategy,
                sgm_disp=sgm_eval,
                da2_aligned=disp_merge,
                confidence_map=conf_fuse,
                image_bgr=left_bgr,
                theta_low=args.theta_low,
                theta_high=args.theta_high,
                detail_suppression=args.detail_suppression,
                residual_gain=args.residual_gain,
            )
            append_result(
                f"merge_{keep_ratio:.3f}",
                disp_merge,
                fused_merge,
                rep_count=rep_count,
                effective_keep_ratio=eff_keep_ratio,
                attn_reduction=attn_reduction,
            )

        adaptive_merge_plan = build_token_merge_groups(
            conf_map=conf_route,
            token_grid_size=(patch_h, patch_w),
            keep_ratio=float(args.adaptive_keep_ratio),
        )
        adaptive_rep_count = int(adaptive_merge_plan["rep_count"])
        adaptive_eff_keep_ratio = float(adaptive_merge_plan["effective_keep_ratio"])
        adaptive_attn_reduction = compute_attn_reduction(
            args.merge_layer,
            adaptive_rep_count,
            total_tokens,
            n_blocks,
        )
        for hp_ratio in args.adaptive_hp_ratios:
            depth_adaptive = run_decoder_weight_caps_merged_da2(
                model=model,
                image_bgr=left_bgr,
                confidence_map=conf_route,
                keep_ratio=float(args.adaptive_keep_ratio),
                input_size=args.input_size,
                merge_layer=args.merge_layer,
                decoder_high_precision_ratio=float(hp_ratio),
                low_precision_bits=args.low_precision_bits,
                decoder_conf_weight=args.decoder_conf_weight,
                decoder_texture_weight=0.0,
                decoder_variance_weight=0.0,
                decoder_stage_policy="coarse_only",
            )
            disp_adaptive, _, _ = align_depth_to_sgm(
                depth_mono=depth_adaptive,
                disparity_raw=sgm_disp_raw,
                confidence_map=conf_align,
                conf_threshold=args.align_conf_threshold,
            )
            disp_adaptive = resize_map(disp_adaptive, gt_shape, cv2.INTER_LINEAR)
            fused_adaptive = fuse_dispatch(
                strategy=args.fusion_strategy,
                sgm_disp=sgm_eval,
                da2_aligned=disp_adaptive,
                confidence_map=conf_fuse,
                image_bgr=left_bgr,
                theta_low=args.theta_low,
                theta_high=args.theta_high,
                detail_suppression=args.detail_suppression,
                residual_gain=args.residual_gain,
            )
            append_result(
                f"wcaps_hp_{int(round(hp_ratio * 100)):02d}",
                disp_adaptive,
                fused_adaptive,
                rep_count=adaptive_rep_count,
                effective_keep_ratio=adaptive_eff_keep_ratio,
                attn_reduction=adaptive_attn_reduction,
                hp_ratio=float(hp_ratio),
                precision_proxy_value=precision_proxy(float(hp_ratio), low_bits=args.low_precision_bits),
            )

    dense_summary = {
        method_key: {
            protocol: aggregate(protocol_records, metric_keys)
            for protocol, protocol_records in protocol_map.items()
        }
        for method_key, protocol_map in dense_records.items()
    }
    fused_summary = {
        method_key: {
            protocol: aggregate(protocol_records, metric_keys)
            for protocol, protocol_records in protocol_map.items()
        }
        for method_key, protocol_map in fused_records.items()
    }

    method_order = list(method_meta.keys())
    merge_keys = [key for key, meta in method_meta.items() if meta["category"] == "merge"]
    adaptive_keys = [key for key, meta in method_meta.items() if meta["category"] == "adaptive"]

    best_merge = select_best_method(merge_keys, method_meta, fused_summary, primary_protocol)
    best_adaptive = select_best_method(adaptive_keys, method_meta, fused_summary, primary_protocol)

    csv_path = os.path.join(out_dir, "per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_lines: list[str] = []
    summary_lines.append(f"Dataset: {dataset}")
    summary_lines.append(f"Samples: {len(samples)}")
    summary_lines.append(f"Fusion strategy: {args.fusion_strategy}")
    summary_lines.append(
        "Edge-aware params: "
        f"theta_low={args.theta_low:.3f}, theta_high={args.theta_high:.3f}, "
        f"detail_suppression={args.detail_suppression:.3f}, residual_gain={args.residual_gain:.3f}"
    )
    summary_lines.append(f"Merge layer: {args.merge_layer}")
    summary_lines.append(f"Adaptive keep ratio: {args.adaptive_keep_ratio:.3f}")
    summary_lines.append(f"Low precision bits: INT{args.low_precision_bits}")
    summary_lines.append("")
    summary_lines.append(
        f"Best merge ({primary_protocol} / fused EPE): "
        f"{best_merge['label']}  keep={best_merge.get('keep_ratio', 'n/a')}"
    )
    summary_lines.append(
        f"Best adaptive ({primary_protocol} / fused EPE): "
        f"{best_adaptive['label']}  hp={best_adaptive.get('hp_ratio', 'n/a')}"
    )

    for protocol in protocol_names:
        summary_lines.append("")
        summary_lines.append(f"[Protocol: {protocol}]")
        for method_key in method_order:
            meta = method_meta[method_key]
            line = f"{meta['label']}"
            if "keep_ratio" in meta:
                line += f" | keep={float(meta['keep_ratio']):.3f}"
            if "hp_ratio" in meta:
                line += f" | hp={float(meta['hp_ratio']):.2f}"
            if "low_bits" in meta:
                line += f" | low_bits=INT{int(meta['low_bits'])}"
            if "precision_proxy" in meta:
                line += f" | precision_proxy={float(meta['precision_proxy']):.4f}"
            protocol_rows = [row for row in csv_rows if row["method_key"] == method_key and row["protocol"] == protocol]
            rep_counts = [int(row["rep_count"]) for row in protocol_rows if row["rep_count"] != ""]
            keep_vals = [float(row["effective_keep_ratio"]) for row in protocol_rows if row["effective_keep_ratio"] != ""]
            attn_vals = [float(row["attn_reduction"]) for row in protocol_rows if row["attn_reduction"] != ""]
            if rep_counts:
                line += (
                    f" | rep_count={int(round(np.mean(rep_counts)))}"
                    f" | eff_keep={np.mean(keep_vals):.4f}"
                    f" | attn_reduction={np.mean(attn_vals):.4f}"
                )
            line += "\n"
            line += f"  Dense: {format_metric_block(dense_summary[method_key][protocol], metric_keys)}\n"
            line += f"  Fused: {format_metric_block(fused_summary[method_key][protocol], metric_keys)}"
            summary_lines.append(line)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    best_configs = {
        "dataset": dataset,
        "sample_count": len(samples),
        "primary_protocol": primary_protocol,
        "fusion_strategy": args.fusion_strategy,
        "merge_layer": args.merge_layer,
        "adaptive_keep_ratio": float(args.adaptive_keep_ratio),
        "low_precision_bits": int(args.low_precision_bits),
        "theta_low": float(args.theta_low),
        "theta_high": float(args.theta_high),
        "detail_suppression": float(args.detail_suppression),
        "residual_gain": float(args.residual_gain),
        "best_merge": best_merge,
        "best_adaptive": best_adaptive,
    }
    best_json_path = os.path.join(out_dir, "best_configs.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)

    print(f"[{dataset}] summary -> {summary_path}")
    print(f"[{dataset}] per-image -> {csv_path}")
    print(f"[{dataset}] best-configs -> {best_json_path}")

    return {
        "dataset": dataset,
        "out_dir": out_dir,
        "summary_path": summary_path,
        "csv_path": csv_path,
        "best_json_path": best_json_path,
        "best_merge": best_merge,
        "best_adaptive": best_adaptive,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified merge + decoder W-CAPS evaluation on KITTI / ETH3D / SceneFlow / Middlebury."
    )
    parser.add_argument("--dataset", default="all", choices=["kitti", "eth3d", "sceneflow", "middlebury", "all"])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(DA2_MODEL_CONFIGS.keys()))
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--merge-layer", type=int, default=DEFAULT_PRUNE_LAYER)
    parser.add_argument("--keep-ratios", type=parse_float_list, default=parse_float_list("0.60,0.70,0.80,0.814,0.90"))
    parser.add_argument("--adaptive-keep-ratio", type=float, default=0.814)
    parser.add_argument("--adaptive-hp-ratios", type=parse_float_list, default=parse_float_list("0.25,0.50,0.75"))
    parser.add_argument("--low-precision-bits", type=int, default=4)
    parser.add_argument("--fusion-strategy", default=DEFAULT_FUSION_STRATEGY, choices=["edge_aware_residual"])
    parser.add_argument("--theta-low", type=float, default=DEFAULT_EDGE_THETA_LOW)
    parser.add_argument("--theta-high", type=float, default=DEFAULT_EDGE_THETA_HIGH)
    parser.add_argument("--detail-suppression", type=float, default=DEFAULT_EDGE_DETAIL_SUPPRESSION)
    parser.add_argument("--residual-gain", type=float, default=DEFAULT_EDGE_RESIDUAL_GAIN)
    parser.add_argument("--align-conf-threshold", type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD)
    parser.add_argument("--conf-sigma", type=float, default=DEFAULT_CONF_SIGMA)
    parser.add_argument("--pkrn-min-dist", type=int, default=DEFAULT_PKRN_MIN_DIST)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--scene-flow-subset", type=int, default=50)
    parser.add_argument("--decoder-conf-weight", type=float, default=1.0)
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    parser.add_argument("--sceneflow-root", default=DEFAULT_SCENEFLOW_ROOT)
    parser.add_argument("--middlebury-root", default=DEFAULT_MIDDLEBURY_ROOT)
    parser.add_argument("--kitti-range", type=int, default=DATASET_DISPARITY_RANGE["kitti"])
    parser.add_argument("--eth3d-range", type=int, default=DATASET_DISPARITY_RANGE["eth3d"])
    parser.add_argument("--sceneflow-range", type=int, default=DATASET_DISPARITY_RANGE["sceneflow"])
    parser.add_argument("--middlebury-range", type=int, default=DATASET_DISPARITY_RANGE["middlebury"])
    parser.add_argument("--out-dir", default=default_results_dir("eval_merge_adaptive"))
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    model = load_da2_model(args.encoder, args.weights, device)

    datasets = DATASET_ORDER if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        evaluate_dataset(args, dataset, model)


if __name__ == "__main__":
    main()
