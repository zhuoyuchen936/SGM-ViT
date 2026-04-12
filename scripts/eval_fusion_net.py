#!/usr/bin/env python3
"""Evaluate heuristic fusion and FusionNet-v1 on cached datasets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch


_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.fusion_net import (  # noqa: E402
    build_fusion_inputs,
    get_fusion_net_runtime_config,
    load_fusion_net,
    run_fusion_net_refinement,
)


DEFAULT_CACHE_ROOT = os.path.join(_PROJECT_DIR, "artifacts", "fusion_cache")
DEFAULT_OUT_ROOT = os.path.join(_PROJECT_DIR, "results", "eval_fusion_net")
DEFAULT_WEIGHTS = os.path.join(_PROJECT_DIR, "artifacts", "fusion_net", "kitti_finetune", "best.pt")
EVAL_SPLITS = {
    "kitti": "val",
    "sceneflow": "test",
    "eth3d": "eval",
    "middlebury": "eval",
}
DATASET_ORDER = ("kitti", "sceneflow", "eth3d", "middlebury")
DATASET_METRIC_KEYS = {
    "kitti": ["epe", "d1"],
    "eth3d": ["epe", "bad1", "bad2", "d1", "delta1"],
    "sceneflow": ["epe", "bad1", "bad2", "bad3", "delta1"],
    "middlebury": ["epe", "bad2", "delta1"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FusionNet-v1 from offline cache.")
    parser.add_argument("--dataset", default="all", choices=["all", "kitti", "sceneflow", "eth3d", "middlebury"])
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


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
        return {"epe": float(err.mean()), "d1": float((err > np.maximum(3.0, 0.05 * gt_m)).mean() * 100.0), "n": n}
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
    raise ValueError(dataset)


def aggregate(records: list[dict[str, float]], metric_keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys:
        vals = [row[key] for row in records if np.isfinite(row[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["image_n"] = len([row for row in records if np.isfinite(row[metric_keys[0]])])
    out["pixel_n"] = int(sum(row["n"] for row in records if np.isfinite(row[metric_keys[0]])))
    return out


def compute_boundary_mask(
    gt_disp: np.ndarray,
    valid_gt: np.ndarray,
    grad_threshold: float = 1.5,
    band_radius: int = 3,
) -> np.ndarray:
    gx = cv2.Sobel(gt_disp.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_disp.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    edge = (grad > grad_threshold) & valid_gt
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))
    band = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0
    return band & valid_gt


def compute_noise_score(
    pred_disp: np.ndarray,
    confidence_map: np.ndarray,
    image_bgr: np.ndarray,
    valid_mask: np.ndarray,
    conf_threshold: float = 0.55,
    grad_threshold: float = 0.15,
    smooth_sigma: float = 1.5,
) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / max(float(np.percentile(grad, 99.0)), 1e-6)
    region = valid_mask & (confidence_map >= conf_threshold) & (grad <= grad_threshold) & np.isfinite(pred_disp)
    if int(region.sum()) < 32:
        return float("nan")
    smooth = cv2.GaussianBlur(pred_disp.astype(np.float32), (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    return float(np.std((pred_disp.astype(np.float32) - smooth.astype(np.float32))[region]))


def compute_boundary_epe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    band = compute_boundary_mask(gt, valid_mask)
    mask = band & np.isfinite(pred) & np.isfinite(gt)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.abs(pred[mask].astype(np.float32) - gt[mask].astype(np.float32)).mean())


def format_metric_block(metrics: dict[str, float], metric_keys: list[str]) -> str:
    return "  ".join(f"{key}={metrics[key]:.4f}" for key in metric_keys if key in metrics)


def resolve_out_dir(args: argparse.Namespace) -> str:
    if args.out_dir:
        return args.out_dir
    leaf = "all" if args.dataset == "all" else args.dataset
    return os.path.join(DEFAULT_OUT_ROOT, leaf)


def build_overall_table(
    datasets: tuple[str, ...],
    per_dataset_summary: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    lines = ["[overall_table]"]
    overall_table: dict[str, dict[str, dict[str, float]]] = {
        method: {} for method in ("mono_aligned", "heuristic_fused", "fusion_net_refined")
    }
    for method in overall_table:
        chunks = [f"{method}:"]
        for dataset in datasets:
            metrics = per_dataset_summary[dataset][method]
            primary_keys = DATASET_METRIC_KEYS[dataset][:2]
            primary_text = ", ".join(
                f"{key}={metrics[key]:.4f}" for key in primary_keys if key in metrics and np.isfinite(metrics[key])
            )
            chunks.append(
                f"{dataset}[{primary_text}; boundary_epe={metrics['boundary_epe']:.4f}; "
                f"flat_region_noise={metrics['flat_region_noise']:.4f}]"
            )
            overall_table[method][dataset] = metrics
        lines.append("  ".join(chunks))
    lines.append("")
    return lines, overall_table


def build_ranking(
    datasets: tuple[str, ...],
    per_dataset_summary: dict[str, dict[str, dict[str, float]]],
) -> dict[str, list[dict[str, float | str]]]:
    ranking: dict[str, list[dict[str, float | str]]] = {}
    for dataset in datasets:
        primary_metric = DATASET_METRIC_KEYS[dataset][0]
        rows: list[dict[str, float | str]] = []
        for method, metrics in per_dataset_summary[dataset].items():
            rows.append(
                {
                    "method": method,
                    "primary_metric": primary_metric,
                    "primary_value": float(metrics.get(primary_metric, float("nan"))),
                    "boundary_epe": float(metrics.get("boundary_epe", float("nan"))),
                    "flat_region_noise": float(metrics.get("flat_region_noise", float("nan"))),
                }
            )
        rows.sort(key=lambda row: (not np.isfinite(float(row["primary_value"])), float(row["primary_value"])))
        ranking[dataset] = rows
    return ranking


def main() -> None:
    args = build_parser().parse_args()
    out_dir = resolve_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    runtime_cfg = get_fusion_net_runtime_config(args.weights)
    model = load_fusion_net(
        args.weights,
        device,
        max_residual_scale=float(runtime_cfg["max_residual_scale"]),
    )
    datasets = DATASET_ORDER if args.dataset == "all" else (args.dataset,)

    csv_rows: list[dict[str, object]] = []
    summary_lines: list[str] = []
    json_summary: dict[str, object] = {
        "weights": args.weights,
        "fusion_net_runtime_config": runtime_cfg,
        "per_dataset": {},
    }
    per_dataset_summary: dict[str, dict[str, dict[str, float]]] = {}

    for dataset in datasets:
        split = EVAL_SPLITS[dataset]
        cache_dir = os.path.join(args.cache_root, dataset, split)
        if not os.path.isdir(cache_dir):
            raise RuntimeError(f"Cache split not found: {cache_dir}")
        files = sorted(os.path.join(cache_dir, name) for name in os.listdir(cache_dir) if name.endswith(".npz"))
        if not files:
            raise RuntimeError(f"No cache files found in {cache_dir}")

        metric_keys = DATASET_METRIC_KEYS[dataset]
        per_method = {key: [] for key in ("mono_aligned", "heuristic_fused", "fusion_net_refined")}
        summary_lines.append(f"[{dataset} / {split}]")
        summary_lines.append(f"Samples: {len(files)}")

        for file_path in files:
            data = np.load(file_path)
            rgb = data["rgb"].astype(np.uint8)
            image_bgr = rgb[:, :, ::-1].copy()
            gt = data["gt_disp"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
            mono = data["mono_disp_aligned"].astype(np.float32)
            heuristic = data["fused_base"].astype(np.float32)
            conf = data["confidence_map"].astype(np.float32)
            sgm = data["sgm_disp"].astype(np.float32)
            detail = data["detail_score"].astype(np.float32)

            bundle = build_fusion_inputs(
                image_bgr=image_bgr,
                mono_disp_aligned=mono,
                sgm_disp=sgm,
                confidence_map=conf,
                fused_base=heuristic,
                detail_score=detail,
                disp_scale=float(data["disp_scale"]),
            )
            refined, residual = run_fusion_net_refinement(
                model,
                bundle,
                device=device,
                apply_anchor_gate=bool(runtime_cfg["apply_anchor_gate"]),
                anchor_gate_strength=float(runtime_cfg["anchor_gate_strength"]),
            )

            method_preds = {
                "mono_aligned": mono,
                "heuristic_fused": heuristic,
                "fusion_net_refined": refined,
            }
            for method_key, pred in method_preds.items():
                metrics = compute_metrics(dataset, pred, gt, valid)
                metrics["boundary_epe"] = compute_boundary_epe(pred, gt, valid)
                metrics["flat_region_noise"] = compute_noise_score(pred, conf, image_bgr, valid)
                per_method[method_key].append(metrics)
                row = {
                    "dataset": dataset,
                    "split": split,
                    "sample_name": str(data["sample_name"]),
                    "method": method_key,
                    "boundary_epe": metrics["boundary_epe"],
                    "flat_region_noise": metrics["flat_region_noise"],
                    "residual_abs_mean": float(np.mean(np.abs(residual))) if method_key == "fusion_net_refined" else "",
                }
                for key in metric_keys:
                    row[key] = metrics[key]
                csv_rows.append(row)

        dataset_summary: dict[str, dict[str, float]] = {}
        for method_key, rows in per_method.items():
            agg = aggregate(rows, metric_keys)
            agg["boundary_epe"] = float(np.nanmean([row["boundary_epe"] for row in rows]))
            agg["flat_region_noise"] = float(np.nanmean([row["flat_region_noise"] for row in rows]))
            dataset_summary[method_key] = agg
            summary_lines.append(
                f"{method_key}: {format_metric_block(agg, metric_keys)}  "
                f"boundary_epe={agg['boundary_epe']:.4f}  flat_region_noise={agg['flat_region_noise']:.4f}"
            )
        summary_lines.append("")
        json_summary["per_dataset"][dataset] = dataset_summary
        per_dataset_summary[dataset] = dataset_summary

    overall_table_lines, overall_table = build_overall_table(datasets, per_dataset_summary)
    summary_lines.extend(overall_table_lines)
    ranking = build_ranking(datasets, per_dataset_summary)
    json_summary["overall_table"] = overall_table
    json_summary["ranking"] = ranking
    summary_lines.append("[ranking]")
    for dataset in datasets:
        ranking_text = "  ".join(
            f"{row['method']}({row['primary_metric']}={float(row['primary_value']):.4f})"
            for row in ranking[dataset]
        )
        summary_lines.append(f"{dataset}: {ranking_text}")
    summary_lines.append("")

    kitti_summary = json_summary["per_dataset"].get("kitti", {})
    promote = False
    if kitti_summary:
        base = kitti_summary["heuristic_fused"]
        refined = kitti_summary["fusion_net_refined"]
        improve_main = (
            (np.isfinite(refined.get("epe", np.nan)) and np.isfinite(base.get("epe", np.nan)) and refined["epe"] < base["epe"])
            or (np.isfinite(refined.get("d1", np.nan)) and np.isfinite(base.get("d1", np.nan)) and refined["d1"] < base["d1"])
        )
        no_boundary_regress = (
            np.isfinite(refined.get("boundary_epe", np.nan))
            and np.isfinite(base.get("boundary_epe", np.nan))
            and refined["boundary_epe"] <= base["boundary_epe"]
        )
        promote = bool(improve_main and no_boundary_regress)
    json_summary["demo_backend_recommendation"] = {
        "backend": "net" if promote else "heuristic",
        "net_recommended": promote,
        "rule": "promote only if KITTI refined improves EPE or D1 and does not regress boundary_epe",
    }
    json_summary["demo_default_backend"] = "net" if promote else "heuristic"
    json_summary["net_recommended"] = promote
    summary_lines.append("[demo_backend_recommendation]")
    summary_lines.append(
        f"backend={json_summary['demo_default_backend']}  net_recommended={json_summary['net_recommended']}"
    )
    summary_lines.append("")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")

    metric_fieldnames = sorted({key for row in csv_rows for key in row.keys() if key not in {
        "dataset", "split", "sample_name", "method", "boundary_epe", "flat_region_noise", "residual_abs_mean",
    }})
    csv_fieldnames = [
        "dataset",
        "split",
        "sample_name",
        "method",
        *metric_fieldnames,
        "boundary_epe",
        "flat_region_noise",
        "residual_abs_mean",
    ]
    csv_path = os.path.join(out_dir, "per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    compat_json_path = os.path.join(out_dir, "best_ckpt_metrics.json")
    with open(compat_json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    print(f"summary -> {summary_path}")
    print(f"per-image -> {csv_path}")
    print(f"json -> {json_path}")


if __name__ == "__main__":
    main()
