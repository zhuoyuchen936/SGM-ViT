#!/usr/bin/env python3
"""Run one merge/adaptive demo per dataset using eval-selected best configs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from scripts.common_config import (  # noqa: E402
    DEFAULT_ETH3D_ROOT,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MIDDLEBURY_ROOT,
    DEFAULT_SCENEFLOW_ROOT,
    DEFAULT_WEIGHTS,
    default_results_dir,
)
from scripts.eval_merge_adaptive import (  # noqa: E402
    DATASET_DISPARITY_RANGE,
    build_sample_list_eth3d,
    build_sample_list_kitti,
    build_sample_list_middlebury,
    build_sample_list_sceneflow,
)


def pick_sample(dataset: str, args: argparse.Namespace) -> dict:
    if dataset == "kitti":
        return build_sample_list_kitti(args.kitti_root)[0]
    if dataset == "eth3d":
        return build_sample_list_eth3d(args.eth3d_root)[0]
    if dataset == "sceneflow":
        return build_sample_list_sceneflow(args.sceneflow_root)[0]
    if dataset == "middlebury":
        return build_sample_list_middlebury(args.middlebury_root)[0]
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four dataset demos using best merge/adaptive configs.")
    parser.add_argument("--eval-root", default=default_results_dir("eval_merge_adaptive"))
    parser.add_argument("--out-dir", default=default_results_dir("demo_merge_adaptive"))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    parser.add_argument("--sceneflow-root", default=DEFAULT_SCENEFLOW_ROOT)
    parser.add_argument("--middlebury-root", default=DEFAULT_MIDDLEBURY_ROOT)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary_lines = []

    for dataset in ("kitti", "eth3d", "sceneflow", "middlebury"):
        best_path = os.path.join(args.eval_root, dataset, "best_configs.json")
        with open(best_path, "r", encoding="utf-8") as f:
            best = json.load(f)

        sample = pick_sample(dataset, args)
        dataset_out = os.path.join(args.out_dir, dataset)
        os.makedirs(dataset_out, exist_ok=True)

        cmd = [
            sys.executable,
            os.path.join(_PROJECT_DIR, "demo.py"),
            "--left",
            sample["left"],
            "--right",
            sample["right"],
            "--gt-disparity",
            sample["gt_path"],
            "--weights",
            args.weights,
            "--out-dir",
            dataset_out,
            "--disparity-range",
            str(DATASET_DISPARITY_RANGE[dataset]),
            "--merge-layer",
            str(best["merge_layer"]),
            "--merge-keep-ratio",
            str(best["best_merge"]["keep_ratio"]),
            "--adaptive-keep-ratio",
            str(best["adaptive_keep_ratio"]),
            "--decoder-high-precision-ratio",
            str(best["best_adaptive"]["hp_ratio"]),
            "--decoder-low-bits",
            str(best["low_precision_bits"]),
            "--theta-low",
            str(best["theta_low"]),
            "--theta-high",
            str(best["theta_high"]),
            "--detail-suppression",
            str(best["detail_suppression"]),
            "--residual-gain",
            str(best["residual_gain"]),
        ]
        if args.cpu:
            cmd.append("--cpu")

        print(f"[{dataset}] running demo -> {dataset_out}")
        subprocess.run(cmd, check=True)

        summary_lines.append(f"Dataset: {dataset}")
        summary_lines.append(f"  Sample: {sample['left']}")
        summary_lines.append(f"  Merge keep ratio: {best['best_merge']['keep_ratio']}")
        summary_lines.append(f"  Adaptive keep ratio: {best['adaptive_keep_ratio']}")
        summary_lines.append(f"  Adaptive hp ratio: {best['best_adaptive']['hp_ratio']}")
        summary_lines.append(f"  Fusion: {best['fusion_strategy']}")
        summary_lines.append(f"  Output: {dataset_out}")
        summary_lines.append("")

    readme_path = os.path.join(args.out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")
    print(f"Overview -> {readme_path}")


if __name__ == "__main__":
    main()
