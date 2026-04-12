#!/usr/bin/env python3
"""
Batch-generate ``sgm_hole`` assets for ETH3D, SceneFlow Driving, and Middlebury.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
import time
from glob import glob
from pathlib import Path

import numpy as np

import core._paths  # noqa: F401
from core.sgm_wrapper import run_sgm_with_confidence
from core.stereo_datasets import _read_pfm
from scripts.common_config import (
    DEFAULT_ETH3D_ROOT,
    DEFAULT_MIDDLEBURY_ROOT,
    DEFAULT_SCENEFLOW_ROOT,
)

SCENEFLOW_SUBPATH = os.path.join("35mm_focallength", "scene_forwards", "fast")


def write_pfm_atomic(path: str, image: np.ndarray) -> None:
    image = np.ascontiguousarray(np.flipud(np.asarray(image, dtype=np.float32)))
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pfm", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(b"Pf\n")
            f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
            f.write(b"-1.0\n")
            image.tofile(f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_npy_atomic(path: str, array: np.ndarray) -> None:
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_eth3d_samples(root: str) -> list[dict]:
    samples: list[dict] = []
    for img0 in sorted(glob(os.path.join(root, "*/im0.png"))):
        scene_dir = Path(img0).parent
        scene = scene_dir.name
        if scene == "training":
            continue
        out_dir = os.path.join(root, "training", "sgm_hole", scene)
        samples.append({
            "dataset": "eth3d",
            "scene": scene,
            "left": str(scene_dir / "im0.png"),
            "right": str(scene_dir / "im1.png"),
            "gt": str(scene_dir / "disp0GT.pfm"),
            "pfm": os.path.join(out_dir, f"{scene}.pfm"),
            "mismatch": os.path.join(out_dir, f"{scene}_mismatches.npy"),
            "occlusion": os.path.join(out_dir, f"{scene}_occlusion.npy"),
            "pkrn": os.path.join(out_dir, f"{scene}_pkrn.npy"),
        })
    return samples


def build_sceneflow_samples(root: str) -> list[dict]:
    samples: list[dict] = []
    right_dir = os.path.join(root, "frames_cleanpass", SCENEFLOW_SUBPATH, "right")
    sgm_dir = os.path.join(root, "sgm_hole", SCENEFLOW_SUBPATH, "left")
    for right_img in sorted(glob(os.path.join(right_dir, "*.png"))):
        frame = Path(right_img).stem
        basename = f"35mm_focallength_scene_forwards_fast_left_{frame}"
        samples.append({
            "dataset": "sceneflow",
            "scene": frame,
            "left": os.path.join(root, "frames_cleanpass", SCENEFLOW_SUBPATH, "left", f"{frame}.png"),
            "right": right_img,
            "gt": os.path.join(root, "disparity", SCENEFLOW_SUBPATH, "left", f"{frame}.pfm"),
            "pfm": os.path.join(sgm_dir, f"{basename}.pfm"),
            "mismatch": os.path.join(sgm_dir, f"{basename}_mismatches.npy"),
            "occlusion": os.path.join(sgm_dir, f"{basename}_occlusion.npy"),
            "pkrn": os.path.join(sgm_dir, f"{basename}_pkrn.npy"),
        })
    return samples


def build_middlebury_samples(root: str) -> list[dict]:
    samples: list[dict] = []
    for img0 in sorted(glob(os.path.join(root, "MiddEval3", "trainingQ", "*/im0.png"))):
        scene_dir = Path(img0).parent
        scene = scene_dir.name
        out_dir = scene_dir / "sgm_hole"
        samples.append({
            "dataset": "middlebury",
            "scene": scene,
            "left": str(scene_dir / "im0.png"),
            "right": str(scene_dir / "im1.png"),
            "gt": str(scene_dir / "disp0GT.pfm"),
            "pfm": str(out_dir / f"{scene}.pfm"),
            "mismatch": str(out_dir / f"{scene}_mismatches.npy"),
            "occlusion": str(out_dir / f"{scene}_occlusion.npy"),
            "pkrn": str(out_dir / f"{scene}_pkrn.npy"),
        })
    return samples


def dataset_entries(args: argparse.Namespace) -> list[tuple[str, int, list[dict]]]:
    ordered = [
        ("middlebury", args.middlebury_range, build_middlebury_samples(args.middlebury_root)),
        ("eth3d", args.eth3d_range, build_eth3d_samples(args.eth3d_root)),
        ("sceneflow", args.sceneflow_range, build_sceneflow_samples(args.sceneflow_root)),
    ]
    if args.dataset == "all":
        return ordered
    return [item for item in ordered if item[0] == args.dataset]


def should_skip(sample: dict, overwrite: bool, only_missing: bool) -> bool:
    outputs = (sample["pfm"], sample["mismatch"], sample["occlusion"], sample["pkrn"])
    if only_missing:
        return all(os.path.exists(p) for p in outputs)
    if overwrite:
        return False
    return all(os.path.exists(p) for p in outputs)


def process_sample(sample: dict, disparity_range: int, dry_run: bool) -> tuple[bool, str, float]:
    missing_inputs = [p for p in (sample["left"], sample["right"], sample["gt"]) if not os.path.exists(p)]
    if missing_inputs:
        return False, f"missing inputs: {missing_inputs}", 0.0
    if dry_run:
        return True, "dry-run", 0.0

    start = time.perf_counter()
    _, _, disp_filled, debug = run_sgm_with_confidence(
        sample["left"],
        sample["right"],
        disparity_range=disparity_range,
        smooth_sigma=0.0,
        verbose=False,
        return_debug=True,
    )
    write_pfm_atomic(sample["pfm"], disp_filled.astype(np.float32))
    write_npy_atomic(sample["mismatch"], debug["mismatches"].astype(bool))
    write_npy_atomic(sample["occlusion"], debug["occlusion"].astype(bool))
    write_npy_atomic(sample["pkrn"], debug["confidence_raw"].astype(np.float32))
    return True, "ok", time.perf_counter() - start


def validate_sample(sample: dict) -> None:
    disp = _read_pfm(sample["pfm"])
    mismatch = np.load(sample["mismatch"]).astype(bool)
    occlusion = np.load(sample["occlusion"]).astype(bool)
    conf = np.load(sample["pkrn"]).astype(np.float32)
    if disp.shape != mismatch.shape or disp.shape != occlusion.shape or disp.shape != conf.shape:
        raise RuntimeError(f"shape mismatch for {sample['dataset']}:{sample['scene']}")


def configure_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-compute sgm_hole assets.")
    p.add_argument("--dataset", choices=("eth3d", "sceneflow", "middlebury", "all"), default="all")
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--only-missing", action="store_true")
    p.add_argument("--log-file", default="")
    p.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    p.add_argument("--sceneflow-root", default=DEFAULT_SCENEFLOW_ROOT)
    p.add_argument("--middlebury-root", default=DEFAULT_MIDDLEBURY_ROOT)
    p.add_argument("--eth3d-range", type=int, default=128)
    p.add_argument("--sceneflow-range", type=int, default=256)
    p.add_argument("--middlebury-range", type=int, default=192)
    return p


def main() -> None:
    args = build_parser().parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("/tmp", f"sgm_rebuild_{stamp}")
    os.makedirs(report_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(report_dir, "precompute_sgm_hole.log")
    manifest_path = os.path.join(report_dir, "manifest.csv")
    configure_logging(log_file)

    manifest_rows: list[dict[str, str]] = []
    for dataset_name, disparity_range, samples in dataset_entries(args):
        if args.start_index:
            samples = samples[args.start_index:]
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        logging.info("=== %s | samples=%d | disparity_range=%d ===", dataset_name, len(samples), disparity_range)
        first_done: dict | None = None
        success = skipped = failed = 0
        dataset_time = 0.0

        for idx, sample in enumerate(samples, start=1):
            if should_skip(sample, args.overwrite, args.only_missing):
                skipped += 1
                manifest_rows.append({
                    "dataset": dataset_name,
                    "scene": sample["scene"],
                    "status": "skipped",
                    "seconds": "0.000",
                    "message": "outputs already exist",
                })
                continue

            ok, message, seconds = process_sample(sample, disparity_range, args.dry_run)
            dataset_time += seconds
            manifest_rows.append({
                "dataset": dataset_name,
                "scene": sample["scene"],
                "status": "ok" if ok else "failed",
                "seconds": f"{seconds:.3f}",
                "message": message,
            })
            if ok:
                success += 1
                if first_done is None and not args.dry_run:
                    first_done = sample
                logging.info("[%s %d/%d] %s ok in %.2fs", dataset_name, idx, len(samples), sample["scene"], seconds)
            else:
                failed += 1
                logging.error("[%s %d/%d] %s failed: %s", dataset_name, idx, len(samples), sample["scene"], message)

        if first_done is not None:
            validate_sample(first_done)
            logging.info("[%s] validation ok on sample %s", dataset_name, first_done["scene"])
        logging.info(
            "[%s] summary: success=%d skipped=%d failed=%d total_time=%.1fs",
            dataset_name,
            success,
            skipped,
            failed,
            dataset_time,
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "scene", "status", "seconds", "message"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    logging.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
