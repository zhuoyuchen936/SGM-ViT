#!/usr/bin/env python3
"""Unified tuned-SGM re-run for all 4 datasets (Phase 7).

Per sample writes /tmp/sgm_tuned_all/<dataset>/<sample_name>.npz with:
  disp_raw        (float32) — raw SGM disp_L
  disp_filled     (float32) — hole-filled version (for compat/debug)
  hole_mask       (bool)     — occlusion | mismatch
  confidence      (float32) — raw PKRN * ~hole (unsmoothed)
  pkrn_raw        (float32)
  params          (float32[4]) — [range, P2, P1, Win]

Usage:
  python scripts/rerun_sgm_tuned.py --dataset sceneflow
  python scripts/rerun_sgm_tuned.py --dataset kitti
  python scripts/rerun_sgm_tuned.py --dataset eth3d --skip-existing
  python scripts/rerun_sgm_tuned.py --dataset middlebury --skip-existing
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.sgm_wrapper import run_sgm_with_confidence


# ----- Default params per dataset -----

DATASET_PARAMS = {
    "kitti":      {"range": 192, "P2": 3.0, "P1": 0.3, "Win": 5},
    "sceneflow":  {"range": 256, "P2": 3.0, "P1": 0.3, "Win": 5},
    "eth3d":      {"range":  80, "P2": 3.0, "P1": 0.3, "Win": 5},
    "middlebury": {"range": 192, "P2": 3.0, "P1": 0.3, "Win": 5},
    "monkaa":     {"range": 256, "P2": 3.0, "P1": 0.3, "Win": 5},
    "flyingthings": {"range": 256, "P2": 3.0, "P1": 0.3, "Win": 5},
}


# ----- Dataset enumeration -----

def enum_kitti(kitti_root: str):
    """Yield (sample_name, left_path, right_path) for KITTI-15 + KITTI-12."""
    from glob import glob as _glob
    # KITTI-15
    for left in sorted(_glob(os.path.join(kitti_root, "training", "image_2", "*_10.png"))):
        frame = Path(left).name  # e.g., 000002_10.png
        right = os.path.join(kitti_root, "training", "image_3", frame)
        if os.path.isfile(right):
            yield f"kitti15__{frame}", left, right
    # KITTI-12
    root_12 = os.path.join(kitti_root, "kitti2012")
    for left in sorted(_glob(os.path.join(root_12, "training", "colored_0", "*_10.png"))):
        frame = Path(left).name
        right = os.path.join(root_12, "training", "colored_1", frame)
        if os.path.isfile(right):
            yield f"kitti12__{frame}", left, right


def enum_sceneflow_driving(sf_root: str):
    """Yield (sample_name, left, right) for ALL 8 splits of SceneFlow driving."""
    from glob import glob as _glob
    for focal in ("15mm_focallength", "35mm_focallength"):
        for scene in ("scene_backwards", "scene_forwards"):
            for speed in ("fast", "slow"):
                left_dir = os.path.join(sf_root, "frames_cleanpass", focal, scene, speed, "left")
                right_dir = os.path.join(sf_root, "frames_cleanpass", focal, scene, speed, "right")
                if not os.path.isdir(left_dir):
                    continue
                for left in sorted(_glob(os.path.join(left_dir, "*.png"))):
                    frame = Path(left).stem
                    right = os.path.join(right_dir, f"{frame}.png")
                    if os.path.isfile(right):
                        yield f"{focal}__{scene}__{speed}__{frame}", left, right


def enum_eth3d(eth3d_root: str):
    from glob import glob as _glob
    for im0 in sorted(_glob(os.path.join(eth3d_root, "*/im0.png"))):
        scene_dir = Path(im0).parent
        scene = scene_dir.name
        if scene == "training":
            continue
        right = str(scene_dir / "im1.png")
        if os.path.isfile(right):
            yield scene, str(im0), right


def enum_monkaa(sf_root: str):
    """SceneFlow monkaa: /monkaa/frames_cleanpass/<scene>/left/*.png"""
    from glob import glob as _glob
    root = os.path.join(sf_root, "monkaa", "frames_cleanpass")
    for scene in sorted(os.listdir(root)):
        left_dir = os.path.join(root, scene, "left")
        right_dir = os.path.join(root, scene, "right")
        if not os.path.isdir(left_dir):
            continue
        for left in sorted(_glob(os.path.join(left_dir, "*.png"))):
            frame = Path(left).stem
            right = os.path.join(right_dir, f"{frame}.png")
            if os.path.isfile(right):
                yield f"monkaa__{scene}__{frame}", left, right


def enum_flyingthings(sf_root: str, subsample: int = 4, train_only: bool = True):
    """FlyingThings3D: TRAIN/{A,B,C}/<seq>/left/*.png. Subsample every Nth frame to keep size reasonable."""
    from glob import glob as _glob
    root = os.path.join(sf_root, "flyingthings3d", "frames_cleanpass")
    splits = ["TRAIN"] if train_only else ["TRAIN", "TEST"]
    idx = 0
    for split in splits:
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            continue
        for letter in sorted(os.listdir(split_root)):
            letter_root = os.path.join(split_root, letter)
            if not os.path.isdir(letter_root):
                continue
            for seq in sorted(os.listdir(letter_root)):
                left_dir = os.path.join(letter_root, seq, "left")
                right_dir = os.path.join(letter_root, seq, "right")
                if not os.path.isdir(left_dir):
                    continue
                for left in sorted(_glob(os.path.join(left_dir, "*.png"))):
                    if idx % subsample != 0:
                        idx += 1
                        continue
                    idx += 1
                    frame = Path(left).stem
                    right = os.path.join(right_dir, f"{frame}.png")
                    if os.path.isfile(right):
                        yield f"ft__{split}__{letter}__{seq}__{frame}", left, right


def enum_middlebury(mid_root: str):
    for scene_dir in sorted(Path(mid_root).iterdir()):
        if not scene_dir.is_dir():
            continue
        left = scene_dir / "im0.png"
        right = scene_dir / "im1.png"
        if left.is_file() and right.is_file():
            yield scene_dir.name, str(left), str(right)


# ----- Main loop -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_PARAMS.keys()))
    ap.add_argument("--out-root", default="/tmp/sgm_tuned_all")
    ap.add_argument("--kitti-root", default="/nfs/usrhome/pdongaa/dataczy/kitti")
    ap.add_argument("--sceneflow-root", default="/nfs/usrhome/pdongaa/dataczy/sceneflow_official/extracted/driving")
    ap.add_argument("--eth3d-root", default="/nfs/usrhome/pdongaa/dataczy/eth3d")
    ap.add_argument("--middlebury-root", default="/nfs/usrhome/pdongaa/dataczy/Middelburry/MiddEval3/trainingQ")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start-index", type=int, default=0, help="Process samples[start:end]")
    ap.add_argument("--end-index", type=int, default=-1, help="-1 means no end limit")
    ap.add_argument("--disparity-range", type=int, default=-1)
    ap.add_argument("--large-penalty", type=float, default=-1.0)
    ap.add_argument("--small-penalty", type=float, default=-1.0)
    ap.add_argument("--window-size", type=int, default=-1)
    args = ap.parse_args()

    p = DATASET_PARAMS[args.dataset].copy()
    if args.disparity_range > 0: p["range"] = args.disparity_range
    if args.large_penalty > 0: p["P2"] = args.large_penalty
    if args.small_penalty > 0: p["P1"] = args.small_penalty
    if args.window_size > 0: p["Win"] = args.window_size

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "kitti":
        samples = list(enum_kitti(args.kitti_root))
    elif args.dataset == "sceneflow":
        samples = list(enum_sceneflow_driving(args.sceneflow_root))
    elif args.dataset == "eth3d":
        samples = list(enum_eth3d(args.eth3d_root))
    elif args.dataset == "middlebury":
        samples = list(enum_middlebury(args.middlebury_root))
    elif args.dataset == "monkaa":
        samples = list(enum_monkaa(os.path.dirname(args.sceneflow_root)))  # parent of driving = sceneflow root
    elif args.dataset == "flyingthings":
        samples = list(enum_flyingthings(os.path.dirname(args.sceneflow_root), subsample=4, train_only=True))
    else:
        raise ValueError(args.dataset)

    if args.limit > 0:
        samples = samples[: args.limit]
    if args.start_index > 0 or args.end_index > 0:
        end = args.end_index if args.end_index > 0 else len(samples)
        samples = samples[args.start_index: end]
    print(f"[init] dataset={args.dataset}  n_samples={len(samples)}  params={p}  out_dir={out_dir}", flush=True)

    t_all = time.perf_counter()
    completed = 0
    skipped = 0
    for i, (name, left, right) in enumerate(samples):
        out_path = os.path.join(out_dir, f"{name}.npz")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        if not (os.path.isfile(left) and os.path.isfile(right)):
            print(f"[{i+1}/{len(samples)}] {name}: SKIP (missing input)", flush=True)
            continue

        t0 = time.perf_counter()
        try:
            _, _, disp_filled, debug = run_sgm_with_confidence(
                left, right,
                disparity_range=p["range"],
                LARGE_PENALTY=p["P2"],
                SMALL_PENALTY=p["P1"],
                Window_Size=p["Win"],
                smooth_sigma=0.0, verbose=False, return_debug=True,
            )
        except Exception as e:
            print(f"[{i+1}/{len(samples)}] {name}: ERROR {e}", flush=True)
            continue
        dt = time.perf_counter() - t0

        disp_raw = debug["disp_left"].astype(np.float32)
        hole_mask = (debug["occlusion"].astype(bool) | debug["mismatches"].astype(bool))
        conf = debug["confidence_raw"].astype(np.float32)
        pkrn = debug["pkrn_raw"].astype(np.float32)

        tmp = out_path + ".tmp.npz"
        np.savez_compressed(
            tmp.removesuffix(".npz"),
            disp_raw=disp_raw,
            disp_filled=disp_filled.astype(np.float32),
            hole_mask=hole_mask,
            confidence=conf,
            pkrn_raw=pkrn,
            params=np.array([p["range"], p["P2"], p["P1"], p["Win"]], dtype=np.float32),
        )
        os.replace(tmp, out_path)
        completed += 1

        if completed % 20 == 0 or i == len(samples) - 1:
            elapsed = time.perf_counter() - t_all
            rate = completed / max(elapsed, 1.0)
            remaining = (len(samples) - i - 1) / max(rate, 1e-6)
            hole_pct = hole_mask.mean() * 100
            print(
                f"[{i+1}/{len(samples)}] {name:50s} ({dt:4.1f}s) holes={hole_pct:4.1f}%  "
                f"done={completed} skip={skipped} elapsed={elapsed/60:.1f}m  eta={remaining/60:.1f}m",
                flush=True,
            )

    total = time.perf_counter() - t_all
    print(f"\n[done] dataset={args.dataset}  completed={completed}  skipped={skipped}  total={total/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
