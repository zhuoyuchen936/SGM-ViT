#!/usr/bin/env python3
"""Evaluate MobileStereoNet (2D or 3D) on same 4-dataset val splits as eval_phase96.
Reconstructs stereo pairs from fusion_cache_v3 sample_name -> original dataset paths.
Writes summary.json in the same shape as eval_effvit.py (methods: mono, heuristic, <model_tag>).
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# MobileStereoNet repo
MSNET_REPO = "/home/pdongaa/workspace/mobilestereonet"
sys.path.insert(0, MSNET_REPO)
from models import __models__  # noqa: E402


# Same splits used by eval_effvit.py (matches Phase 9.6 JSON counts)
DATASET_SPLITS = {
    "kitti": "val",
    "sceneflow": "test",
    "eth3d": "eval",
    "middlebury": "eval",
}
DATASET_D1 = {
    "kitti": (3.0, "d1"),
    "sceneflow": (1.0, "bad1"),
    "eth3d": (1.0, "bad1"),
    "middlebury": (2.0, "bad2"),
}

KITTI_ROOT = "/nfs/usrhome/pdongaa/dataczy/kitti"
SF_ROOT = "/nfs/usrhome/pdongaa/dataczy/sceneflow_official/extracted/driving"
ETH3D_ROOT = "/nfs/usrhome/pdongaa/dataczy/eth3d"
MID_ROOT = "/nfs/usrhome/pdongaa/dataczy/Middelburry/MiddEval3/trainingQ"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sample_to_paths(name: str, dataset: str):
    if dataset == "kitti":
        if name.startswith("kitti15__"):
            frame = name[len("kitti15__"):]
            return f"{KITTI_ROOT}/training/image_2/{frame}", f"{KITTI_ROOT}/training/image_3/{frame}"
        if name.startswith("kitti12__"):
            frame = name[len("kitti12__"):]
            return f"{KITTI_ROOT}/kitti2012/training/colored_0/{frame}", f"{KITTI_ROOT}/kitti2012/training/colored_1/{frame}"
        raise ValueError(f"unknown kitti sample: {name}")
    if dataset == "sceneflow":
        parts = name.split("__")
        if len(parts) != 4:
            raise ValueError(f"bad sf name: {name}")
        focal, scene, speed, frame = parts
        return (f"{SF_ROOT}/frames_cleanpass/{focal}/{scene}/{speed}/left/{frame}.png",
                f"{SF_ROOT}/frames_cleanpass/{focal}/{scene}/{speed}/right/{frame}.png")
    if dataset == "eth3d":
        return f"{ETH3D_ROOT}/{name}/im0.png", f"{ETH3D_ROOT}/{name}/im1.png"
    if dataset == "middlebury":
        return f"{MID_ROOT}/{name}/im0.png", f"{MID_ROOT}/{name}/im1.png"
    raise ValueError(dataset)


def _load_rgb_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t


def _pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    H, W = x.shape[-2:]
    new_h = math.ceil(H / multiple) * multiple
    new_w = math.ceil(W / multiple) * multiple
    pt = new_h - H
    pl = new_w - W
    if (pt, pl) == (0, 0):
        return x, (0, 0, 0, 0)
    return F.pad(x, (pl, 0, pt, 0), mode="replicate"), (pt, 0, pl, 0)


def _crop(x: torch.Tensor, pads):
    pt, pb, pl, pr = pads
    H, W = x.shape[-2:]
    return x[..., pt:, pl:]


def evaluate_dataset(model, device, cache_dir: str, dataset: str, model_tag: str, max_samples: int = 0) -> dict:
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if max_samples > 0:
        files = files[:max_samples]
    print(f"[{dataset}] evaluating {len(files)} samples", flush=True)
    if not files:
        return {"count": 0}
    tau, d1_key = DATASET_D1[dataset]
    agg = {model_tag: {"epe": 0.0, d1_key: 0.0, "n": 0}}
    n_bad_path = 0

    for i, f in enumerate(files):
        data = np.load(f)
        name = str(data["sample_name"])
        gt = data["gt_disp"].astype(np.float32)
        valid = data["valid_mask"].astype(bool)
        try:
            lp, rp = sample_to_paths(name, dataset)
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)
            n_bad_path += 1
            continue
        if not (os.path.isfile(lp) and os.path.isfile(rp)):
            print(f"  [skip] {name}: missing stereo pair", flush=True)
            n_bad_path += 1
            continue

        L = _load_rgb_tensor(lp).to(device)
        R = _load_rgb_tensor(rp).to(device)

        # Match GT resolution: cache stores pre-resized rgb/gt at shape (H, W).
        # MSNet expects L/R at the same resolution as GT.
        gt_h, gt_w = gt.shape
        if L.shape[-2:] != (gt_h, gt_w):
            L = F.interpolate(L, size=(gt_h, gt_w), mode="bilinear", align_corners=False)
            R = F.interpolate(R, size=(gt_h, gt_w), mode="bilinear", align_corners=False)

        Lp, pads = _pad_to_multiple(L, 32)
        Rp, _ = _pad_to_multiple(R, 32)

        with torch.no_grad():
            pred = model(Lp, Rp)
            # MSNet in eval mode may still return list/tuple of multi-scale outputs; take last (finest)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
        # pred is (B, H, W) or (B, 1, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = _crop(pred, pads)
        pred_np = pred.squeeze(0).squeeze(0).cpu().float().numpy()

        err = np.abs(pred_np - gt)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        epe = float(err[valid].mean())
        d1v = float(((err > tau) & valid).sum()) / n_valid * 100.0
        agg[model_tag]["epe"] += epe
        agg[model_tag][d1_key] += d1v
        agg[model_tag]["n"] += 1

        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {name:60s} epe={epe:.3f} {d1_key}={d1v:.2f}%", flush=True)

    out = {"count": len(files), "n_used": agg[model_tag]["n"], "n_skipped": n_bad_path,
           "d1_key": d1_key, "tau": tau, "methods": {}}
    n = max(agg[model_tag]["n"], 1)
    out["methods"][model_tag] = {
        "epe": agg[model_tag]["epe"] / n,
        d1_key: agg[model_tag][d1_key] / n,
    }
    print(f"  {model_tag:22s}: epe={out['methods'][model_tag]['epe']:.4f} {d1_key}={out['methods'][model_tag][d1_key]:.2f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["MSNet2D", "MSNet3D"], default="MSNet2D")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", default="/home/pdongaa/workspace/SGM-ViT/artifacts/fusion_cache_v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--maxdisp", type=int, default=192)
    ap.add_argument("--tag", default="", help="method tag in JSON (defaults to model name)")
    ap.add_argument("--smoke", type=int, default=0, help=">0: limit samples per dataset for smoke test")
    args = ap.parse_args()

    tag = args.tag or args.model
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    ModelCls = __models__[args.model]
    model = ModelCls(args.maxdisp)
    model = nn.DataParallel(model).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    # strip any double-nested 'module.' if present
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_bytes = os.path.getsize(args.ckpt)
    print(f"[ckpt] {args.ckpt}  model={args.model} maxdisp={args.maxdisp} params={n_params/1e6:.2f}M file={ckpt_bytes/1024/1024:.1f}MB", flush=True)

    results = {
        "ckpt": args.ckpt, "model": args.model, "tag": tag, "maxdisp": args.maxdisp,
        "n_params": n_params, "ckpt_bytes": ckpt_bytes, "per_dataset": {},
    }
    for ds, split in DATASET_SPLITS.items():
        cache_dir = os.path.join(args.cache_root, ds, split)
        if not os.path.isdir(cache_dir):
            print(f"[{ds}] skip (no cache at {cache_dir})", flush=True)
            continue
        results["per_dataset"][ds] = evaluate_dataset(model, device, cache_dir, ds, tag, max_samples=args.smoke)

    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_json}", flush=True)

    print(f"\n==== Summary ({tag}, ckpt={ckpt_bytes/1024/1024:.1f}MB) ====")
    for ds, res in results["per_dataset"].items():
        if not res or "methods" not in res:
            continue
        m = res["methods"][tag]
        d1k = res["d1_key"]
        print(f"  {ds:12s} n={res.get('n_used',0):4d}  epe={m['epe']:.3f}  {d1k}={m[d1k]:.2f}%  (skipped={res.get('n_skipped',0)})")


if __name__ == "__main__":
    main()
