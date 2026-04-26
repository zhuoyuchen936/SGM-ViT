#!/usr/bin/env python3
"""Phase 9 — Eval QAT (fake-quant INT8) EffViTDepthNet on 4 datasets.

Loads a QAT ckpt produced by train_qat_effvit.py, re-builds the model with
fake-quant wrappers, loads weights, freezes observers, runs 4-dataset eval.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import EffViTDepthNet, build_effvit_depth_inputs
from core.effvit_qat import prepare_qat_effvit, freeze_observers, count_quantized_bytes


DATASET_SPLITS = {"kitti": "val", "sceneflow": "test", "eth3d": "eval", "middlebury": "eval"}
DATASET_D1 = {"kitti": (3.0, "d1"), "sceneflow": (1.0, "bad1"), "eth3d": (1.0, "bad1"), "middlebury": (2.0, "bad2")}


def _pad_to_multiple(x, m=32):
    H, W = x.shape[-2:]
    nh = math.ceil(H / m) * m
    nw = math.ceil(W / m) * m
    pt = (nh - H) // 2; pb = nh - H - pt
    pl = (nw - W) // 2; pr = nw - W - pl
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return x, (0, 0, 0, 0)
    return F.pad(x, (pl, pr, pt, pb), value=0), (pt, pb, pl, pr)


def _unpad(x, pads):
    pt, pb, pl, pr = pads
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return x
    H, W = x.shape[-2:]
    return x[..., pt:H - pb if pb else H, pl:W - pr if pr else W]


def eval_dataset(model, device, cache_dir, dataset):
    tau, d1_key = DATASET_D1[dataset]
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    print(f"[{dataset}] {len(files)} samples", flush=True)
    agg = {"mono": {"epe": 0, d1_key: 0, "n": 0},
           "heuristic": {"epe": 0, d1_key: 0, "n": 0},
           "effvit_qat": {"epe": 0, d1_key: 0, "n": 0}}
    for i, path in enumerate(files):
        data = np.load(path)
        rgb_np = data["rgb"].astype(np.float32) / 255.0
        mono_np = data["mono_disp_aligned"].astype(np.float32)
        sgm_np = data["sgm_disp"].astype(np.float32)
        conf_np = data["confidence_map"].astype(np.float32)
        gt_np = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid_np = data["valid_mask"].astype(bool) & np.isfinite(data["gt_disp"])
        disp_scale = float(data["disp_scale"])
        heur_np = data["fused_base"].astype(np.float32) if "fused_base" in data.files else mono_np
        sgm_valid_np = data["sgm_valid"].astype(bool) if "sgm_valid" in data.files else (sgm_np > 0)

        rgb = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        mono = torch.from_numpy(mono_np).float().unsqueeze(0).unsqueeze(0).to(device)
        sgm = torch.from_numpy(sgm_np).float().unsqueeze(0).unsqueeze(0).to(device)
        conf = torch.from_numpy(conf_np).float().unsqueeze(0).unsqueeze(0).to(device)
        sgm_v = torch.from_numpy(sgm_valid_np).to(torch.bool).unsqueeze(0).unsqueeze(0).to(device)
        scale_t = torch.tensor([disp_scale]).float().to(device)
        rgb_p, pads = _pad_to_multiple(rgb)
        mono_p, _ = _pad_to_multiple(mono)
        sgm_p, _ = _pad_to_multiple(sgm)
        conf_p, _ = _pad_to_multiple(conf)
        sgmv_p, _ = _pad_to_multiple(sgm_v)
        with torch.no_grad():
            x = build_effvit_depth_inputs(rgb_p, mono_p, sgm_p, conf_p, scale_t, sgm_valid=sgmv_p)
            residual_n = model(x)
            pred_pad = mono_p + residual_n * scale_t.view(1, 1, 1, 1).clamp_min(1.0)
            pred = _unpad(pred_pad, pads)
        effvit_np = pred.squeeze(0).squeeze(0).cpu().float().numpy()

        for name, pn in (("mono", mono_np), ("heuristic", heur_np), ("effvit_qat", effvit_np)):
            err = np.abs(pn - gt_np)
            m = valid_np
            denom = max(m.sum(), 1)
            agg[name]["epe"] += float(err[m].mean())
            agg[name][d1_key] += float(((err > tau) & m).sum()) / denom * 100.0
            agg[name]["n"] += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]", flush=True)

    out = {"count": len(files), "d1_key": d1_key, "tau": tau, "methods": {}}
    for name, b in agg.items():
        n = b["n"]
        out["methods"][name] = {"epe": b["epe"] / max(n, 1), d1_key: b[d1_key] / max(n, 1)}
        print(f"  {name:15s}: epe={out['methods'][name]['epe']:.4f} {d1_key}={out['methods'][name][d1_key]:.2f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat-ckpt", required=True)
    ap.add_argument("--cache-root", default="artifacts/fusion_cache_v3")
    ap.add_argument("--out-dir", default="results/eval_phase9_qat")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.qat_ckpt, map_location="cpu")
    variant = state.get("variant", "b1")
    head_ch = state.get("head_ch", 48)
    in_channels = state.get("in_channels", 7)
    w_bits = state.get("w_bits", 8)
    a_bits = state.get("a_bits", 8)
    model = EffViTDepthNet(variant=variant, head_ch=head_ch, in_channels=in_channels).to(device)
    prepare_qat_effvit(model, w_bits=w_bits, a_bits=a_bits)
    model.load_state_dict(state["model"])
    freeze_observers(model)
    model.eval()
    int8_b, fp32_b = count_quantized_bytes(model)
    print(f"[ckpt] {args.qat_ckpt} variant={variant} head_ch={head_ch} w{w_bits}a{a_bits}", flush=True)
    print(f"[size] INT8 conv weights={int8_b/1024:.1f} KB vs FP32 conv weights={fp32_b/1024/1024:.2f} MB", flush=True)

    results = {"ckpt": args.qat_ckpt, "variant": variant, "w_bits": w_bits, "a_bits": a_bits,
               "int8_weight_bytes": int8_b, "fp32_weight_bytes": fp32_b,
               "per_dataset": {}}
    for dataset, split in DATASET_SPLITS.items():
        cache_dir = os.path.join(args.cache_root, dataset, split)
        if not os.path.isdir(cache_dir):
            continue
        results["per_dataset"][dataset] = eval_dataset(model, device, cache_dir, dataset)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n==== QAT Summary ====")
    for ds, r in results["per_dataset"].items():
        if "methods" not in r: continue
        d1k = r["d1_key"]
        m = r["methods"]
        print(f"{ds:12s} effvit_qat: EPE={m['effvit_qat']['epe']:.3f} {d1k}={m['effvit_qat'][d1k]:.2f}% | heur: EPE={m['heuristic']['epe']:.3f} {d1k}={m['heuristic'][d1k]:.2f}%", flush=True)
    print(f"\n[saved] {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
