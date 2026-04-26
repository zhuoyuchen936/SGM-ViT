#!/usr/bin/env python3
"""Phase 9 — INT8 QAT fine-tune for EffViTDepthNet.

Loads an FP32 ckpt, inserts fake-quant on Conv2d/Linear (outside LiteMLA),
calibrates activations on a few batches, then fine-tunes for ~5 epochs with
straight-through gradients. Final ckpt is "INT8 simulated" (fake-quant'd).

Usage:
  python scripts/train_qat_effvit.py \
    --fp32-ckpt artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt \
    --cache-root artifacts/fusion_cache_v3 \
    --out-dir artifacts/fusion_phase9_qat/b1_h24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import (
    EffViTDepthNet,
    build_effvit_depth_inputs,
    compute_effvit_losses,
    TAU_BY_DATASET,
)
from core.effvit_qat import (
    prepare_qat_effvit,
    calibrate_activations,
    freeze_observers,
    count_wrapped_layers,
    count_quantized_bytes,
)
from scripts.train_effvit import EffViTCacheDataset, _get_dataset_crop, pad_to_multiple, unpad


def collate(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "mono": torch.stack([b["mono"] for b in batch]),
        "sgm": torch.stack([b["sgm"] for b in batch]),
        "conf": torch.stack([b["conf"] for b in batch]),
        "sgm_valid": torch.stack([b["sgm_valid"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "disp_scale": torch.stack([b["disp_scale"] for b in batch]),
        "dataset": [b["dataset"] for b in batch],
    }


@torch.no_grad()
def quick_val(model, loader, device, dataset_name: str) -> dict:
    model.eval()
    total_epe, total_cnt = 0.0, 0
    total_d1 = 0.0
    tau = TAU_BY_DATASET.get(dataset_name, 3.0)
    for batch in loader:
        rgb = batch["rgb"].to(device); mono = batch["mono"].to(device)
        sgm = batch["sgm"].to(device); conf = batch["conf"].to(device)
        sgm_v = batch["sgm_valid"].to(device); gt = batch["gt"].to(device)
        valid = batch["valid"].to(device); ds_scale = batch["disp_scale"].to(device)
        padded, pads = pad_to_multiple({"rgb": rgb, "mono": mono, "sgm": sgm, "conf": conf, "sgm_valid": sgm_v}, 32)
        x = build_effvit_depth_inputs(padded["rgb"], padded["mono"], padded["sgm"], padded["conf"], ds_scale, sgm_valid=padded["sgm_valid"])
        residual_n = model(x)
        B = rgb.shape[0]
        scale = ds_scale.view(B, 1, 1, 1).clamp_min(1.0)
        pred_pad = padded["mono"] + residual_n * scale
        pred = unpad(pred_pad, pads)
        err = (pred - gt).abs()
        m = valid.float(); denom = m.sum().clamp_min(1.0)
        total_epe += ((err * m).sum() / denom).item() * B
        total_d1 += (((err > tau).float() * m).sum() / denom).item() * 100.0 * B
        total_cnt += B
    return {"epe": total_epe / max(total_cnt, 1), "d1": total_d1 / max(total_cnt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-ckpt", required=True)
    ap.add_argument("--cache-root", default="artifacts/fusion_cache_v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib-batches", type=int, default=20)
    ap.add_argument("--hole-aug-prob", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load FP32.
    state = torch.load(args.fp32_ckpt, map_location="cpu")
    variant = state.get("variant", "b1")
    head_ch = state.get("head_ch", 48)
    in_channels = state.get("in_channels", 7)
    model = EffViTDepthNet(variant=variant, head_ch=head_ch, in_channels=in_channels).to(device)
    model.load_state_dict(state["model"])
    print(f"[init] FP32 loaded: variant={variant} head_ch={head_ch} in_ch={in_channels}", flush=True)

    # 2) Insert fake-quant wrappers.
    prepare_qat_effvit(model, w_bits=args.w_bits, a_bits=args.a_bits)
    stats = count_wrapped_layers(model)
    int8_b, fp32_b = count_quantized_bytes(model)
    print(f"[qat] wrapped: {stats}", flush=True)
    print(f"[qat] quantizable weights: INT8={int8_b/1024:.1f} KB vs FP32={fp32_b/1024/1024:.2f} MB", flush=True)

    # 3) Data loaders. Use mixed KITTI+SF like Phase 7 finetune.
    kitti_tr = EffViTCacheDataset(os.path.join(args.cache_root, "kitti", "train"), "kitti",
                                  crop_hw=_get_dataset_crop("kitti"), training=True,
                                  hole_aug_prob=args.hole_aug_prob, hole_aug_max_frac=0.3)
    sf_tr = EffViTCacheDataset(os.path.join(args.cache_root, "sceneflow", "train"), "sceneflow",
                               crop_hw=_get_dataset_crop("kitti"), training=True,
                               hole_aug_prob=args.hole_aug_prob, hole_aug_max_frac=0.3)
    train_ds = ConcatDataset([kitti_tr, sf_tr])
    weights = [1.0 / len(kitti_tr)] * len(kitti_tr) + [1.0 / len(sf_tr)] * len(sf_tr)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=collate)
    val_ds = EffViTCacheDataset(os.path.join(args.cache_root, "kitti", "val"), "kitti",
                                crop_hw=None, training=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate)
    print(f"[data] train: kitti={len(kitti_tr)} sf={len(sf_tr)}  val: kitti={len(val_ds)}", flush=True)

    # 4) Calibrate activations (forward-only) to set a_min/a_max.
    print(f"[calib] running {args.calib_batches} forward batches to warm observers ...", flush=True)
    calib_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    calibrate_activations(model, calib_loader, device, n_batches=args.calib_batches)
    print(f"[calib] done", flush=True)

    # 5) QAT finetune.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10)
    scaler = torch.cuda.amp.GradScaler()

    best_val_epe = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== QAT Epoch {epoch}/{args.epochs} ===", flush=True)
        model.train()
        total_loss = 0.0; n_seen = 0; t0 = time.time()
        for it, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            mono = batch["mono"].to(device, non_blocking=True)
            sgm = batch["sgm"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            sgm_v = batch["sgm_valid"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            ds_scale = batch["disp_scale"].to(device, non_blocking=True)
            tau = torch.tensor([TAU_BY_DATASET.get(n, 3.0) for n in batch["dataset"]], device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                x = build_effvit_depth_inputs(rgb, mono, sgm, conf, ds_scale, sgm_valid=sgm_v)
                residual_n = model(x)
                B = rgb.shape[0]
                scale = ds_scale.view(B, 1, 1, 1).clamp_min(1.0)
                pred = mono + residual_n * scale
                losses = compute_effvit_losses(pred, gt, valid, tau=tau, lambda_d1=0.2)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += losses["total"].item() * B
            n_seen += B
            if (it + 1) % 50 == 0:
                print(f"  iter {it+1}/{len(train_loader)}  loss={total_loss/max(n_seen,1):.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  t={time.time()-t0:.0f}s", flush=True)

        val_m = quick_val(model, val_loader, device, "kitti")
        row = {"epoch": epoch, "train_loss": total_loss / max(n_seen, 1), "val": val_m}
        history.append(row)
        print(f"[epoch {epoch}] train_loss={row['train_loss']:.4f}  val_epe={val_m['epe']:.4f}  val_d1={val_m['d1']:.2f}%", flush=True)
        save = {
            "model": model.state_dict(), "epoch": epoch, "val_metrics": val_m,
            "variant": variant, "head_ch": head_ch, "in_channels": in_channels,
            "qat": True, "w_bits": args.w_bits, "a_bits": args.a_bits,
        }
        torch.save(save, os.path.join(args.out_dir, "last.pt"))
        if val_m["epe"] < best_val_epe:
            best_val_epe = val_m["epe"]
            torch.save(save, os.path.join(args.out_dir, "best.pt"))
            print(f"[best] saved at epoch {epoch} (val_epe={best_val_epe:.4f})", flush=True)

    # 6) Final freeze + save INT8 summary.
    freeze_observers(model)
    torch.save({"model": model.state_dict(), "variant": variant, "head_ch": head_ch, "in_channels": in_channels,
                "qat": True, "w_bits": args.w_bits, "a_bits": args.a_bits, "frozen": True},
               os.path.join(args.out_dir, "int8_frozen.pt"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump({"best_val_epe": best_val_epe, "history": history,
                   "int8_bytes": int8_b, "fp32_bytes": fp32_b,
                   "wrapping": stats}, f, indent=2)
    print(f"\n[done] best val_epe={best_val_epe:.4f}  out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
