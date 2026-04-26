#!/usr/bin/env python3
"""Phase 6 Iter 1: Train EffViTDepthNet on existing fusion_cache (v1).

Usage (SceneFlow pretrain):
  python scripts/train_effvit.py --stage sf_pretrain \
      --cache-root artifacts/fusion_cache --dataset sceneflow \
      --train-split train --val-split val \
      --epochs 20 --lr 3e-4 --batch-size 8 \
      --out-dir artifacts/fusion_phase6_iter1/sf_pretrain

Usage (KITTI finetune, resuming from SF ckpt):
  python scripts/train_effvit.py --stage kitti_finetune \
      --cache-root artifacts/fusion_cache --dataset kitti \
      --train-split train --val-split val \
      --epochs 15 --lr 5e-5 --batch-size 4 \
      --init-ckpt artifacts/fusion_phase6_iter1/sf_pretrain/best.pt \
      --out-dir artifacts/fusion_phase6_iter1/kitti_finetune

Smoke test:
  python scripts/train_effvit.py --smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import (
    EffViTDepthNet,
    TAU_BY_DATASET,
    build_effvit_depth_inputs,
    compute_effvit_losses,
    count_parameters,
)


# --------------------------- Dataset --------------------------- #


class EffViTCacheDataset(Dataset):
    """Reads minimal fields from v1 fusion_cache NPZ files.

    Returns per sample:
      - rgb:   (3, H, W) float [0, 1]
      - mono:  (1, H, W) float (absolute pixels)
      - sgm:   (1, H, W) float
      - conf:  (1, H, W) float [0, 1]
      - gt:    (1, H, W) float
      - valid: (1, H, W) bool -> float32
      - disp_scale: scalar
      - dataset: string (for tau lookup)
    """

    def __init__(
        self,
        cache_dir: str,
        dataset: str,
        crop_hw: tuple[int, int] | None = None,
        training: bool = False,
        hflip: bool = True,
        disp_jitter: tuple[float, float] | None = (0.9, 1.1),
        hole_aug_prob: float = 0.0,
        hole_aug_max_frac: float = 0.3,
    ) -> None:
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.crop_hw = crop_hw
        self.training = training
        self.hflip = hflip and training
        self.disp_jitter = disp_jitter if training else None
        self.hole_aug_prob = hole_aug_prob if training else 0.0
        self.hole_aug_max_frac = hole_aug_max_frac
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        data = np.load(path)
        rgb = data["rgb"].astype(np.float32) / 255.0  # HWC
        mono = data["mono_disp_aligned"].astype(np.float32)
        sgm = data["sgm_disp"].astype(np.float32)
        conf = data["confidence_map"].astype(np.float32)
        gt = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid = data["valid_mask"].astype(np.bool_)
        disp_scale = float(data["disp_scale"])
        # Phase 7: explicit sgm_valid field. Fall back to (sgm > 0) for v1/v2 caches.
        if "sgm_valid" in data.files:
            sgm_valid = data["sgm_valid"].astype(np.bool_)
        else:
            sgm_valid = (sgm > 0)

        # Optional disparity scale jitter (synchronous across mono/sgm/gt).
        if self.disp_jitter is not None:
            jit = random.uniform(*self.disp_jitter)
            mono = mono * jit
            sgm = sgm * jit
            gt = gt * jit
            disp_scale = disp_scale * jit

        H, W = gt.shape
        if self.crop_hw is not None:
            ch, cw = self.crop_hw
            if self.training:
                top = random.randint(0, max(0, H - ch))
                left = random.randint(0, max(0, W - cw))
            else:
                top = max(0, (H - ch) // 2)
                left = max(0, (W - cw) // 2)
            sl = slice(top, top + ch)
            sc = slice(left, left + cw)
            rgb = rgb[sl, sc]
            mono = mono[sl, sc]
            sgm = sgm[sl, sc]
            conf = conf[sl, sc]
            gt = gt[sl, sc]
            valid = valid[sl, sc]
            sgm_valid = sgm_valid[sl, sc]

        if self.hflip and random.random() < 0.5:
            rgb = rgb[:, ::-1].copy()
            mono = mono[:, ::-1].copy()
            sgm = sgm[:, ::-1].copy()
            conf = conf[:, ::-1].copy()
            gt = gt[:, ::-1].copy()
            valid = valid[:, ::-1].copy()
            sgm_valid = sgm_valid[:, ::-1].copy()

        # Phase 7 hole-aug: drop random rectangular blocks from SGM (set to 0) to make
        # the model learn to handle "sgm has holes" distribution. Also updates sgm_valid/conf.
        if self.hole_aug_prob > 0 and random.random() < self.hole_aug_prob:
            Hc, Wc = sgm.shape
            target_frac = random.uniform(0.05, self.hole_aug_max_frac)
            # Drop 3-8 random rectangles to reach target_frac.
            n_rect = random.randint(3, 8)
            per_area = target_frac / n_rect
            for _ in range(n_rect):
                rh = int(np.sqrt(per_area) * Hc * random.uniform(0.6, 1.6))
                rw = int(np.sqrt(per_area) * Wc * random.uniform(0.6, 1.6))
                rh = min(max(rh, 8), Hc)
                rw = min(max(rw, 8), Wc)
                top = random.randint(0, Hc - rh)
                left = random.randint(0, Wc - rw)
                sgm[top:top+rh, left:left+rw] = 0.0
                conf[top:top+rh, left:left+rw] = 0.0
                sgm_valid[top:top+rh, left:left+rw] = False

        return {
            "sgm_valid": torch.from_numpy(sgm_valid[None, :, :]).to(torch.bool),
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
            "mono": torch.from_numpy(mono[None, :, :]).float(),
            "sgm": torch.from_numpy(sgm[None, :, :]).float(),
            "conf": torch.from_numpy(conf[None, :, :]).float(),
            "gt": torch.from_numpy(gt[None, :, :]).float(),
            "valid": torch.from_numpy(valid[None, :, :]).to(torch.bool),
            "disp_scale": torch.tensor(disp_scale).float(),
            "dataset": self.dataset,
        }


def pad_to_multiple(tensors: dict, multiple: int = 32) -> tuple[dict, tuple[int, int, int, int]]:
    """Pad dict of tensors so H, W are multiples of `multiple`. Returns dict + (pad_t, pad_b, pad_l, pad_r)."""
    H, W = tensors["rgb"].shape[-2:]
    new_h = math.ceil(H / multiple) * multiple
    new_w = math.ceil(W / multiple) * multiple
    pad_t = (new_h - H) // 2
    pad_b = new_h - H - pad_t
    pad_l = (new_w - W) // 2
    pad_r = new_w - W - pad_l
    if (pad_t, pad_b, pad_l, pad_r) == (0, 0, 0, 0):
        return tensors, (0, 0, 0, 0)
    out = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.dim() < 2:
            out[k] = v
            continue
        out[k] = F.pad(v, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)
    return out, (pad_t, pad_b, pad_l, pad_r)


def unpad(t: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
    pt, pb, pl, pr = pads
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return t
    H, W = t.shape[-2:]
    return t[..., pt:H - pb if pb else H, pl:W - pr if pr else W]


# --------------------------- Train / eval loops --------------------------- #


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_d1, scheduler=None, log_every=20):
    model.train()
    total, n = 0.0, 0
    epe_sum, d1_sum = 0.0, 0.0
    t0 = time.time()
    for it, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        mono = batch["mono"].to(device, non_blocking=True)
        sgm = batch["sgm"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        sgm_valid_b = batch["sgm_valid"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        disp_scale = batch["disp_scale"].to(device, non_blocking=True)
        # Per-sample tau (so mixed batches use correct per-sample threshold).
        ds_field = batch["dataset"]
        if isinstance(ds_field, str):
            tau = TAU_BY_DATASET.get(ds_field, 3.0)
        else:
            tau = torch.tensor([TAU_BY_DATASET.get(n, 3.0) for n in ds_field], device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = build_effvit_depth_inputs(rgb, mono, sgm, conf, disp_scale, sgm_valid=sgm_valid_b)
            residual_n = model(x)
            B = rgb.shape[0]
            scale = disp_scale.view(B, 1, 1, 1).clamp_min(1.0)
            pred = mono + residual_n * scale
            losses = compute_effvit_losses(pred, gt, valid, tau=tau, lambda_d1=lambda_d1)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total += losses["total"].item() * rgb.size(0)
        epe_sum += losses["epe"].item() * rgb.size(0)
        d1_sum += losses["soft_d1"].item() * rgb.size(0)
        n += rgb.size(0)

        if log_every and (it + 1) % log_every == 0:
            print(
                f"  iter {it+1}/{len(loader)}  total={total/n:.4f}  epe={epe_sum/n:.4f}  soft_d1={d1_sum/n:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  t={time.time()-t0:.1f}s",
                flush=True,
            )
    return {"total": total / max(n, 1), "epe": epe_sum / max(n, 1), "soft_d1": d1_sum / max(n, 1)}


@torch.no_grad()
def validate(model, loader, device, dataset_name: str) -> dict:
    model.eval()
    total_epe, total_d1, total_cnt = 0.0, 0.0, 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        mono = batch["mono"].to(device)
        sgm = batch["sgm"].to(device)
        conf = batch["conf"].to(device)
        sgm_valid_b = batch["sgm_valid"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid"].to(device)
        disp_scale = batch["disp_scale"].to(device)

        # Pad to multiple of 32 (5 downsample stages).
        padded, pads = pad_to_multiple({"rgb": rgb, "mono": mono, "sgm": sgm, "conf": conf, "sgm_valid": sgm_valid_b}, 32)
        x = build_effvit_depth_inputs(padded["rgb"], padded["mono"], padded["sgm"], padded["conf"], disp_scale, sgm_valid=padded["sgm_valid"])
        residual_n = model(x)
        B = rgb.shape[0]
        scale = disp_scale.view(B, 1, 1, 1).clamp_min(1.0)
        pred_pad = padded["mono"] + residual_n * scale
        pred = unpad(pred_pad, pads)

        err = (pred - gt).abs()
        m = valid.float()
        denom = m.sum().clamp_min(1.0)
        epe = (err * m).sum() / denom
        # Exact D1 using the dataset's tau (hard threshold for eval).
        tau = TAU_BY_DATASET.get(dataset_name, 3.0)
        d1 = ((err > tau).float() * m).sum() / denom * 100.0
        total_epe += epe.item() * rgb.size(0)
        total_d1 += d1.item() * rgb.size(0)
        total_cnt += rgb.size(0)
    return {"epe": total_epe / max(total_cnt, 1), "d1": total_d1 / max(total_cnt, 1)}


# --------------------------- Main --------------------------- #


def _get_dataset_crop(dataset: str) -> tuple[int, int]:
    if dataset == "kitti":
        return (320, 832)
    if dataset == "sceneflow":
        return (384, 768)
    if dataset == "eth3d":
        return (384, 512)
    return (384, 576)  # middlebury


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sf_pretrain", choices=["sf_pretrain", "kitti_finetune", "mixed_finetune"])
    ap.add_argument("--cache-root", default="artifacts/fusion_cache")
    ap.add_argument("--dataset", default="sceneflow")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    # Mixed-dataset training (used when --stage=mixed_finetune).
    ap.add_argument("--mixed-datasets", default="", help="Comma-separated dataset names, e.g. 'kitti,sceneflow'")
    ap.add_argument("--mixed-weights", default="", help="Comma-separated weights (same len as --mixed-datasets)")
    ap.add_argument("--mixed-val-dataset", default="kitti", help="Dataset used for val in mixed mode")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-d1", type=float, default=0.2)
    ap.add_argument("--variant", default="b1", choices=["b0", "b1", "b2"])
    ap.add_argument("--head-ch", type=int, default=48, help="FPN head channel width (Phase 8 Pareto knob)")
    ap.add_argument("--in-channels", type=int, default=7, help="Input channel count; 7 is Phase 8 default (no sgm_pos)")
    ap.add_argument("--hole-aug-prob", type=float, default=0.5, help="Probability of applying hole augmentation to SGM during training")
    ap.add_argument("--hole-aug-max-frac", type=float, default=0.3, help="Max fraction of pixels zeroed by hole-aug")
    ap.add_argument("--out-dir", required=False, default="artifacts/fusion_phase6_iter1/default")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Run a short smoke test (few iters)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage == "mixed_finetune" and args.mixed_datasets:
        # Mixed-dataset training: concat + WeightedRandomSampler.
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        dsnames = [s.strip() for s in args.mixed_datasets.split(",") if s.strip()]
        weights_cfg = [float(s) for s in args.mixed_weights.split(",")] if args.mixed_weights else [1.0] * len(dsnames)
        assert len(weights_cfg) == len(dsnames), "--mixed-weights length must match --mixed-datasets"
        parts = []
        sample_w: list[float] = []
        # Use the largest crop size among selected datasets (clipped by smallest dim).
        crop_hw = _get_dataset_crop(dsnames[0])
        for ds_name, w in zip(dsnames, weights_cfg):
            tr_dir = os.path.join(args.cache_root, ds_name, "train" if ds_name in ("kitti", "sceneflow") else "eval")
            ds_part = EffViTCacheDataset(tr_dir, ds_name, crop_hw=crop_hw, training=True, hole_aug_prob=args.hole_aug_prob, hole_aug_max_frac=args.hole_aug_max_frac)
            if len(ds_part) == 0:
                print(f"[warn] mixed part {ds_name} has 0 samples at {tr_dir}")
                continue
            parts.append(ds_part)
            sample_w.extend([w / len(ds_part)] * len(ds_part))
        train_ds = ConcatDataset(parts)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=sum(len(p) for p in parts), replacement=True)
        val_dir = os.path.join(args.cache_root, args.mixed_val_dataset, args.val_split)
        val_ds = EffViTCacheDataset(val_dir, args.mixed_val_dataset, crop_hw=None, training=False)
        print(f"[data] mixed train={len(train_ds)} ({', '.join(f'{n}:{len(p)}' for n,p in zip(dsnames, parts))})  val={args.mixed_val_dataset}={len(val_ds)}", flush=True)
        val_dataset_name = args.mixed_val_dataset
    else:
        train_dir = os.path.join(args.cache_root, args.dataset, args.train_split)
        val_dir = os.path.join(args.cache_root, args.dataset, args.val_split)
        crop_hw = _get_dataset_crop(args.dataset)
        train_ds = EffViTCacheDataset(train_dir, args.dataset, crop_hw=crop_hw, training=True, hole_aug_prob=args.hole_aug_prob, hole_aug_max_frac=args.hole_aug_max_frac)
        val_ds = EffViTCacheDataset(val_dir, args.dataset, crop_hw=None, training=False)
        sampler = None
        val_dataset_name = args.dataset
        print(f"[data] train={len(train_ds)}  val={len(val_ds)}  crop={crop_hw}", flush=True)
    assert len(train_ds) > 0, "No train samples"

    def _collate(batch):
        return {
            "rgb": torch.stack([b["rgb"] for b in batch]),
            "mono": torch.stack([b["mono"] for b in batch]),
            "sgm": torch.stack([b["sgm"] for b in batch]),
            "conf": torch.stack([b["conf"] for b in batch]),
            "sgm_valid": torch.stack([b["sgm_valid"] for b in batch]),
            "gt": torch.stack([b["gt"] for b in batch]),
            "valid": torch.stack([b["valid"] for b in batch]),
            "disp_scale": torch.stack([b["disp_scale"] for b in batch]),
            # Per-sample dataset names (list). The train loop builds per-sample tau from this.
            "dataset": [b["dataset"] for b in batch],
        }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, collate_fn=_collate,
    )

    model = EffViTDepthNet(variant=args.variant, head_ch=args.head_ch, in_channels=args.in_channels).to(device)
    print(f"[model] EffViTDepthNet({args.variant}) params={count_parameters(model)/1e6:.2f}M", flush=True)

    if args.init_ckpt and os.path.isfile(args.init_ckpt):
        state = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"[init] loaded {args.init_ckpt}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(len(train_loader), 1)
    if args.smoke:
        total_steps = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr_min,
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_epe = float("inf")
    history = []

    if args.smoke:
        print("[smoke] running 1 batch forward+backward...", flush=True)
        batch = next(iter(train_loader))
        rgb = batch["rgb"].to(device)
        mono = batch["mono"].to(device)
        sgm = batch["sgm"].to(device)
        conf = batch["conf"].to(device)
        sgm_valid_b = batch["sgm_valid"].to(device)
        gt = batch["gt"].to(device)
        valid = batch["valid"].to(device)
        disp_scale = batch["disp_scale"].to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = build_effvit_depth_inputs(rgb, mono, sgm, conf, disp_scale, sgm_valid=sgm_valid_b)
            residual_n = model(x)
            B = rgb.shape[0]
            scale = disp_scale.view(B, 1, 1, 1).clamp_min(1.0)
            pred = mono + residual_n * scale
            losses = compute_effvit_losses(pred, gt, valid, tau=3.0, lambda_d1=args.lambda_d1)
        print(f"[smoke] loss={losses['total'].item():.4f} epe={losses['epe'].item():.4f} d1={losses['soft_d1'].item():.4f}", flush=True)
        scaler.scale(losses["total"]).backward()
        print("[smoke] backward OK", flush=True)
        return

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===", flush=True)
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, args.lambda_d1, scheduler=scheduler)
        val_metrics = validate(model, val_loader, device, val_dataset_name) if len(val_ds) > 0 else {"epe": -1, "d1": -1}
        row = {"epoch": epoch, "train": tr, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"[epoch {epoch}] train_total={tr['total']:.4f}  val_epe={val_metrics['epe']:.4f}  val_d1={val_metrics['d1']:.2f}%", flush=True)
        ckpt_meta = {
            "model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics,
            "args": vars(args), "variant": args.variant,
            "head_ch": args.head_ch, "in_channels": args.in_channels,
        }
        if val_metrics["epe"] < best_val_epe:
            best_val_epe = val_metrics["epe"]
            torch.save(ckpt_meta, os.path.join(args.out_dir, "best.pt"))
            print(f"[best] saved best.pt at epoch {epoch} (val_epe={best_val_epe:.4f})", flush=True)
        # Always save last checkpoint
        torch.save(ckpt_meta, os.path.join(args.out_dir, "last.pt"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump({"best_val_epe": best_val_epe, "history": history}, f, indent=2)

    print(f"\n[done] best_val_epe={best_val_epe:.4f}  out={args.out_dir}/best.pt", flush=True)


if __name__ == "__main__":
    main()
