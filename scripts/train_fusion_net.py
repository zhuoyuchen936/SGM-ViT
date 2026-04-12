#!/usr/bin/env python3
"""Train FusionNet-v1 from offline cache files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader


_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.fusion_net import (  # noqa: E402
    DEFAULT_ANCHOR_GATE_STRENGTH,
    DEFAULT_APPLY_ANCHOR_GATE,
    DEFAULT_LAMBDA_ANCHOR,
    DEFAULT_MAX_RESIDUAL_SCALE,
    FusionCacheDataset,
    FusionResidualNet,
    apply_anchor_gate_to_residual,
    compute_fusion_net_losses,
    copy_best_state_dict,
)


DEFAULT_CACHE_ROOT = os.path.join(_PROJECT_DIR, "artifacts", "fusion_cache")
DEFAULT_OUT_ROOT = os.path.join(_PROJECT_DIR, "artifacts", "fusion_net")


@dataclass
class StageConfig:
    name: str
    train_dir: str
    val_dir: str
    epochs: int
    batch_size: int
    crop_h: int
    crop_w: int
    lr: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FusionNet-v1 from offline cache.")
    parser.add_argument("--stage", default="both", choices=["sceneflow", "kitti", "both"])
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--resume", default=None, help="Resume checkpoint for a single-stage run.")
    parser.add_argument("--sceneflow-epochs", type=int, default=20)
    parser.add_argument("--kitti-epochs", type=int, default=10)
    parser.add_argument("--sceneflow-batch-size", type=int, default=8)
    parser.add_argument("--kitti-batch-size", type=int, default=4)
    parser.add_argument("--sceneflow-lr", type=float, default=1e-3)
    parser.add_argument("--kitti-lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-residual-scale", type=float, default=DEFAULT_MAX_RESIDUAL_SCALE)
    parser.add_argument("--lambda-anchor", type=float, default=DEFAULT_LAMBDA_ANCHOR)
    parser.add_argument("--anchor-gate-strength", type=float, default=DEFAULT_ANCHOR_GATE_STRENGTH)
    parser.add_argument("--apply-anchor-gate", dest="apply_anchor_gate", action="store_true")
    parser.add_argument("--no-apply-anchor-gate", dest="apply_anchor_gate", action="store_false")
    parser.set_defaults(apply_anchor_gate=DEFAULT_APPLY_ANCHOR_GATE)
    parser.add_argument("--cpu", action="store_true")
    return parser


def epe_from_batch(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = torch.abs(pred - gt) * valid
    return err.sum() / valid.sum().clamp_min(1.0)


def run_epoch(
    model: FusionResidualNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_anchor: float,
    apply_anchor_gate: bool,
    anchor_gate_strength: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    stats = {"loss": 0.0, "disp_loss": 0.0, "grad_loss": 0.0, "anchor_loss": 0.0, "epe": 0.0}
    batches = 0

    for batch in loader:
        x = batch["input"].to(device)
        gt = batch["gt_norm"].to(device)
        valid = batch["valid_mask"].to(device)
        fused_base = batch["fused_base_norm"].to(device)
        conf = batch["confidence_map"].to(device)
        detail = batch["detail_score"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        residual = model(x)
        residual = apply_anchor_gate_to_residual(
            residual=residual,
            confidence_map=conf,
            detail_score=detail,
            apply_anchor_gate=apply_anchor_gate,
            anchor_gate_strength=anchor_gate_strength,
        )
        pred = torch.clamp(fused_base + residual, min=0.0)
        losses = compute_fusion_net_losses(
            pred_norm=pred,
            gt_norm=gt,
            fused_base_norm=fused_base,
            valid_mask=valid,
            confidence_map=conf,
            detail_score=detail,
            lambda_anchor=lambda_anchor,
        )
        if training:
            losses["loss"].backward()
            optimizer.step()

        stats["loss"] += float(losses["loss"].detach().cpu())
        stats["disp_loss"] += float(losses["disp_loss"].cpu())
        stats["grad_loss"] += float(losses["grad_loss"].cpu())
        stats["anchor_loss"] += float(losses["anchor_loss"].cpu())
        stats["epe"] += float(epe_from_batch(pred, gt, valid).detach().cpu())
        batches += 1

    if batches == 0:
        return {key: float("nan") for key in stats}
    return {key: value / batches for key, value in stats.items()}


def train_stage(
    model: FusionResidualNet,
    stage: StageConfig,
    device: torch.device,
    num_workers: int,
    weight_decay: float,
    out_root: str,
    lambda_anchor: float,
    apply_anchor_gate: bool,
    anchor_gate_strength: float,
) -> str:
    os.makedirs(out_root, exist_ok=True)
    stage_dir = os.path.join(out_root, stage.name)
    os.makedirs(stage_dir, exist_ok=True)
    log_path = os.path.join(stage_dir, "train.log")
    metrics_jsonl_path = os.path.join(stage_dir, "metrics.jsonl")
    config_path = os.path.join(stage_dir, "config.json")

    def log_line(message: str) -> None:
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    train_ds = FusionCacheDataset(stage.train_dir, crop_hw=(stage.crop_h, stage.crop_w), training=True)
    val_ds = FusionCacheDataset(stage.val_dir, crop_hw=(stage.crop_h, stage.crop_w), training=False)
    if len(train_ds) == 0:
        raise RuntimeError(f"No cache samples found in {stage.train_dir}")
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation cache samples found in {stage.val_dir}")

    train_loader = DataLoader(train_ds, batch_size=stage.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=stage.batch_size, shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage.lr, weight_decay=weight_decay)

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_state = copy_best_state_dict(model)
    best_path = os.path.join(stage_dir, "best.pt")
    last_path = os.path.join(stage_dir, "last.pt")
    run_config = {
        "stage": asdict(stage),
        "weight_decay": weight_decay,
        "lambda_anchor": lambda_anchor,
        "apply_anchor_gate": apply_anchor_gate,
        "anchor_gate_strength": anchor_gate_strength,
        "max_residual_scale": model.max_residual_scale,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{stage.name}] training start\n")
        f.write(json.dumps(run_config, indent=2) + "\n")
    with open(metrics_jsonl_path, "w", encoding="utf-8") as f:
        f.write("")

    for epoch in range(1, stage.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lambda_anchor=lambda_anchor,
            apply_anchor_gate=apply_anchor_gate,
            anchor_gate_strength=anchor_gate_strength,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                device,
                lambda_anchor=lambda_anchor,
                apply_anchor_gate=apply_anchor_gate,
                anchor_gate_strength=anchor_gate_strength,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_disp_loss": train_stats["disp_loss"],
            "train_grad_loss": train_stats["grad_loss"],
            "train_anchor_loss": train_stats["anchor_loss"],
            "train_epe": train_stats["epe"],
            "val_loss": val_stats["loss"],
            "val_disp_loss": val_stats["disp_loss"],
            "val_grad_loss": val_stats["grad_loss"],
            "val_anchor_loss": val_stats["anchor_loss"],
            "val_epe": val_stats["epe"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "stage": stage.name,
        }
        history.append(row)
        log_line(
            f"[{stage.name}] epoch {epoch:02d}/{stage.epochs:02d} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"train_epe={train_stats['epe']:.4f} val_epe={val_stats['epe']:.4f} "
            f"train_anchor={train_stats['anchor_loss']:.4f} val_anchor={val_stats['anchor_loss']:.4f}"
        )
        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "stage": asdict(stage),
            "history": history,
            "epoch": epoch,
            "fusion_net_config": {
                "max_residual_scale": model.max_residual_scale,
                "lambda_anchor": lambda_anchor,
                "apply_anchor_gate": apply_anchor_gate,
                "anchor_gate_strength": anchor_gate_strength,
            },
        }
        torch.save(checkpoint, last_path)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = copy_best_state_dict(model)
            torch.save({**checkpoint, "model": best_state, "best_val_loss": best_val}, best_path)
            log_line(f"[{stage.name}] best checkpoint updated -> {best_path}")

    model.load_state_dict(best_state)
    history_path = os.path.join(stage_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": asdict(stage),
                "history": history,
                "best_val_loss": best_val,
                "fusion_net_config": {
                    "max_residual_scale": model.max_residual_scale,
                    "lambda_anchor": lambda_anchor,
                    "apply_anchor_gate": apply_anchor_gate,
                    "anchor_gate_strength": anchor_gate_strength,
                },
            },
            f,
            indent=2,
        )
    log_line(f"[{stage.name}] best checkpoint -> {best_path}")
    return best_path


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_root, exist_ok=True)

    model = FusionResidualNet(max_residual_scale=args.max_residual_scale).to(device)
    resume_path = args.resume
    if args.stage == "kitti" and resume_path is None:
        candidate = os.path.join(args.out_root, "sceneflow_pretrain", "best.pt")
        if os.path.isfile(candidate):
            resume_path = candidate

    if resume_path:
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print(f"Loaded resume checkpoint: {resume_path}")

    sceneflow_stage = StageConfig(
        name="sceneflow_pretrain",
        train_dir=os.path.join(args.cache_root, "sceneflow", "train"),
        val_dir=os.path.join(args.cache_root, "sceneflow", "val"),
        epochs=args.sceneflow_epochs,
        batch_size=args.sceneflow_batch_size,
        crop_h=384,
        crop_w=768,
        lr=args.sceneflow_lr,
    )
    kitti_stage = StageConfig(
        name="kitti_finetune",
        train_dir=os.path.join(args.cache_root, "kitti", "train"),
        val_dir=os.path.join(args.cache_root, "kitti", "val"),
        epochs=args.kitti_epochs,
        batch_size=args.kitti_batch_size,
        crop_h=320,
        crop_w=960,
        lr=args.kitti_lr,
    )

    if args.stage in ("sceneflow", "both"):
        best_pretrain = train_stage(
            model=model,
            stage=sceneflow_stage,
            device=device,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
            out_root=args.out_root,
            lambda_anchor=args.lambda_anchor,
            apply_anchor_gate=args.apply_anchor_gate,
            anchor_gate_strength=args.anchor_gate_strength,
        )
        if args.stage == "both":
            state = torch.load(best_pretrain, map_location="cpu")
            model.load_state_dict(state["model"])

    if args.stage in ("kitti", "both"):
        train_stage(
            model=model,
            stage=kitti_stage,
            device=device,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
            out_root=args.out_root,
            lambda_anchor=args.lambda_anchor,
            apply_anchor_gate=args.apply_anchor_gate,
            anchor_gate_strength=args.anchor_gate_strength,
        )


if __name__ == "__main__":
    main()
