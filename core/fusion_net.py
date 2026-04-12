"""Lightweight learned residual fusion for SGM-ViT."""
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fusion import fuse_edge_aware_residual


FUSION_INPUT_CHANNELS = (
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "mono_disp_aligned",
    "sgm_disp",
    "fused_base",
    "confidence_map",
    "sgm_valid",
    "disp_disagreement",
    "detail_score",
)

LEGACY_MAX_RESIDUAL_SCALE = 0.25
LEGACY_LAMBDA_ANCHOR = 0.05
DEFAULT_MAX_RESIDUAL_SCALE = 0.10
DEFAULT_LAMBDA_ANCHOR = 0.20
DEFAULT_APPLY_ANCHOR_GATE = True
DEFAULT_ANCHOR_GATE_STRENGTH = 1.0


@dataclass
class FusionInputBundle:
    input_tensor: torch.Tensor
    fused_base: np.ndarray
    fused_base_norm: np.ndarray
    disp_scale: float
    detail_score: np.ndarray
    confidence_map: np.ndarray
    sgm_valid: np.ndarray
    mono_disp_aligned: np.ndarray
    sgm_disp: np.ndarray
    disp_disagreement: np.ndarray


def compute_disp_scale(fused_base: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    fused_base = np.asarray(fused_base, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(fused_base) & (fused_base > 0)
    else:
        valid_mask = valid_mask & np.isfinite(fused_base) & (fused_base > 0)
    if int(valid_mask.sum()) == 0:
        return 1.0
    scale = float(np.percentile(fused_base[valid_mask], 95.0))
    return max(scale, 1.0)


def _to_float_map(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def build_fusion_inputs(
    image_bgr: np.ndarray,
    mono_disp_aligned: np.ndarray,
    sgm_disp: np.ndarray,
    confidence_map: np.ndarray,
    fused_base: np.ndarray | None = None,
    detail_score: np.ndarray | None = None,
    theta_low: float = 0.10,
    theta_high: float = 0.65,
    detail_suppression: float = 0.75,
    residual_gain: float = 0.90,
    disp_scale: float | None = None,
) -> FusionInputBundle:
    """
    Build normalized FusionNet inputs from heuristic fusion ingredients.
    """
    mono_disp_aligned = _to_float_map(mono_disp_aligned)
    sgm_disp = _to_float_map(sgm_disp)
    confidence_map = np.clip(_to_float_map(confidence_map), 0.0, 1.0)
    if fused_base is None or detail_score is None:
        fused_out, debug = fuse_edge_aware_residual(
            sgm_disp=sgm_disp,
            da2_aligned=mono_disp_aligned,
            confidence_map=confidence_map,
            image_bgr=image_bgr,
            theta_low=theta_low,
            theta_high=theta_high,
            detail_suppression=detail_suppression,
            residual_gain=residual_gain,
            return_debug=True,
        )
        if fused_base is None:
            fused_base = fused_out
        if detail_score is None:
            detail_score = debug["detail_score"]
    fused_base = _to_float_map(fused_base)
    detail_score = np.clip(_to_float_map(detail_score), 0.0, 1.0)

    sgm_valid = (sgm_disp > 0).astype(np.float32)
    disp_disagreement = np.abs(sgm_disp - mono_disp_aligned).astype(np.float32)
    if disp_scale is None:
        disp_scale = compute_disp_scale(fused_base, sgm_valid > 0)
    fused_base_norm = fused_base / disp_scale
    mono_norm = mono_disp_aligned / disp_scale
    sgm_norm = sgm_disp / disp_scale
    disagree_norm = disp_disagreement / disp_scale
    rgb = image_bgr.astype(np.float32) / 255.0

    stacked = np.stack(
        [
            rgb[:, :, 2],
            rgb[:, :, 1],
            rgb[:, :, 0],
            mono_norm,
            sgm_norm,
            fused_base_norm,
            confidence_map,
            sgm_valid,
            disagree_norm,
            detail_score,
        ],
        axis=0,
    ).astype(np.float32)

    return FusionInputBundle(
        input_tensor=torch.from_numpy(stacked),
        fused_base=fused_base.astype(np.float32),
        fused_base_norm=fused_base_norm.astype(np.float32),
        disp_scale=float(disp_scale),
        detail_score=detail_score.astype(np.float32),
        confidence_map=confidence_map.astype(np.float32),
        sgm_valid=sgm_valid.astype(np.float32),
        mono_disp_aligned=mono_disp_aligned.astype(np.float32),
        sgm_disp=sgm_disp.astype(np.float32),
        disp_disagreement=disp_disagreement.astype(np.float32),
    )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.norm2(x)
        x = self.act(x + residual)
        return self.act(self.ffn(x) + x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = DepthwiseSeparableBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = DepthwiseSeparableBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.block(x)


class FusionResidualNet(nn.Module):
    """Small U-Net-style residual refiner."""

    def __init__(self, in_channels: int = 10, max_residual_scale: float = DEFAULT_MAX_RESIDUAL_SCALE) -> None:
        super().__init__()
        self.max_residual_scale = float(max_residual_scale)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.enc0 = DepthwiseSeparableBlock(24)
        self.enc1 = DownBlock(24, 32)
        self.enc2 = DownBlock(32, 48)
        self.mid = DepthwiseSeparableBlock(48)
        self.up1 = UpBlock(48, 32, 32)
        self.up0 = UpBlock(32, 24, 24)
        self.head = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(self.stem(x))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        xm = self.mid(x2)
        y1 = self.up1(xm, x1)
        y0 = self.up0(y1, x0)
        return torch.tanh(self.head(y0)) * self.max_residual_scale


def get_fusion_net_runtime_config(state_or_path: str | dict | None) -> dict[str, float | bool]:
    state: dict | None
    if isinstance(state_or_path, str):
        state = torch.load(state_or_path, map_location="cpu")
    elif isinstance(state_or_path, dict):
        state = state_or_path
    else:
        state = None
    cfg = state.get("fusion_net_config", {}) if isinstance(state, dict) else {}
    return {
        "max_residual_scale": float(cfg.get("max_residual_scale", LEGACY_MAX_RESIDUAL_SCALE)),
        "lambda_anchor": float(cfg.get("lambda_anchor", LEGACY_LAMBDA_ANCHOR)),
        "apply_anchor_gate": bool(cfg.get("apply_anchor_gate", False)),
        "anchor_gate_strength": float(cfg.get("anchor_gate_strength", 0.0)),
    }


def load_fusion_net(
    weights_path: str,
    device: torch.device | str,
    max_residual_scale: float | None = None,
) -> FusionResidualNet:
    device = torch.device(device)
    state = torch.load(weights_path, map_location="cpu")
    runtime_cfg = get_fusion_net_runtime_config(state)
    if max_residual_scale is None:
        max_residual_scale = float(runtime_cfg["max_residual_scale"])
    model = FusionResidualNet(max_residual_scale=max_residual_scale)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model


def apply_anchor_gate_to_residual(
    residual: torch.Tensor,
    confidence_map: torch.Tensor,
    detail_score: torch.Tensor,
    apply_anchor_gate: bool = DEFAULT_APPLY_ANCHOR_GATE,
    anchor_gate_strength: float = DEFAULT_ANCHOR_GATE_STRENGTH,
) -> torch.Tensor:
    if not apply_anchor_gate:
        return residual
    gate_strength = float(np.clip(anchor_gate_strength, 0.0, 1.0))
    if gate_strength <= 0.0:
        return residual
    anchor_region = build_anchor_region(confidence_map, detail_score).to(residual.dtype)
    gate = 1.0 - gate_strength * anchor_region
    return residual * gate


def run_fusion_net_refinement(
    model: FusionResidualNet,
    fusion_inputs: FusionInputBundle,
    device: torch.device | str | None = None,
    apply_anchor_gate: bool = DEFAULT_APPLY_ANCHOR_GATE,
    anchor_gate_strength: float = DEFAULT_ANCHOR_GATE_STRENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    x = fusion_inputs.input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        residual = model(x)
        residual = apply_anchor_gate_to_residual(
            residual=residual,
            confidence_map=torch.from_numpy(fusion_inputs.confidence_map).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32),
            detail_score=torch.from_numpy(fusion_inputs.detail_score).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32),
            apply_anchor_gate=apply_anchor_gate,
            anchor_gate_strength=anchor_gate_strength,
        )
        residual_norm = residual[0, 0].detach().cpu().numpy().astype(np.float32)
    residual = residual_norm * fusion_inputs.disp_scale
    pred = np.clip(fusion_inputs.fused_base + residual, 0.0, None).astype(np.float32)
    return pred, residual.astype(np.float32)


def build_anchor_region(confidence_map: torch.Tensor, detail_score: torch.Tensor) -> torch.Tensor:
    return ((confidence_map >= 0.55) & (detail_score <= 0.25)).float()


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / denom


def masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    valid_dx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    valid_dy = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    loss_x = (torch.abs(pred_dx - target_dx) * valid_dx).sum() / valid_dx.sum().clamp_min(1.0)
    loss_y = (torch.abs(pred_dy - target_dy) * valid_dy).sum() / valid_dy.sum().clamp_min(1.0)
    return 0.5 * (loss_x + loss_y)


def compute_fusion_net_losses(
    pred_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    fused_base_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence_map: torch.Tensor,
    detail_score: torch.Tensor,
    lambda_disp: float = 1.0,
    lambda_grad: float = 0.2,
    lambda_anchor: float = DEFAULT_LAMBDA_ANCHOR,
) -> dict[str, torch.Tensor]:
    disp_loss = masked_smooth_l1(pred_norm, gt_norm, valid_mask)
    grad_loss = masked_gradient_l1(pred_norm, gt_norm, valid_mask)
    anchor_region = build_anchor_region(confidence_map, detail_score) * valid_mask
    anchor_loss = (torch.abs(pred_norm - fused_base_norm) * anchor_region).sum() / anchor_region.sum().clamp_min(1.0)
    total = lambda_disp * disp_loss + lambda_grad * grad_loss + lambda_anchor * anchor_loss
    return {
        "loss": total,
        "disp_loss": disp_loss.detach(),
        "grad_loss": grad_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
    }


def _random_crop_params(h: int, w: int, crop_h: int, crop_w: int) -> tuple[int, int]:
    if h <= crop_h:
        top = 0
    else:
        top = random.randint(0, h - crop_h)
    if w <= crop_w:
        left = 0
    else:
        left = random.randint(0, w - crop_w)
    return top, left


def _crop_or_pad(arr: np.ndarray, top: int, left: int, crop_h: int, crop_w: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    else:
        squeeze = False
    h, w = arr.shape[:2]
    out = np.zeros((crop_h, crop_w, arr.shape[2]), dtype=arr.dtype)
    src_top = min(top, max(0, h - 1))
    src_left = min(left, max(0, w - 1))
    copy_h = min(crop_h, max(0, h - src_top))
    copy_w = min(crop_w, max(0, w - src_left))
    out[:copy_h, :copy_w] = arr[src_top:src_top + copy_h, src_left:src_left + copy_w]
    if squeeze:
        return out[:, :, 0]
    return out


class FusionCacheDataset(Dataset):
    """Dataset over offline FusionNet cache files."""

    def __init__(
        self,
        root_dir: str,
        crop_hw: tuple[int, int] | None = None,
        training: bool = False,
        include_target: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.crop_hw = crop_hw
        self.training = training
        self.include_target = include_target
        self.files = sorted(
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if name.endswith(".npz")
        ) if os.path.isdir(root_dir) else []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        path = self.files[idx]
        data = np.load(path)
        bundle = build_fusion_inputs(
            image_bgr=data["rgb"][:, :, ::-1].astype(np.uint8),
            mono_disp_aligned=data["mono_disp_aligned"],
            sgm_disp=data["sgm_disp"],
            confidence_map=data["confidence_map"],
            fused_base=data["fused_base"],
            detail_score=data["detail_score"],
            disp_scale=float(data["disp_scale"]),
        )
        sample = {
            "input": bundle.input_tensor.numpy().transpose(1, 2, 0),
            "gt_norm": (data["gt_disp"].astype(np.float32) / float(data["disp_scale"])),
            "valid_mask": data["valid_mask"].astype(np.float32),
            "fused_base_norm": bundle.fused_base_norm.astype(np.float32),
            "confidence_map": bundle.confidence_map.astype(np.float32),
            "detail_score": bundle.detail_score.astype(np.float32),
            "disp_scale": float(data["disp_scale"]),
            "sample_name": str(data["sample_name"]),
            "dataset_name": str(data["dataset_name"]),
        }

        if self.crop_hw is not None:
            crop_h, crop_w = self.crop_hw
            h, w = sample["gt_norm"].shape
            if self.training:
                top, left = 0, 0
                min_valid = min(64, int(np.sum(sample["valid_mask"] > 0)))
                for _ in range(10):
                    cand_top, cand_left = _random_crop_params(h, w, crop_h, crop_w)
                    valid_crop = _crop_or_pad(sample["valid_mask"], cand_top, cand_left, crop_h, crop_w)
                    if float(np.sum(valid_crop > 0)) >= float(min_valid):
                        top, left = cand_top, cand_left
                        break
                else:
                    top, left = _random_crop_params(h, w, crop_h, crop_w)
            else:
                top = max((h - crop_h) // 2, 0)
                left = max((w - crop_w) // 2, 0)
            for key in ("input", "gt_norm", "valid_mask", "fused_base_norm", "confidence_map", "detail_score"):
                sample[key] = _crop_or_pad(sample[key], top, left, crop_h, crop_w)

        if self.training and random.random() < 0.5:
            for key in ("input", "gt_norm", "valid_mask", "fused_base_norm", "confidence_map", "detail_score"):
                sample[key] = np.flip(sample[key], axis=1).copy()

        x = torch.from_numpy(sample["input"].transpose(2, 0, 1)).float()
        out = {
            "input": x,
            "gt_norm": torch.from_numpy(sample["gt_norm"]).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(sample["valid_mask"]).unsqueeze(0).float(),
            "fused_base_norm": torch.from_numpy(sample["fused_base_norm"]).unsqueeze(0).float(),
            "confidence_map": torch.from_numpy(sample["confidence_map"]).unsqueeze(0).float(),
            "detail_score": torch.from_numpy(sample["detail_score"]).unsqueeze(0).float(),
            "disp_scale": sample["disp_scale"],
            "sample_name": sample["sample_name"],
            "dataset_name": sample["dataset_name"],
        }
        return out


def copy_best_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return copy.deepcopy(model.state_dict())
