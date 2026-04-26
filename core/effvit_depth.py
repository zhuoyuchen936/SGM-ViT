"""EffViT-Depth: Phase 6 end-to-end depth refinement network.

Design (approved plan):
- Backbone: MIT EfficientViT-B1 (~5.8M params), 7-input-channel adapted (RGB + mono + sgm + conf + sgm_valid).
- Head: lightweight FPN decoder (~400K params) upsampling from stage4 (1/32) back to full resolution.
- Output is a residual added to the mono disparity: pred = mono + head(features).
- No visual smoothness / calibration losses. Train with L_EPE + lambda_D1 * L_D1_soft only.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_EFFVIT_DIR = os.path.join(_PROJECT_DIR, "third_party", "efficientvit")
if _EFFVIT_DIR not in sys.path:
    sys.path.insert(0, _EFFVIT_DIR)

# NOTE: third_party/efficientvit is a git clone of mit-han-lab/efficientvit.
from efficientvit.models.efficientvit.backbone import (  # type: ignore
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
)


EFFVIT_DEPTH_ARCH = "effvit_depth"
DEFAULT_IN_CHANNELS = 7  # RGB(3) + mono(1) + sgm(1) + conf(1) + sgm_valid(1)
# Phase 8 change: 8 → 7. In cache v3 `sgm_disp` holes are strictly 0, so
# `sgm_pos = (sgm_disp > 0)` is identical to `sgm_valid = ~hole_mask` in every
# train/eval/hole-aug condition. The duplicate `sgm_pos` channel adds ~144 extra
# stem-conv params and 1 channel of FLOPs per image for zero information gain.
# Keeping `sgm_valid` (the named, explicit signal) and dropping `sgm_pos`.


def _make_backbone(variant: str, in_channels: int) -> nn.Module:
    if variant == "b0":
        return efficientvit_backbone_b0(in_channels=in_channels)
    if variant == "b1":
        return efficientvit_backbone_b1(in_channels=in_channels)
    if variant == "b2":
        return efficientvit_backbone_b2(in_channels=in_channels)
    raise ValueError(f"Unknown EffViT backbone variant: {variant!r}")


class _UpBlock(nn.Module):
    """Lightweight FPN upsample block: upsample + 1x1 lateral + 3x3 fuse."""

    def __init__(self, top_ch: int, lateral_ch: int, out_ch: int) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(lateral_ch, out_ch, kernel_size=1, bias=False)
        self.project_top = nn.Conv2d(top_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Hardswish(inplace=True),
        )

    def forward(self, top: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(
            self.project_top(top), size=lateral.shape[-2:], mode="bilinear", align_corners=False,
        )
        out = up + self.lateral(lateral)
        return self.fuse(out)


class EffViTDepthNet(nn.Module):
    """EfficientViT backbone + lightweight FPN depth head.

    Input:   (B, 7, H, W) normalized (see build_inputs below).
    Output:  residual disparity (B, 1, H, W). Caller adds it to mono_disp.
    """

    def __init__(
        self,
        variant: str = "b1",
        in_channels: int = DEFAULT_IN_CHANNELS,
        head_ch: int = 48,
        residual_clamp: float = 32.0,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.in_channels = in_channels
        self.head_ch = head_ch
        self.residual_clamp = residual_clamp

        self.backbone = _make_backbone(variant, in_channels=in_channels)
        # backbone.width_list is populated after construction; use it for safety.
        wl: List[int] = list(self.backbone.width_list)
        # B1 example: [16, 32, 64, 128, 256] at resolutions [1/2, 1/4, 1/8, 1/16, 1/32].
        assert len(wl) == 5, f"Expected 5-stage backbone, got widths={wl}"
        c0, c1, c2, c3, c4 = wl

        self.up4_3 = _UpBlock(top_ch=c4, lateral_ch=c3, out_ch=head_ch)
        self.up3_2 = _UpBlock(top_ch=head_ch, lateral_ch=c2, out_ch=head_ch)
        self.up2_1 = _UpBlock(top_ch=head_ch, lateral_ch=c1, out_ch=head_ch)
        self.up1_0 = _UpBlock(top_ch=head_ch, lateral_ch=c0, out_ch=head_ch)

        # After up1_0 resolution is 1/2. Final 2x upsample + residual conv.
        self.head_conv = nn.Sequential(
            nn.Conv2d(head_ch, head_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_ch // 2),
            nn.Hardswish(inplace=True),
        )
        self.residual_conv = nn.Conv2d(head_ch // 2, 1, kernel_size=3, padding=1)
        # Zero-init residual so the network starts as identity over mono.
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        # backbone returns dict with stage0..stage4 ; stage_final is alias of stage4.
        s0 = feat["stage0"]
        s1 = feat["stage1"]
        s2 = feat["stage2"]
        s3 = feat["stage3"]
        s4 = feat["stage4"]

        p = self.up4_3(s4, s3)
        p = self.up3_2(p, s2)
        p = self.up2_1(p, s1)
        p = self.up1_0(p, s0)
        # p is at 1/2 resolution.
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        p = self.head_conv(p)
        residual = self.residual_conv(p)
        residual = torch.clamp(residual, -self.residual_clamp, self.residual_clamp)
        return residual


def build_effvit_depth_inputs(
    rgb: torch.Tensor,        # (B, 3, H, W) float in [0, 1]
    mono_disp: torch.Tensor,  # (B, 1, H, W) raw disparity
    sgm_disp: torch.Tensor,   # (B, 1, H, W) raw disparity (0 where invalid)
    confidence: torch.Tensor, # (B, 1, H, W) in [0, 1]
    disp_scale: torch.Tensor, # (B,) per-sample disparity normalization
    sgm_valid: torch.Tensor | None = None,  # (B, 1, H, W) bool/float
) -> torch.Tensor:
    """Stack 7-channel input (Phase 8 default). Disparities normalized by per-sample disp_scale.

    Channel layout: RGB(3) + mono_norm(1) + sgm_norm(1) + conf(1) + sgm_valid(1).
    If `sgm_valid` is None, falls back to `(sgm_disp > 0)` — correct when the cache
    stores zero-at-hole SGM (v3+), otherwise slightly over-permissive.
    """
    B = rgb.shape[0]
    scale = disp_scale.view(B, 1, 1, 1).clamp_min(1.0).to(rgb.dtype)
    mono_n = mono_disp / scale
    sgm_n = sgm_disp / scale
    if sgm_valid is None:
        sgm_valid_ch = (sgm_disp > 0).to(rgb.dtype)
    else:
        sgm_valid_ch = sgm_valid.to(rgb.dtype)
    return torch.cat([rgb, mono_n, sgm_n, confidence, sgm_valid_ch], dim=1)


def predict_effvit_depth(
    model: EffViTDepthNet,
    rgb: torch.Tensor,
    mono_disp: torch.Tensor,
    sgm_disp: torch.Tensor,
    confidence: torch.Tensor,
    disp_scale: torch.Tensor,
    sgm_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run forward and add residual back to mono. All disparities are in absolute pixels (not normalized)."""
    x = build_effvit_depth_inputs(rgb, mono_disp, sgm_disp, confidence, disp_scale, sgm_valid=sgm_valid)
    residual_n = model(x)  # residual in normalized units
    # Denormalize the residual to absolute disparity pixels.
    B = rgb.shape[0]
    scale = disp_scale.view(B, 1, 1, 1).clamp_min(1.0).to(residual_n.dtype)
    residual = residual_n * scale
    return mono_disp + residual


def compute_effvit_losses(
    pred: torch.Tensor,      # (B, 1, H, W) absolute disparity pixels
    gt: torch.Tensor,         # (B, 1, H, W)
    valid: torch.Tensor,      # (B, 1, H, W) bool
    tau: float | torch.Tensor = 3.0,  # D1 threshold in pixels. Scalar or (B,) per-sample tensor.
    k: float = 5.0,           # soft D1 sharpness
    lambda_d1: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """L_total = L_EPE (smooth_l1) + lambda_d1 * L_D1_soft (sigmoid surrogate).

    If `tau` is a (B,) tensor, a different threshold is used per sample (for mixed-dataset batches).
    """
    mask = valid.to(pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    err = (pred - gt) * mask
    abs_err = err.abs()
    # Smooth L1 per pixel, averaged over valid pixels.
    l1_map = torch.where(abs_err < 1.0, 0.5 * err.pow(2), abs_err - 0.5)
    epe = (l1_map * mask).sum() / denom
    # Soft D1: sigmoid(k * (|err| - tau)) over valid pixels. Tau may be per-sample.
    if isinstance(tau, torch.Tensor) and tau.dim() > 0:
        tau_b = tau.view(-1, 1, 1, 1).to(abs_err.dtype)
    else:
        tau_b = float(tau)
    soft_d1 = torch.sigmoid(k * (abs_err - tau_b))
    soft_d1 = (soft_d1 * mask).sum() / denom
    total = epe + lambda_d1 * soft_d1
    return {"total": total, "epe": epe, "soft_d1": soft_d1}


TAU_BY_DATASET = {
    "kitti": 3.0,
    "sceneflow": 1.0,
    "eth3d": 1.0,
    "middlebury": 2.0,
}


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Standalone sanity check (Phase 7: 8-channel input).
    model = EffViTDepthNet(variant="b1")
    print(f"EffViTDepthNet(b1) in_ch={model.in_channels} parameters: {count_parameters(model)/1e6:.2f}M")
    x = torch.randn(1, model.in_channels, 256, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {tuple(x.shape)}  Output: {tuple(y.shape)}")
