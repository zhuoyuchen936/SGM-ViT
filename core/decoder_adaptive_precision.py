"""
core/decoder_adaptive_precision.py
==================================
Spatially adaptive precision prototype for the DPT decoder.

The decoder bottleneck in SGM-ViT is spatial rather than token-sequence driven.
This module applies confidence-aware or confidence+texture-aware fake
quantization masks to decoder feature maps so that sensitive spatial regions
keep higher precision while smooth/reliable regions use lower precision.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from .token_merge import fake_quantize_tensor


def _normalize_map(x: np.ndarray) -> np.ndarray:
    """Robustly normalize a 2-D score map into [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_decoder_sensitivity_map(
    conf_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    conf_weight: float = 1.0,
    texture_weight: float = 0.0,
    variance_weight: float = 0.0,
) -> np.ndarray:
    """
    Build a spatial sensitivity map for decoder precision assignment.

    ``1 - conf`` preserves difficult geometric regions. ``texture`` and
    ``variance`` help protect intra-object fine structure that SGM tends to
    smooth away even when stereo confidence is high.
    """
    conf_score = 1.0 - _normalize_map(conf_map)
    score = conf_weight * conf_score

    if image_bgr is not None and (texture_weight > 0.0 or variance_weight > 0.0):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if texture_weight > 0.0:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx * gx + gy * gy)
            score += texture_weight * _normalize_map(grad_mag)

        if variance_weight > 0.0:
            mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
            sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=1.2)
            local_var = np.maximum(sq_mean - mean * mean, 0.0)
            score += variance_weight * _normalize_map(local_var)

    return _normalize_map(score)


def build_stage_high_precision_mask(
    sensitivity_map: np.ndarray,
    target_hw: tuple[int, int],
    high_precision_ratio: float,
) -> torch.Tensor:
    """Resize sensitivity to a decoder stage and select the top sensitive pixels."""
    h, w = target_hw
    if high_precision_ratio >= 1.0:
        return torch.ones(1, 1, h, w, dtype=torch.float32)
    if high_precision_ratio <= 0.0:
        return torch.zeros(1, 1, h, w, dtype=torch.float32)

    resized = cv2.resize(
        sensitivity_map.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    flat = torch.from_numpy(resized.reshape(-1))
    n_total = flat.numel()
    n_high = max(1, min(n_total, int(round(n_total * high_precision_ratio))))
    _, top_idx = flat.topk(n_high, largest=True)
    mask = torch.zeros(n_total, dtype=torch.float32)
    mask[top_idx] = 1.0
    return mask.view(1, 1, h, w)


def fake_quantize_feature_map(x: torch.Tensor, bits: int | None) -> torch.Tensor:
    """Apply per-spatial-location fake quantization over the channel dimension."""
    if bits is None or bits >= 16:
        return x
    b, c, h, w = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(-1, c)
    q = fake_quantize_tensor(flat, bits)
    return q.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def apply_spatial_precision(
    x: torch.Tensor,
    high_precision_mask: torch.Tensor,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
) -> torch.Tensor:
    """
    Mix high/low precision feature maps according to a spatial mask.
    """
    mask = high_precision_mask.to(device=x.device, dtype=x.dtype)
    q_hp = fake_quantize_feature_map(x, high_precision_bits)
    q_lp = fake_quantize_feature_map(x, low_precision_bits)
    return mask * q_hp + (1.0 - mask) * q_lp


def blend_spatial_outputs(
    fp_out: torch.Tensor,
    lp_out: torch.Tensor,
    high_precision_mask: torch.Tensor,
) -> torch.Tensor:
    """Blend FP and low-precision stage outputs under a spatial mask."""
    mask = high_precision_mask.to(device=fp_out.device, dtype=fp_out.dtype)
    return mask * fp_out + (1.0 - mask) * lp_out


def quantize_weight_tensor_(weight: torch.Tensor, bits: int) -> None:
    """In-place per-output-channel symmetric weight PTQ."""
    if bits >= 16:
        return
    qmax = (1 << (bits - 1)) - 1
    with torch.no_grad():
        w_flat = weight.view(weight.shape[0], -1)
        amax = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = amax / qmax
        w_q = (w_flat / scale).round().clamp(-qmax, qmax) * scale
        weight.copy_(w_q.view_as(weight))


def quantize_module_weights_inplace(module: nn.Module, bits: int) -> None:
    """Apply weight-only PTQ to conv/linear kernels inside a module tree."""
    if bits >= 16:
        return
    for sub in module.modules():
        if isinstance(sub, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            quantize_weight_tensor_(sub.weight, bits)


def get_weight_quantized_depth_head(depth_head: nn.Module, bits: int) -> nn.Module:
    """
    Return a cached weight-quantized copy of the DPT decoder head.

    The cache lives on the original ``depth_head`` object so repeated image
    evaluation does not deep-copy and re-quantize the decoder every time.
    """
    cache = getattr(depth_head, "_weight_quant_cache", None)
    if cache is None:
        cache = {}
        setattr(depth_head, "_weight_quant_cache", cache)
    if bits not in cache:
        device = next(depth_head.parameters()).device
        q_head = copy.deepcopy(depth_head).to(device)
        q_head.eval()
        quantize_module_weights_inplace(q_head, bits)
        cache[bits] = q_head
    return cache[bits]


def should_apply_decoder_precision(tag: str, stage_policy: str) -> bool:
    """
    Return whether adaptive precision should be applied for a decoder stage tag.

    Policies
    --------
    - ``all``        : every decoder stage
    - ``coarse_only``: low-resolution decoder stages only
    - ``fine_only``  : high-resolution decoder stages only
    """
    coarse_tags = {
        "proj_3",
        "proj_4",
        "rn_3",
        "rn_4",
        "path_3",
        "path_4",
    }
    fine_tags = {
        "proj_1",
        "proj_2",
        "rn_1",
        "rn_2",
        "path_1",
        "path_2",
        "output",
    }

    if stage_policy == "all":
        return True
    if stage_policy == "coarse_only":
        return tag in coarse_tags
    if stage_policy == "fine_only":
        return tag in fine_tags
    raise ValueError(f"Unknown decoder stage_policy: {stage_policy}")


def run_dpt_decoder_with_adaptive_precision(
    depth_head,
    out_features,
    patch_h: int,
    patch_w: int,
    sensitivity_map: np.ndarray,
    high_precision_ratio: float,
    high_precision_bits: int = 8,
    low_precision_bits: int = 4,
    stage_policy: str = "all",
) -> torch.Tensor:
    """
    Re-implement ``DPTHead.forward`` with stage-wise spatial precision masks.
    """
    out = []
    for i, x in enumerate(out_features):
        if depth_head.use_clstoken:
            x, cls_token = x[0], x[1]
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = depth_head.readout_projects[i](torch.cat((x, readout), -1))
        else:
            x = x[0]

        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        x = depth_head.projects[i](x)
        x = depth_head.resize_layers[i](x)
        tag = f"proj_{i + 1}"
        if should_apply_decoder_precision(tag, stage_policy):
            mask = build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=x.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(x.device)
            x = apply_spatial_precision(
                x,
                mask,
                high_precision_bits=high_precision_bits,
                low_precision_bits=low_precision_bits,
            )
        out.append(x)

    layer_1, layer_2, layer_3, layer_4 = out

    layer_1_rn = depth_head.scratch.layer1_rn(layer_1)
    layer_2_rn = depth_head.scratch.layer2_rn(layer_2)
    layer_3_rn = depth_head.scratch.layer3_rn(layer_3)
    layer_4_rn = depth_head.scratch.layer4_rn(layer_4)

    for tag, feat in (
        ("rn_1", layer_1_rn),
        ("rn_2", layer_2_rn),
        ("rn_3", layer_3_rn),
        ("rn_4", layer_4_rn),
    ):
        if should_apply_decoder_precision(tag, stage_policy):
            mask = build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=feat.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(feat.device)
            feat.copy_(
                apply_spatial_precision(
                    feat,
                    mask,
                    high_precision_bits=high_precision_bits,
                    low_precision_bits=low_precision_bits,
                )
            )

    path_4 = depth_head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    if should_apply_decoder_precision("path_4", stage_policy):
        path_4 = apply_spatial_precision(
            path_4,
            build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=path_4.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(path_4.device),
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )
    path_3 = depth_head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
    if should_apply_decoder_precision("path_3", stage_policy):
        path_3 = apply_spatial_precision(
            path_3,
            build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=path_3.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(path_3.device),
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )
    path_2 = depth_head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
    if should_apply_decoder_precision("path_2", stage_policy):
        path_2 = apply_spatial_precision(
            path_2,
            build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=path_2.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(path_2.device),
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )
    path_1 = depth_head.scratch.refinenet1(path_2, layer_1_rn)
    if should_apply_decoder_precision("path_1", stage_policy):
        path_1 = apply_spatial_precision(
            path_1,
            build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=path_1.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(path_1.device),
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )

    out = depth_head.scratch.output_conv1(path_1)
    out = F.interpolate(
        out,
        (int(patch_h * 14), int(patch_w * 14)),
        mode="bilinear",
        align_corners=True,
    )
    if should_apply_decoder_precision("output", stage_policy):
        out = apply_spatial_precision(
            out,
            build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=out.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(out.device),
            high_precision_bits=high_precision_bits,
            low_precision_bits=low_precision_bits,
        )
    out = depth_head.scratch.output_conv2(out)
    return out


def run_dpt_decoder_with_weight_adaptive_precision(
    depth_head,
    quantized_depth_head,
    out_features,
    patch_h: int,
    patch_w: int,
    sensitivity_map: np.ndarray,
    high_precision_ratio: float,
    high_precision_depth_head=None,
    high_precision_bits: int | None = None,
    low_precision_bits: int = 4,
    stage_policy: str = "coarse_only",
) -> torch.Tensor:
    """
    Weight-aware dual-path decoder prototype.

    Each selected stage computes:
    - high-precision output using the original decoder weights or a
      weight-quantized high-precision copy
    - low-precision output using a weight-quantized low-precision copy

    The two outputs are then mixed spatially with the high-precision mask.
    This is closer to a kernel-level dual-path realization than the earlier
    activation-only proxy.
    """
    hp_depth_head = high_precision_depth_head if high_precision_depth_head is not None else depth_head
    lp_depth_head = quantized_depth_head

    out = []
    for i, feat in enumerate(out_features):
        if depth_head.use_clstoken:
            x_fp, cls_token = feat[0], feat[1]
            readout = cls_token.unsqueeze(1).expand_as(x_fp)
            x_fp = depth_head.readout_projects[i](torch.cat((x_fp, readout), -1))
            x_lp = lp_depth_head.readout_projects[i](torch.cat((feat[0], readout), -1))
            x_hp = hp_depth_head.readout_projects[i](torch.cat((feat[0], readout), -1)) if hp_depth_head is not depth_head else x_fp
        else:
            x_fp = feat[0]
            x_hp = x_fp
            x_lp = x_fp

        x_fp = x_fp.permute(0, 2, 1).reshape((x_fp.shape[0], x_fp.shape[-1], patch_h, patch_w))
        x_hp = x_hp.permute(0, 2, 1).reshape((x_hp.shape[0], x_hp.shape[-1], patch_h, patch_w))
        x_lp = x_lp.permute(0, 2, 1).reshape((x_lp.shape[0], x_lp.shape[-1], patch_h, patch_w))

        fp_stage = depth_head.resize_layers[i](depth_head.projects[i](x_fp))
        hp_stage = hp_depth_head.resize_layers[i](hp_depth_head.projects[i](x_hp))
        tag = f"proj_{i + 1}"
        if should_apply_decoder_precision(tag, stage_policy):
            lp_stage = lp_depth_head.resize_layers[i](lp_depth_head.projects[i](x_lp))
            mask = build_stage_high_precision_mask(
                sensitivity_map,
                target_hw=hp_stage.shape[2:],
                high_precision_ratio=high_precision_ratio,
            ).to(hp_stage.device)
            stage = blend_spatial_outputs(hp_stage, lp_stage, mask)
        else:
            stage = fp_stage
        out.append(stage)

    layer_1, layer_2, layer_3, layer_4 = out

    def _blend_module(
        tag: str,
        fp_module: nn.Module,
        hp_module: nn.Module,
        lp_module: nn.Module,
        x,
        *args,
        **kwargs,
    ):
        fp_out = fp_module(x, *args, **kwargs) if callable(getattr(fp_module, "forward", None)) else fp_module(x)
        if not should_apply_decoder_precision(tag, stage_policy):
            return fp_out
        hp_out = (
            hp_module(x, *args, **kwargs)
            if hp_module is not fp_module
            else fp_out
        )
        lp_out = lp_module(x, *args, **kwargs) if callable(getattr(lp_module, "forward", None)) else lp_module(x)
        mask = build_stage_high_precision_mask(
            sensitivity_map,
            target_hw=hp_out.shape[2:],
            high_precision_ratio=high_precision_ratio,
        ).to(hp_out.device)
        return blend_spatial_outputs(hp_out, lp_out, mask)

    layer_1_rn = _blend_module("rn_1", depth_head.scratch.layer1_rn, hp_depth_head.scratch.layer1_rn, lp_depth_head.scratch.layer1_rn, layer_1)
    layer_2_rn = _blend_module("rn_2", depth_head.scratch.layer2_rn, hp_depth_head.scratch.layer2_rn, lp_depth_head.scratch.layer2_rn, layer_2)
    layer_3_rn = _blend_module("rn_3", depth_head.scratch.layer3_rn, hp_depth_head.scratch.layer3_rn, lp_depth_head.scratch.layer3_rn, layer_3)
    layer_4_rn = _blend_module("rn_4", depth_head.scratch.layer4_rn, hp_depth_head.scratch.layer4_rn, lp_depth_head.scratch.layer4_rn, layer_4)

    path_4 = _blend_module(
        "path_4",
        depth_head.scratch.refinenet4,
        hp_depth_head.scratch.refinenet4,
        lp_depth_head.scratch.refinenet4,
        layer_4_rn,
        size=layer_3_rn.shape[2:],
    )
    path_3 = _blend_module(
        "path_3",
        depth_head.scratch.refinenet3,
        hp_depth_head.scratch.refinenet3,
        lp_depth_head.scratch.refinenet3,
        path_4,
        layer_3_rn,
        size=layer_2_rn.shape[2:],
    )
    path_2 = _blend_module(
        "path_2",
        depth_head.scratch.refinenet2,
        hp_depth_head.scratch.refinenet2,
        lp_depth_head.scratch.refinenet2,
        path_3,
        layer_2_rn,
        size=layer_1_rn.shape[2:],
    )
    path_1 = _blend_module(
        "path_1",
        depth_head.scratch.refinenet1,
        hp_depth_head.scratch.refinenet1,
        lp_depth_head.scratch.refinenet1,
        path_2,
        layer_1_rn,
    )

    out_fp = depth_head.scratch.output_conv1(path_1)
    out_fp = F.interpolate(
        out_fp,
        (int(patch_h * 14), int(patch_w * 14)),
        mode="bilinear",
        align_corners=True,
    )
    if should_apply_decoder_precision("output", stage_policy):
        out_hp = hp_depth_head.scratch.output_conv1(path_1)
        out_hp = F.interpolate(
            out_hp,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out_lp = lp_depth_head.scratch.output_conv1(path_1)
        out_lp = F.interpolate(
            out_lp,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        mask = build_stage_high_precision_mask(
            sensitivity_map,
            target_hw=out_hp.shape[2:],
            high_precision_ratio=high_precision_ratio,
        ).to(out_hp.device)
        out = blend_spatial_outputs(out_hp, out_lp, mask)
    else:
        out = out_fp

    out = depth_head.scratch.output_conv2(out)
    return out
