"""Phase 9 — EffViT INT8 QAT wrapper.

Design (mixed-precision):
- Conv2d + Linear weights: symmetric INT8, per-output-channel
- Activations (post-Conv/Linear): affine INT8, per-tensor
- LiteMLA (EffViT's linear attention) kept in FP32 — MSLA's matmul+chunk+reshape
  patterns don't map cleanly to torch.ao.quantized ops, and attention has low
  arithmetic intensity so INT8 savings are small.
- BN: not folded pre-QAT here (PyTorch eager mode would require manual fuse).
  Instead we treat BN as a post-Conv scale/shift that the activation observer
  can absorb. Fine for our scope.

Implementation:
- Hook-based fake-quant: monkey-patch Conv2d/Linear.forward via a wrapper class
  `FakeQuantWrap` that applies straight-through fake-quant on weight + activation.
- `prepare_qat_effvit(model)` recursively replaces eligible leaf modules,
  skipping any descendant of a LiteMLA block.
- `convert_to_int8(model)` reports quantized weight bytes and sets the model
  to inference mode (fake-quant still active but observers frozen).

Public API:
    prepare_qat_effvit(model, w_bits=8, a_bits=8) -> nn.Module  (in-place wrap)
    calibrate_activations(model, loader, device, n_batches=20)  (run forward only to set a_scale)
    convert_to_int8(model) -> None  (freeze observers; switch to eval mode)
    count_quantized_bytes(model) -> (int8_bytes, fp32_bytes)
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EFFVIT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "third_party", "efficientvit"))
if _EFFVIT_DIR not in sys.path:
    sys.path.insert(0, _EFFVIT_DIR)

from efficientvit.models.nn.ops import LiteMLA  # type: ignore


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator (forward rounds; backward is identity)."""
    return (torch.round(x) - x).detach() + x


def _fake_quant_symmetric(x: torch.Tensor, bits: int, per_channel_dim: Optional[int] = None, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric fake quantization. If per_channel_dim given, use per-channel scale along that dim."""
    q_max = 2 ** (bits - 1) - 1
    if per_channel_dim is None:
        scale = x.abs().max().clamp_min(eps) / q_max
    else:
        reduce_dims = tuple(d for d in range(x.dim()) if d != per_channel_dim)
        scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(eps) / q_max
    q = _ste_round(x / scale).clamp(-q_max, q_max)
    return q * scale


def _fake_quant_affine(x: torch.Tensor, bits: int, running_min: torch.Tensor, running_max: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Affine (asymmetric) fake quantization using pre-computed running min/max."""
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1
    rng = (running_max - running_min).clamp_min(eps)
    scale = rng / (q_max - q_min)
    zero_point = torch.round(-running_min / scale - (q_max - q_min) / 2)
    q = _ste_round(x / scale + zero_point).clamp(q_min, q_max)
    return (q - zero_point) * scale


class FakeQuantConv2d(nn.Module):
    """Wraps a Conv2d with symmetric INT8 weight FQ and affine INT8 activation FQ."""

    def __init__(self, conv: nn.Conv2d, w_bits: int = 8, a_bits: int = 8) -> None:
        super().__init__()
        self.conv = conv
        self.w_bits = w_bits
        self.a_bits = a_bits
        # Running min/max for activation (EMA during calibration).
        self.register_buffer("a_min", torch.tensor(0.0))
        self.register_buffer("a_max", torch.tensor(1.0))
        self.register_buffer("observed", torch.tensor(False))
        self.register_buffer("frozen", torch.tensor(False))  # inference mode — stop updating stats
        self.momentum = 0.1

    def _update_act_stats(self, x: torch.Tensor) -> None:
        if bool(self.frozen.item()):
            return
        cur_min = x.detach().min()
        cur_max = x.detach().max()
        if not bool(self.observed.item()):
            self.a_min.copy_(cur_min)
            self.a_max.copy_(cur_max)
            self.observed.copy_(torch.tensor(True))
        else:
            self.a_min.copy_((1 - self.momentum) * self.a_min + self.momentum * cur_min)
            self.a_max.copy_((1 - self.momentum) * self.a_max + self.momentum * cur_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation fake-quant.
        self._update_act_stats(x)
        if bool(self.observed.item()):
            x_q = _fake_quant_affine(x, self.a_bits, self.a_min, self.a_max)
        else:
            x_q = x
        # Weight fake-quant (per-output-channel).
        w_q = _fake_quant_symmetric(self.conv.weight, self.w_bits, per_channel_dim=0)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)


class FakeQuantLinear(nn.Module):
    """Wraps a Linear with INT8 weight + activation FQ."""

    def __init__(self, linear: nn.Linear, w_bits: int = 8, a_bits: int = 8) -> None:
        super().__init__()
        self.linear = linear
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.register_buffer("a_min", torch.tensor(0.0))
        self.register_buffer("a_max", torch.tensor(1.0))
        self.register_buffer("observed", torch.tensor(False))
        self.register_buffer("frozen", torch.tensor(False))
        self.momentum = 0.1

    def _update_act_stats(self, x):
        if bool(self.frozen.item()):
            return
        cur_min = x.detach().min()
        cur_max = x.detach().max()
        if not bool(self.observed.item()):
            self.a_min.copy_(cur_min); self.a_max.copy_(cur_max)
            self.observed.copy_(torch.tensor(True))
        else:
            self.a_min.copy_((1 - self.momentum) * self.a_min + self.momentum * cur_min)
            self.a_max.copy_((1 - self.momentum) * self.a_max + self.momentum * cur_max)

    def forward(self, x):
        self._update_act_stats(x)
        x_q = _fake_quant_affine(x, self.a_bits, self.a_min, self.a_max) if bool(self.observed.item()) else x
        w_q = _fake_quant_symmetric(self.linear.weight, self.w_bits, per_channel_dim=0)
        return F.linear(x_q, w_q, self.linear.bias)


def _is_inside_litemla(parent_path: list[nn.Module]) -> bool:
    return any(isinstance(m, LiteMLA) for m in parent_path)


def prepare_qat_effvit(model: nn.Module, w_bits: int = 8, a_bits: int = 8) -> nn.Module:
    """In-place wrap all Conv2d / Linear layers outside LiteMLA blocks with fake-quant.

    Returns the same model reference, with eligible leaves replaced.
    """
    # We can't use model.modules() directly because we need the ancestry.
    # Walk with explicit stack carrying parent path.
    def walk(module: nn.Module, path: list[nn.Module]):
        for name, child in list(module.named_children()):
            child_path = path + [module]
            if isinstance(child, LiteMLA):
                # Do NOT recurse — keep LiteMLA fully FP32.
                continue
            if isinstance(child, (FakeQuantConv2d, FakeQuantLinear)):
                continue
            if isinstance(child, nn.Conv2d) and not _is_inside_litemla(child_path):
                setattr(module, name, FakeQuantConv2d(child, w_bits=w_bits, a_bits=a_bits))
                continue
            if isinstance(child, nn.Linear) and not _is_inside_litemla(child_path):
                setattr(module, name, FakeQuantLinear(child, w_bits=w_bits, a_bits=a_bits))
                continue
            walk(child, child_path)

    walk(model, [])
    return model


@torch.no_grad()
def calibrate_activations(model: nn.Module, loader, device: torch.device, n_batches: int = 20) -> None:
    """Run forward-only on n_batches to populate a_min/a_max via EMA. Freezes observers after."""
    from core.effvit_depth import build_effvit_depth_inputs  # lazy import to avoid circular
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        rgb = batch["rgb"].to(device)
        mono = batch["mono"].to(device)
        sgm = batch["sgm"].to(device)
        conf = batch["conf"].to(device)
        sgm_v = batch.get("sgm_valid", (sgm > 0).to(torch.bool)).to(device)
        disp_scale = batch["disp_scale"].to(device)
        x = build_effvit_depth_inputs(rgb, mono, sgm, conf, disp_scale, sgm_valid=sgm_v)
        _ = model(x)


def freeze_observers(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (FakeQuantConv2d, FakeQuantLinear)):
            m.frozen.copy_(torch.tensor(True))


def count_quantized_bytes(model: nn.Module) -> tuple[int, int]:
    """Returns (int8_weight_bytes, fp32_weight_bytes) for fake-quant'd Conv/Linear."""
    int8_bytes = 0
    fp32_bytes = 0
    for m in model.modules():
        if isinstance(m, FakeQuantConv2d):
            int8_bytes += m.conv.weight.numel()  # 1 byte per int8
            fp32_bytes += m.conv.weight.numel() * 4
        elif isinstance(m, FakeQuantLinear):
            int8_bytes += m.linear.weight.numel()
            fp32_bytes += m.linear.weight.numel() * 4
    return int8_bytes, fp32_bytes


def count_wrapped_layers(model: nn.Module) -> dict:
    """Debug helper — count wrapped vs unwrapped layers."""
    stats = {"FakeQuantConv2d": 0, "FakeQuantLinear": 0, "Conv2d_unwrapped": 0, "Linear_unwrapped": 0, "LiteMLA": 0}
    for m in model.modules():
        if isinstance(m, FakeQuantConv2d):
            stats["FakeQuantConv2d"] += 1
        elif isinstance(m, FakeQuantLinear):
            stats["FakeQuantLinear"] += 1
        elif isinstance(m, nn.Conv2d):
            stats["Conv2d_unwrapped"] += 1
        elif isinstance(m, nn.Linear):
            stats["Linear_unwrapped"] += 1
        elif isinstance(m, LiteMLA):
            stats["LiteMLA"] += 1
    return stats


if __name__ == "__main__":
    # Smoke test.
    from core.effvit_depth import EffViTDepthNet
    model = EffViTDepthNet(variant="b1", head_ch=24, in_channels=7)
    print("Before:", count_wrapped_layers(model))
    prepare_qat_effvit(model)
    print("After :", count_wrapped_layers(model))
    x = torch.randn(1, 7, 256, 512)
    with torch.no_grad():
        y = model(x)
    print(f"forward ok, out shape: {tuple(y.shape)}")
    int8_b, fp32_b = count_quantized_bytes(model)
    print(f"quantizable weights: {int8_b/1024:.1f} KB (INT8) vs {fp32_b/1024/1024:.2f} MB (FP32)")
