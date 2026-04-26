#!/usr/bin/env python3
"""Phase 8 Stage 3: report params and GFLOPs for an EffViTDepthNet variant.

Usage:
  python scripts/profile_effvit.py --variant b0 --head-ch 24 --in-channels 7
  python scripts/profile_effvit.py --ckpt artifacts/fusion_phase8_pareto/b1_h48/mixed_finetune/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import EffViTDepthNet


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_flops(model: torch.nn.Module, input_hw: tuple[int, int], in_ch: int) -> float:
    """Return GFLOPs for one forward on a fixed input size."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e:
        print(f"[warn] fvcore unavailable ({e}); attempting thop fallback", file=sys.stderr)
        try:
            from thop import profile
            model.eval()
            dummy = torch.randn(1, in_ch, input_hw[0], input_hw[1])
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            return float(macs * 2) / 1e9
        except ImportError:
            return float("nan")
    model.eval()
    dummy = torch.randn(1, in_ch, input_hw[0], input_hw[1])
    with torch.no_grad():
        fca = FlopCountAnalysis(model, dummy).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = fca.total()
    return float(total) / 1e9  # GFLOPs (fvcore counts MACs = multiply-accumulate; close to FLOPs/2 convention)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="b1", choices=["b0", "b1", "b2"])
    ap.add_argument("--head-ch", type=int, default=48)
    ap.add_argument("--in-channels", type=int, default=7)
    ap.add_argument("--ckpt", default=None, help="If provided, inherit variant/head_ch/in_channels from ckpt")
    ap.add_argument("--input-hw", type=int, nargs=2, default=[384, 768], help="Profile at this input size")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    variant, head_ch, in_channels = args.variant, args.head_ch, args.in_channels
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        variant = state.get("variant", variant)
        head_ch = state.get("head_ch", head_ch)
        in_channels = state.get("in_channels", in_channels)

    model = EffViTDepthNet(variant=variant, head_ch=head_ch, in_channels=in_channels)
    params = count_params(model)
    gflops = compute_flops(model, tuple(args.input_hw), in_channels)

    out = {
        "variant": variant,
        "head_ch": head_ch,
        "in_channels": in_channels,
        "input_hw": list(args.input_hw),
        "params_M": round(params / 1e6, 3),
        "params_raw": int(params),
        "gflops": round(gflops, 3) if gflops == gflops else None,
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
