#!/usr/bin/env python3
"""
scripts/eval_quantization.py
=============================
Post-Training Quantization (PTQ) accuracy evaluation for DepthAnythingV2 on KITTI.

Evaluates depth/disparity accuracy under four precision configurations:
  1. FP32        — baseline (no quantization)
  2. INT8        — all linear/conv layers quantized to 8-bit
  3. INT4        — all linear/conv layers quantized to 4-bit
  4. Mixed       — INT8 for attention (qkv, proj) in all blocks
                   + INT4 for MLP (fc1, fc2) in early blocks 0..split-1
                   + INT8 for MLP in late blocks split..11
                   + INT8 for decoder

Quantization method: per-channel symmetric weight-only PTQ (simulated).
  scale_c = max(|w_c|) / (2^(bits-1) - 1)   per output channel
  w_q = clamp(round(w / scale), -qmax, qmax) * scale

Metrics (paper table + KITTI standard):
  AbsRel  = mean(|pred - gt| / gt)
  RMSE    = sqrt(mean((pred - gt)^2))
  delta_1 = % pixels where max(pred/gt, gt/pred) < 1.25
  EPE     = mean(|pred - gt|)
  D1      = % pixels where |pred - gt| > max(3.0, 0.05 * gt)

All metrics are computed on aligned disparity (DA2 aligned to SGM space)
evaluated against KITTI GT disparity at valid pixels.

Usage
-----
  python scripts/eval_quantization.py                      # full 394-image eval
  python scripts/eval_quantization.py --max-samples 20     # quick sanity check
  python scripts/eval_quantization.py --configs fp32 int8 mixed  # subset
  python scripts/eval_quantization.py --mixed-split 6      # INT4 MLP in blocks 0-5
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

import core._paths  # noqa: F401  — ensures DA2 is on sys.path
from core.pipeline import (
    DA2_MODEL_CONFIGS,
    align_depth_to_sgm,
    load_da2_model,
)
from scripts.eval_kitti import (
    build_sample_list,
    read_kitti_gt,
    read_pfm,
)
from scripts.common_config import (
    DEFAULT_ALIGN_CONF_THRESHOLD,
    DEFAULT_CONF_SIGMA,
    DEFAULT_DISPARITY_RANGE,
    DEFAULT_ENCODER,
    DEFAULT_KITTI_ROOT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PKRN_MIN_DIST,
    DEFAULT_WEIGHTS,
    default_results_dir,
)

# ---------------------------------------------------------------------------
# Simulated PTQ — per-channel symmetric weight quantization
# ---------------------------------------------------------------------------

def quantize_weight_(weight: torch.Tensor, bits: int) -> None:
    """In-place per-channel symmetric quantization (simulate round-trip).

    For Conv2d: channels = weight.shape[0] (output channels)
    For Linear: channels = weight.shape[0] (output features)
    """
    qmax = (1 << (bits - 1)) - 1  # 127 for INT8, 7 for INT4
    with torch.no_grad():
        # Per output-channel scale
        w_flat = weight.view(weight.shape[0], -1)  # (C_out, *)
        amax = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (C_out, 1)
        scale = amax / qmax

        # Quantize-dequantize round-trip
        w_q = (w_flat / scale).round().clamp(-qmax, qmax) * scale
        weight.copy_(w_q.view_as(weight))


def apply_ptq(model: nn.Module, bits: int) -> None:
    """Apply simulated PTQ to ALL nn.Linear and nn.Conv2d layers."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            quantize_weight_(module.weight, bits)


def apply_mixed_ptq(
    model: nn.Module,
    attn_bits: int = 8,
    mlp_early_bits: int = 4,
    mlp_late_bits: int = 8,
    decoder_bits: int = 8,
    split_block: int = 8,
) -> None:
    """Apply mixed-precision PTQ to a DepthAnythingV2 model.

    Precision assignment:
      - Patch embedding:         attn_bits
      - Encoder block attention: attn_bits (qkv, proj)
      - Encoder block MLP:       mlp_early_bits (blocks 0..split-1)
                                  mlp_late_bits  (blocks split..end)
      - Decoder (depth_head):    decoder_bits
    """
    # 1) Patch embedding
    patch_embed = model.pretrained.patch_embed
    for m in patch_embed.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            quantize_weight_(m.weight, attn_bits)

    # 2) Transformer blocks
    if getattr(model.pretrained, 'chunked_blocks', False):
        blocks = [b for chunk in model.pretrained.blocks for b in chunk]
    else:
        blocks = list(model.pretrained.blocks)

    for i, block in enumerate(blocks):
        # Attention layers — always attn_bits
        quantize_weight_(block.attn.qkv.weight, attn_bits)
        quantize_weight_(block.attn.proj.weight, attn_bits)

        # MLP layers — early vs late
        mlp_bits = mlp_early_bits if i < split_block else mlp_late_bits
        quantize_weight_(block.mlp.fc1.weight, mlp_bits)
        quantize_weight_(block.mlp.fc2.weight, mlp_bits)

    # 3) Decoder
    for m in model.depth_head.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            quantize_weight_(m.weight, decoder_bits)


# ---------------------------------------------------------------------------
# Metrics (depth/disparity domain)
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    max_disp: float = 192.0,
) -> dict:
    """Compute AbsRel, RMSE, delta_1, EPE, D1 at valid pixels.

    All inputs are in disparity space (pixels).
    """
    mask = valid & (gt < max_disp) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    n = int(mask.sum())
    if n == 0:
        return {
            'absrel': float('nan'), 'rmse': float('nan'),
            'delta1': float('nan'), 'delta2': float('nan'),
            'delta3': float('nan'),
            'epe': float('nan'), 'd1': float('nan'), 'n': 0,
        }

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    err = np.abs(p - g)

    # Depth-style metrics (applied to disparity; valid since both > 0)
    absrel = float(np.mean(err / g))
    rmse   = float(np.sqrt(np.mean(err ** 2)))

    ratio  = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    # KITTI disparity metrics
    epe = float(np.mean(err))
    d1  = float(np.mean(err > np.maximum(3.0, 0.05 * g)) * 100.0)

    return {
        'absrel': absrel, 'rmse': rmse,
        'delta1': delta1, 'delta2': delta2, 'delta3': delta3,
        'epe': epe, 'd1': d1, 'n': n,
    }


def aggregate_depth(records: list[dict]) -> dict:
    """Macro-average across images for all metrics."""
    keys = ['absrel', 'rmse', 'delta1', 'delta2', 'delta3', 'epe', 'd1']
    out = {}
    for k in keys:
        vals = [r[k] for r in records if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    out['n'] = sum(1 for r in records if not np.isnan(r.get('epe', float('nan'))))
    return out


# ---------------------------------------------------------------------------
# Build quantized model variants
# ---------------------------------------------------------------------------

QUANT_CONFIGS = {
    'fp32': {
        'label': 'FP32 (baseline)',
        'apply': lambda model, args: None,  # no-op
    },
    'int8': {
        'label': 'INT8 (all layers)',
        'apply': lambda model, args: apply_ptq(model, bits=8),
    },
    'int4': {
        'label': 'INT4 (all layers)',
        'apply': lambda model, args: apply_ptq(model, bits=4),
    },
    'mixed': {
        'label': 'Mixed (INT8 attn + INT4 early MLP)',
        'apply': lambda model, args: apply_mixed_ptq(
            model,
            attn_bits=8,
            mlp_early_bits=4,
            mlp_late_bits=8,
            decoder_bits=8,
            split_block=args.mixed_split,
        ),
    },
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_quantization(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_quantization.log')),
            logging.StreamHandler(),
        ],
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    configs = args.configs
    logging.info('=' * 70)
    logging.info('SGM-ViT  PTQ Quantization Accuracy Evaluation')
    logging.info('=' * 70)
    logging.info(f'  KITTI root       : {args.kitti_root}')
    logging.info(f'  Encoder          : {args.encoder}')
    logging.info(f'  Precision configs: {configs}')
    logging.info(f'  Mixed split block: {args.mixed_split}')
    logging.info(f'  Device           : {device}')
    logging.info(f'  Output           : {args.out_dir}')
    logging.info('=' * 70)

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found — check --kitti-root path.')
        return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f'Limiting to {args.max_samples} samples (--max-samples)')
    logging.info(f'Total samples: {len(samples)}')

    # ---- Load FP32 reference weights once --------------------------------
    logging.info(f'Loading FP32 DepthAnythingV2 ({args.encoder}) ...')
    model_fp32 = load_da2_model(args.encoder, args.weights, device)
    fp32_state = copy.deepcopy(model_fp32.state_dict())

    # ---- Evaluate each precision config ----------------------------------
    all_results: dict[str, dict] = {}    # config_name -> aggregated metrics
    csv_rows: list[dict] = []

    for cfg_name in configs:
        if cfg_name not in QUANT_CONFIGS:
            logging.warning(f'Unknown config "{cfg_name}", skipping.')
            continue

        cfg = QUANT_CONFIGS[cfg_name]
        logging.info(f'\n{"="*70}')
        logging.info(f'  Evaluating: {cfg["label"]}')
        logging.info(f'{"="*70}')

        # Reset to FP32 weights, then apply quantization
        model_fp32.load_state_dict(copy.deepcopy(fp32_state))
        cfg['apply'](model_fp32, args)
        model = model_fp32  # reuse the same model object

        # Count quantization statistics
        n_params_total = sum(p.numel() for p in model.parameters())
        logging.info(f'  Parameters: {n_params_total / 1e6:.2f}M')

        records_all: list[dict] = []
        records_kitti15: list[dict] = []
        records_kitti12: list[dict] = []
        t_infer = 0.0

        for idx, s in enumerate(tqdm(samples, desc=cfg_name, unit='img')):
            left_bgr = cv2.imread(s['left'])
            if left_bgr is None:
                logging.warning(f'Cannot read: {s["left"]}')
                continue

            gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
            sgm_disp = read_pfm(s['sgm_pfm'])

            # Alignment confidence (binary LR-check, smoothed)
            mismatch = np.load(s['mismatch']).astype(bool)
            occlusion = np.load(s['occlusion']).astype(bool)
            conf_align = (~(mismatch | occlusion)).astype(np.float32)
            conf_align = gaussian_filter(conf_align, sigma=args.conf_sigma)

            # ---- Inference -------------------------------------------------
            t0 = time.perf_counter()
            with torch.no_grad():
                depth = model.infer_image(left_bgr, input_size=518)
            t_infer += time.perf_counter() - t0

            # ---- Align to SGM disparity space ------------------------------
            disp_aligned, scale, shift = align_depth_to_sgm(
                depth_mono=depth,
                disparity_raw=sgm_disp,
                confidence_map=conf_align,
                conf_threshold=args.conf_threshold,
            )

            # Resize to GT shape
            if disp_aligned.shape != gt_disp.shape:
                disp_aligned = cv2.resize(
                    disp_aligned,
                    (gt_disp.shape[1], gt_disp.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            # ---- Metrics ---------------------------------------------------
            m = compute_depth_metrics(disp_aligned, gt_disp, valid_gt)
            records_all.append(m)
            if s['split'] == 'kitti15':
                records_kitti15.append(m)
            else:
                records_kitti12.append(m)

            csv_rows.append({
                'config': cfg_name,
                'idx': idx,
                'split': s['split'],
                'image': os.path.basename(s['left']),
                'absrel': f"{m['absrel']:.6f}",
                'rmse': f"{m['rmse']:.4f}",
                'delta1': f"{m['delta1']:.6f}",
                'delta2': f"{m['delta2']:.6f}",
                'delta3': f"{m['delta3']:.6f}",
                'epe': f"{m['epe']:.4f}",
                'd1': f"{m['d1']:.4f}",
                'scale': f"{scale:.4f}",
                'shift': f"{shift:.4f}",
            })

            if (idx + 1) % max(1, len(samples) // 5) == 0 or idx < 3:
                logging.info(
                    f'  [{idx+1:3d}/{len(samples)}] '
                    f'AbsRel={m["absrel"]:.4f}  RMSE={m["rmse"]:.2f}  '
                    f'delta1={m["delta1"]:.4f}  D1={m["d1"]:.2f}%'
                )

        # Aggregate
        agg_all = aggregate_depth(records_all)
        agg_15 = aggregate_depth(records_kitti15)
        agg_12 = aggregate_depth(records_kitti12)
        all_results[cfg_name] = {
            'label': cfg['label'],
            'all': agg_all,
            'kitti15': agg_15,
            'kitti12': agg_12,
            'time_ms': t_infer / max(1, len(samples)) * 1e3,
        }

        logging.info(
            f'\n  {cfg["label"]} — COMBINED ({agg_all["n"]} images):\n'
            f'    AbsRel = {agg_all["absrel"]:.4f}    '
            f'RMSE = {agg_all["rmse"]:.4f}    '
            f'delta1 = {agg_all["delta1"]:.4f}\n'
            f'    EPE    = {agg_all["epe"]:.4f}    '
            f'D1   = {agg_all["d1"]:.2f}%    '
            f'Avg time = {all_results[cfg_name]["time_ms"]:.1f} ms/img'
        )

    # ====================================================================
    # Summary table
    # ====================================================================
    logging.info(f'\n{"="*70}')
    logging.info('  PTQ Quantization Results Summary')
    logging.info(f'{"="*70}')

    header = (
        f'  {"Config":<32s}  {"AbsRel":>8s}  {"RMSE":>8s}  '
        f'{"delta1":>8s}  {"delta2":>8s}  {"delta3":>8s}  '
        f'{"EPE":>8s}  {"D1%":>8s}  {"ms/img":>8s}'
    )
    sep = f'  {"-"*110}'
    logging.info(header)
    logging.info(sep)

    result_lines = [
        f'\nSGM-ViT PTQ Quantization Results ({len(samples)} KITTI images)\n',
        f'Encoder: {args.encoder}   Mixed split block: {args.mixed_split}\n',
        header, sep,
    ]

    for cfg_name in configs:
        if cfg_name not in all_results:
            continue
        r = all_results[cfg_name]
        a = r['all']
        line = (
            f'  {r["label"]:<32s}  {a["absrel"]:>8.4f}  {a["rmse"]:>8.4f}  '
            f'{a["delta1"]:>8.4f}  {a["delta2"]:>8.4f}  {a["delta3"]:>8.4f}  '
            f'{a["epe"]:>8.4f}  {a["d1"]:>8.2f}  {r["time_ms"]:>8.1f}'
        )
        logging.info(line)
        result_lines.append(line)

    # Per-split breakdown
    for split_tag, split_key in [('KITTI-2015', 'kitti15'), ('KITTI-2012', 'kitti12')]:
        logging.info(f'\n  {split_tag}:')
        result_lines.append(f'\n  {split_tag}:')
        result_lines.append(header)
        result_lines.append(sep)
        for cfg_name in configs:
            if cfg_name not in all_results:
                continue
            r = all_results[cfg_name]
            a = r[split_key]
            if a['n'] == 0:
                continue
            line = (
                f'  {r["label"]:<32s}  {a["absrel"]:>8.4f}  {a["rmse"]:>8.4f}  '
                f'{a["delta1"]:>8.4f}  {a["delta2"]:>8.4f}  {a["delta3"]:>8.4f}  '
                f'{a["epe"]:>8.4f}  {a["d1"]:>8.2f}  {"":>8s}'
            )
            logging.info(line)
            result_lines.append(line)

    # Paper-format table (copy-paste ready)
    logging.info('\n  Paper table (LaTeX-ready):')
    result_lines.append('\n  Paper table (LaTeX-ready):')
    latex_header = '  Precision & AbsRel & RMSE & $\\delta_1$ & D1-all (\\%) \\\\'
    logging.info(latex_header)
    result_lines.append(latex_header)
    logging.info('  \\midrule')
    result_lines.append('  \\midrule')
    for cfg_name in configs:
        if cfg_name not in all_results:
            continue
        r = all_results[cfg_name]
        a = r['all']
        precision_label = {
            'fp32': 'FP32', 'int8': 'INT8', 'int4': 'INT4',
            'mixed': f'Mixed (INT8/INT4, split={args.mixed_split})',
        }.get(cfg_name, cfg_name)
        latex_line = (
            f'  {precision_label} & {a["absrel"]:.3f} & {a["rmse"]:.2f} '
            f'& {a["delta1"]:.3f} & {a["d1"]:.2f} \\\\'
        )
        logging.info(latex_line)
        result_lines.append(latex_line)

    logging.info(f'\n{"="*70}')

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'quantization_results.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
    logging.info(f'Summary: {txt_path}')

    csv_path = os.path.join(args.out_dir, 'quantization_per_image.csv')
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f'Per-image CSV: {csv_path}')

    logging.info(f'{"="*70}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='SGM-ViT PTQ quantization accuracy evaluation on KITTI.'
    )
    p.add_argument('--kitti-root', default=DEFAULT_KITTI_ROOT,
        help='KITTI dataset root.')
    p.add_argument('--weights',
        default=DEFAULT_WEIGHTS,
        help='Path to DepthAnythingV2 checkpoint.')
    p.add_argument('--encoder', default=DEFAULT_ENCODER,
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--configs', nargs='+',
        default=['fp32', 'int8', 'int4', 'mixed'],
        choices=list(QUANT_CONFIGS.keys()),
        help='Which precision configs to evaluate.')
    p.add_argument('--mixed-split', type=int, default=8,
        help='Block index where MLP switches from INT4 to INT8 '
             '(blocks 0..split-1 use INT4 MLP, blocks split..end use INT8).')
    p.add_argument('--conf-threshold', type=float, default=DEFAULT_ALIGN_CONF_THRESHOLD,
        help='Minimum SGM confidence for alignment regression.')
    p.add_argument('--conf-sigma', type=float, default=DEFAULT_CONF_SIGMA,
        help='Gaussian sigma for confidence smoothing.')
    p.add_argument('--disparity-range', type=int, default=DEFAULT_DISPARITY_RANGE)
    p.add_argument('--pkrn-min-dist', type=int, default=DEFAULT_PKRN_MIN_DIST)
    p.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES,
        help='Limit to first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=default_results_dir('eval_quantization'),
        help='Directory to save results.')
    p.add_argument('--cpu', action='store_true',
        help='Force CPU inference.')
    return p.parse_args()


if __name__ == '__main__':
    evaluate_quantization(parse_args())
