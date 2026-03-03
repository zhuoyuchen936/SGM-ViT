#!/usr/bin/env python3
"""
scripts/eval_kitti.py
=====================
KITTI disparity accuracy evaluation for SGM-ViT.

Evaluates three predictors against KITTI ground-truth (annotated) disparity
on the KITTI 2012 + KITTI 2015 combined training sets:

  1. SGM (pre-computed)    — raw stereo baseline from sgm_hole/*.pfm
  2. Dense DA2 + align     — DepthAnythingV2 monocular depth, per-image
                             least-squares aligned to SGM disparity space
  3. Sparse DA2 + align    — SGM-ViT: token-pruned + re-assembled depth,
                             aligned to SGM disparity space

Alignment for (2) and (3):
    argmin_{s,t}  Σ_i || s·d_i + t − disp_sgm_i ||²
at pixels where SGM confidence >= conf_threshold (derived from the
precomputed mismatch + occlusion masks stored in sgm_hole/).
GT disparity is NOT used for alignment — only as the evaluation target.

Metrics (KITTI standard protocol):
  EPE  — mean absolute disparity error at valid GT pixels
  D1   — % of valid pixels where |pred − gt| > max(3.0, 0.05·gt)

Dataset paths (identical to rSGM_Mamba/core/stereo_datasets.py):
  KITTI 2015 root {R}:
    {R}/training/image_2/*_10.png            left images
    {R}/training/disp_occ_0/*_10.png         GT disparity  (16-bit / 256)
    {R}/training/sgm_hole/*_10.pfm           pre-computed SGM disparity
    {R}/training/sgm_hole/*_10_mismatches.npy
    {R}/training/sgm_hole/*_10_occlusion.npy

  KITTI 2012 root {R}/kitti2012:
    .../training/colored_0/*_10.png
    .../training/disp_occ/*_10.png
    .../training/sgm_hole/*_10.pfm / *_mismatches.npy / *_occlusion.npy

Usage
-----
  python scripts/eval_kitti.py
  python scripts/eval_kitti.py --encoder vitb --threshold 0.5
  python scripts/eval_kitti.py --prune-layer 6
  python scripts/eval_kitti.py --max-samples 20   # quick sanity check
  python scripts/eval_kitti.py --no-sparse        # skip sparse DA2
  python scripts/eval_kitti.py --kitti-root /your/path/to/kitti
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch

# ---------------------------------------------------------------------------
# Path setup — importable from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_DA2_DIR     = os.path.join(_PROJECT_DIR, "Depth-Anything-V2")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _DA2_DIR)

from core.token_router import SGMConfidenceTokenRouter
from core.sgm_wrapper  import confidence_to_token_grid
from demo import (
    load_da2_model, run_sparse_da2, align_depth_to_sgm,
    DA2_MODEL_CONFIGS, TOKEN_GRID_SIZE, EMBED_DIM_MAP,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_pfm(path: str) -> np.ndarray:
    """Read a .pfm disparity file → (H, W) float32."""
    with open(path, 'rb') as f:
        header = f.readline().rstrip()
        assert header in (b'PF', b'Pf'), f"Not a PFM file: {path}"
        w, h  = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data  = np.fromfile(f, endian + 'f').reshape(h, w)
    return np.ascontiguousarray(np.flipud(data)).astype(np.float32)


def read_kitti_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read KITTI ground-truth disparity (16-bit PNG, value / 256 = disparity).
    Returns (disp_gt, valid_mask).
    """
    raw   = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    disp  = raw.astype(np.float32) / 256.0
    valid = disp > 0.0
    return disp, valid


# ---------------------------------------------------------------------------
# Sample list builder
# ---------------------------------------------------------------------------

def build_sample_list(kitti_root: str) -> list[dict]:
    """
    Build sorted list of sample dicts for KITTI 2015 + KITTI 2012.

    Each dict has keys:
        left      : path to left colour image
        gt_disp   : path to GT disparity PNG
        sgm_pfm   : path to pre-computed SGM disparity PFM
        mismatch  : path to mismatch mask .npy
        occlusion : path to occlusion mask .npy
        split     : 'kitti15' or 'kitti12'
    """
    def sg(pat): return sorted(glob(pat))

    root_15 = kitti_root
    root_12 = os.path.join(kitti_root, 'kitti2012')

    samples: list[dict] = []

    # --- KITTI 2015 ---
    lefts = sg(os.path.join(root_15, 'training', 'image_2',    '*_10.png'))
    gts   = sg(os.path.join(root_15, 'training', 'disp_occ_0', '*_10.png'))
    sgms  = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10.pfm'))
    mis   = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10_mismatches.npy'))
    occ   = sg(os.path.join(root_15, 'training', 'sgm_hole',   '*_10_occlusion.npy'))
    for l, g, s, m, o in zip(lefts, gts, sgms, mis, occ):
        samples.append(dict(left=l, gt_disp=g, sgm_pfm=s,
                            mismatch=m, occlusion=o, split='kitti15'))
    logging.info(f"KITTI 2015: {len(lefts)} samples found")

    # --- KITTI 2012 ---
    lefts = sg(os.path.join(root_12, 'training', 'colored_0', '*_10.png'))
    gts   = sg(os.path.join(root_12, 'training', 'disp_occ',  '*_10.png'))
    sgms  = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10.pfm'))
    mis   = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10_mismatches.npy'))
    occ   = sg(os.path.join(root_12, 'training', 'sgm_hole',  '*_10_occlusion.npy'))
    for l, g, s, m, o in zip(lefts, gts, sgms, mis, occ):
        samples.append(dict(left=l, gt_disp=g, sgm_pfm=s,
                            mismatch=m, occlusion=o, split='kitti12'))
    logging.info(f"KITTI 2012: {len(lefts)} samples found")

    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_gt: np.ndarray,
    max_disp: float = 192.0,
) -> dict:
    """
    EPE and D1 at valid GT pixels with gt < max_disp.

    D1 criterion (KITTI standard):
        |pred − gt| > max(3.0, 0.05·gt)
    """
    mask = valid_gt & (gt < max_disp) & np.isfinite(pred) & (pred >= 0)
    n    = int(mask.sum())
    if n == 0:
        return {'epe': float('nan'), 'd1': float('nan'), 'n': 0}
    err  = np.abs(pred[mask] - gt[mask])
    epe  = float(err.mean())
    d1   = float((err > np.maximum(3.0, 0.05 * gt[mask])).mean() * 100.0)
    return {'epe': epe, 'd1': d1, 'n': n}


def aggregate(records: list[dict]) -> dict:
    """Image-level macro mean for EPE and D1."""
    epes = [r['epe'] for r in records if not np.isnan(r['epe'])]
    d1s  = [r['d1']  for r in records if not np.isnan(r['d1'])]
    return {
        'epe': float(np.mean(epes)) if epes else float('nan'),
        'd1':  float(np.mean(d1s))  if d1s  else float('nan'),
        'n':   len(epes),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, 'eval_kitti.log')),
            logging.StreamHandler(),
        ]
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    logging.info('=' * 66)
    logging.info('SGM-ViT  KITTI Disparity Evaluation')
    logging.info('=' * 66)
    logging.info(f'  KITTI root    : {args.kitti_root}')
    logging.info(f'  Encoder       : {args.encoder}')
    logging.info(f'  Prune layer   : {args.prune_layer}')
    logging.info(f'  Threshold θ   : {args.threshold}')
    logging.info(f'  Re-assembly   : {"on" if not args.no_reassembly else "off"}')
    logging.info(f'  Device        : {device}')
    logging.info(f'  Output        : {args.out_dir}')
    logging.info('=' * 66)

    # ---- Load DA2 model ------------------------------------------------
    logging.info(f'Loading DepthAnythingV2 ({args.encoder}) ...')
    model = load_da2_model(args.encoder, args.weights, device)

    # ---- Token router --------------------------------------------------
    embed_dim = EMBED_DIM_MAP[args.encoder]
    N         = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
    router    = SGMConfidenceTokenRouter(
        token_grid_size      = TOKEN_GRID_SIZE,
        confidence_threshold = args.threshold,
        learnable_threshold  = False,
    )

    # ---- Sample list ---------------------------------------------------
    samples = build_sample_list(args.kitti_root)
    if not samples:
        logging.error('No samples found — check --kitti-root path.')
        return
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f'Limiting to {args.max_samples} samples (--max-samples)')
    logging.info(f'Total samples: {len(samples)}\n')

    # ---- Per-method record collectors ----------------------------------
    records: dict[str, dict[str, list]] = {
        method: {'all': [], 'kitti15': [], 'kitti12': []}
        for method in ('sgm', 'dense', 'sparse')
    }

    csv_rows: list[dict] = []
    t_dense_total  = 0.0
    t_sparse_total = 0.0

    for idx, s in enumerate(tqdm(samples, desc='Evaluating', unit='img')):

        # ---- Load inputs -----------------------------------------------
        left_bgr = cv2.imread(s['left'])
        if left_bgr is None:
            logging.warning(f"Cannot read image: {s['left']}"); continue

        gt_disp, valid_gt = read_kitti_gt(s['gt_disp'])
        sgm_disp          = read_pfm(s['sgm_pfm'])

        mismatch  = np.load(s['mismatch']).astype(bool)
        occlusion = np.load(s['occlusion']).astype(bool)

        # Confidence: 1 = reliable SGM pixel, 0 = mismatch or occlusion
        conf_map = (~(mismatch | occlusion)).astype(np.float32)
        conf_map = gaussian_filter(conf_map, sigma=args.conf_sigma)

        # ---- SGM metrics -----------------------------------------------
        sgm_metrics = compute_metrics(sgm_disp, gt_disp, valid_gt)
        for key in ('all', s['split']):
            records['sgm'][key].append(sgm_metrics)

        # ---- Dense DA2 -------------------------------------------------
        t0          = time.perf_counter()
        depth_dense = model.infer_image(left_bgr, input_size=518)
        t_dense_total += time.perf_counter() - t0

        disp_dense, scale_d, shift_d = align_depth_to_sgm(
            depth_mono     = depth_dense,
            disparity_raw  = sgm_disp,
            confidence_map = conf_map,
            conf_threshold = args.conf_threshold,
        )
        dense_metrics = compute_metrics(disp_dense, gt_disp, valid_gt)
        for key in ('all', s['split']):
            records['dense'][key].append(dense_metrics)

        # ---- Sparse DA2 (SGM-ViT) --------------------------------------
        if not args.no_sparse:
            # Build prune mask from token routing
            conf_tensor  = torch.from_numpy(conf_map).unsqueeze(0).unsqueeze(0)
            dummy_tokens = torch.zeros(1, N, embed_dim)
            with torch.no_grad():
                routing = router(conf_tensor, dummy_tokens)

            prune_mask_1d = torch.zeros(N, dtype=torch.bool)
            prune_mask_1d[routing['prune_idx'][0]] = True

            t0           = time.perf_counter()
            depth_sparse = run_sparse_da2(
                model         = model,
                image_bgr     = left_bgr,
                prune_mask    = prune_mask_1d,
                input_size    = 518,
                prune_layer   = args.prune_layer,
                do_reassembly = not args.no_reassembly,
            )
            t_sparse_total += time.perf_counter() - t0

            disp_sparse, scale_s, shift_s = align_depth_to_sgm(
                depth_mono     = depth_sparse,
                disparity_raw  = sgm_disp,
                confidence_map = conf_map,
                conf_threshold = args.conf_threshold,
            )
            sparse_metrics = compute_metrics(disp_sparse, gt_disp, valid_gt)
            prune_ratio    = float(routing['prune_ratio'])
        else:
            sparse_metrics = {'epe': float('nan'), 'd1': float('nan'), 'n': 0}
            scale_s = shift_s = prune_ratio = float('nan')

        for key in ('all', s['split']):
            records['sparse'][key].append(sparse_metrics)

        # ---- CSV row ---------------------------------------------------
        csv_rows.append({
            'idx':       idx,
            'split':     s['split'],
            'image':     os.path.basename(s['left']),
            'sgm_epe':   f"{sgm_metrics['epe']:.4f}",
            'sgm_d1':    f"{sgm_metrics['d1']:.4f}",
            'dense_epe': f"{dense_metrics['epe']:.4f}",
            'dense_d1':  f"{dense_metrics['d1']:.4f}",
            'sparse_epe':f"{sparse_metrics['epe']:.4f}",
            'sparse_d1': f"{sparse_metrics['d1']:.4f}",
            'prune_pct': f"{prune_ratio*100:.1f}" if not np.isnan(prune_ratio) else 'n/a',
            'scale_d':   f"{scale_d:.4f}",
            'shift_d':   f"{shift_d:.4f}",
        })

        # ---- Periodic console update -----------------------------------
        if (idx + 1) % max(1, len(samples) // 10) == 0 or idx < 5:
            logging.info(
                f"[{idx+1:3d}/{len(samples)}]  SGM EPE={sgm_metrics['epe']:.2f} D1={sgm_metrics['d1']:.2f}%"
                f"  Dense EPE={dense_metrics['epe']:.2f} D1={dense_metrics['d1']:.2f}%"
                + (f"  Sparse EPE={sparse_metrics['epe']:.2f} D1={sparse_metrics['d1']:.2f}%"
                   if not args.no_sparse else '')
            )

    # ---- Aggregate and print results -----------------------------------
    splits_to_report = ['all', 'kitti15', 'kitti12']

    header = f"\n{'='*66}\n  SGM-ViT Evaluation Results\n{'='*66}"
    logging.info(header)

    result_lines: list[str] = [header]
    for split in splits_to_report:
        r_sgm    = aggregate(records['sgm'][split])
        r_dense  = aggregate(records['dense'][split])
        r_sparse = aggregate(records['sparse'][split])

        n = r_sgm['n']
        if n == 0:
            continue

        tag = {'all': 'COMBINED', 'kitti15': 'KITTI-2015', 'kitti12': 'KITTI-2012'}[split]
        line = (
            f"\n  {tag}  ({n} samples)\n"
            f"  {'Method':<20s}  {'EPE':>8s}  {'D1 (%)':>8s}\n"
            f"  {'-'*42}\n"
            f"  {'SGM (pre-computed)':<20s}  {r_sgm['epe']:>8.4f}  {r_sgm['d1']:>8.4f}\n"
            f"  {'Dense DA2 + align':<20s}  {r_dense['epe']:>8.4f}  {r_dense['d1']:>8.4f}\n"
        )
        if not args.no_sparse:
            line += f"  {'Sparse DA2 + align':<20s}  {r_sparse['epe']:>8.4f}  {r_sparse['d1']:>8.4f}\n"
        logging.info(line)
        result_lines.append(line)

    n_samples = len(samples)
    timing = (
        f"\n  Timing ({n_samples} images):\n"
        f"    Dense DA2  total: {t_dense_total:.1f}s  avg: {t_dense_total/max(1,n_samples)*1e3:.1f}ms/img\n"
    )
    if not args.no_sparse:
        timing += (
            f"    Sparse DA2 total: {t_sparse_total:.1f}s  avg: {t_sparse_total/max(1,n_samples)*1e3:.1f}ms/img\n"
        )
    logging.info(timing)

    # ---- Save results --------------------------------------------------
    txt_path = os.path.join(args.out_dir, 'eval_results.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(result_lines))
        f.write(timing)
    logging.info(f"Results saved: {txt_path}")

    csv_path = os.path.join(args.out_dir, 'eval_per_image.csv')
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logging.info(f"Per-image CSV: {csv_path}")

    logging.info('=' * 66)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='SGM-ViT KITTI disparity accuracy evaluation.'
    )
    p.add_argument('--kitti-root', default='/nfs/usrhome/pdongaa/dataczy/kitti',
        help='KITTI dataset root (contains training/ and kitti2012/).')
    p.add_argument('--weights',
        default=os.path.join(_PROJECT_DIR, 'Depth-Anything-V2', 'checkpoints',
                             'depth_anything_v2_vits.pth'),
        help='Path to DepthAnythingV2 checkpoint.')
    p.add_argument('--encoder', default='vits',
        choices=list(DA2_MODEL_CONFIGS.keys()))
    p.add_argument('--threshold', type=float, default=0.65,
        help='Token pruning confidence threshold θ.')
    p.add_argument('--prune-layer', type=int, default=0,
        help='ViT block at which pruned tokens are zeroed (0-based).')
    p.add_argument('--no-reassembly', action='store_true',
        help='Disable token re-assembly before the DPT decoder.')
    p.add_argument('--no-sparse', action='store_true',
        help='Skip sparse DA2 evaluation (only SGM and dense DA2).')
    p.add_argument('--conf-threshold', type=float, default=0.7,
        help='Minimum confidence to use a pixel for depth alignment.')
    p.add_argument('--conf-sigma', type=float, default=5.0,
        help='Gaussian σ for smoothing the confidence map (token routing).')
    p.add_argument('--max-samples', type=int, default=0,
        help='Evaluate only the first N samples (0 = all).')
    p.add_argument('--out-dir',
        default=os.path.join(_PROJECT_DIR, 'results', 'eval_kitti'),
        help='Directory to save results and log.')
    p.add_argument('--cpu', action='store_true',
        help='Force CPU inference.')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
