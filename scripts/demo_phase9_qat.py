#!/usr/bin/env python3
"""Phase 9 demo — FP32 vs W8A8 vs W4A8 side-by-side.

Per sample, vertical 9-panel summary:
  1. left
  2. gt
  3. mono (DA2)
  4. SGM (tuned)
  5. heuristic
  6. EffViT b1_h24 FP32   [4.70M | 18.8MB]
  7. EffViT b1_h24 INT8 W8A8  [4.70M | 5.6MB conv-W8 + MSLA-FP32]
  8. EffViT b1_h24 INT4 W4A8  [4.70M | 4.2MB conv-W4 + MSLA-FP32]
  9. EffViT b0_h24 INT4 W4A8  [0.74M | 1.7MB tiny]
"""
from __future__ import annotations

import argparse, glob, math, os, sys
from pathlib import Path

import numpy as np, cv2, torch, torch.nn.functional as F

_PROJECT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core.effvit_depth import EffViTDepthNet, build_effvit_depth_inputs
from core.effvit_qat import prepare_qat_effvit, freeze_observers


def _pad_to_multiple(x, m=32):
    H, W = x.shape[-2:]
    nh, nw = math.ceil(H / m) * m, math.ceil(W / m) * m
    pt, pb = (nh - H) // 2, nh - H - (nh - H) // 2
    pl, pr = (nw - W) // 2, nw - W - (nw - W) // 2
    if (pt, pb, pl, pr) == (0, 0, 0, 0):
        return x, (0, 0, 0, 0)
    return F.pad(x, (pl, pr, pt, pb), value=0), (pt, pb, pl, pr)


def _unpad(x, pads):
    pt, pb, pl, pr = pads
    if (pt, pb, pl, pr) == (0, 0, 0, 0): return x
    H, W = x.shape[-2:]
    return x[..., pt:H - pb if pb else H, pl:W - pr if pr else W]


def colorize(disp, vmin, vmax, invalid=None):
    d = disp.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    if invalid is not None: valid = valid & ~invalid
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    d = (d * 255).astype(np.uint8)
    out = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    out[~valid] = (40, 40, 40)
    return out


def metric(p, gt, valid, tau):
    if not valid.any(): return 0.0, 0.0
    err = np.abs(p[valid] - gt[valid])
    return float(err.mean()), float((err > tau).mean()) * 100.0


DATASET_CFG = {
    "kitti":      {"split": "val", "tau": 3.0, "bad_key": "D1"},
    "sceneflow":  {"split": "test", "tau": 1.0, "bad_key": "bad1"},
    "eth3d":      {"split": "eval", "tau": 1.0, "bad_key": "bad1"},
    "middlebury": {"split": "eval", "tau": 2.0, "bad_key": "bad2"},
}


# (label, ckpt_path, is_qat, params_M, weight_KB_summary)
VARIANTS = [
    ("b1_h24 FP32",     "artifacts/fusion_phase8_pareto/b1_h24/mixed_finetune/best.pt",   False, 4.70, "18.8MB"),
    ("b1_h24 W8A8",     "artifacts/fusion_phase9_qat/b1_h24/best.pt",                     True,  4.70, "Conv 2.88MB INT8"),
    ("b1_h24 W4A8",     "artifacts/fusion_phase9_qat/b1_h24_w4/best.pt",                  True,  4.70, "Conv 1.44MB INT4"),
    ("b0_h24 W4A8",     "artifacts/fusion_phase9_qat/b0_h24_w4/best.pt",                  True,  0.74, "Conv 232KB INT4"),
]


@torch.no_grad()
def load_model(path, device):
    state = torch.load(path, map_location="cpu")
    variant = state.get("variant", "b1")
    head_ch = state.get("head_ch", 24)
    in_channels = state.get("in_channels", 7)
    is_qat = state.get("qat", False)
    w_bits = state.get("w_bits", 8)
    a_bits = state.get("a_bits", 8)
    model = EffViTDepthNet(variant=variant, head_ch=head_ch, in_channels=in_channels).to(device)
    if is_qat:
        prepare_qat_effvit(model, w_bits=w_bits, a_bits=a_bits)
    model.load_state_dict(state["model"])
    if is_qat:
        freeze_observers(model)
    model.eval()
    return model, in_channels


@torch.no_grad()
def run_one(model, device, rgb_np, mono_np, sgm_np, conf_np, sgm_valid_np, disp_scale):
    rgb = torch.from_numpy(rgb_np.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
    mono = torch.from_numpy(mono_np).float().unsqueeze(0).unsqueeze(0).to(device)
    sgm = torch.from_numpy(sgm_np).float().unsqueeze(0).unsqueeze(0).to(device)
    conf = torch.from_numpy(conf_np).float().unsqueeze(0).unsqueeze(0).to(device)
    sgm_v = torch.from_numpy(sgm_valid_np).to(torch.bool).unsqueeze(0).unsqueeze(0).to(device)
    scale_t = torch.tensor([disp_scale]).float().to(device)
    rgb_p, pads = _pad_to_multiple(rgb)
    mono_p, _ = _pad_to_multiple(mono)
    sgm_p, _ = _pad_to_multiple(sgm)
    conf_p, _ = _pad_to_multiple(conf)
    sgmv_p, _ = _pad_to_multiple(sgm_v)
    x = build_effvit_depth_inputs(rgb_p, mono_p, sgm_p, conf_p, scale_t, sgm_valid=sgmv_p)
    residual_n = model(x)
    pred_pad = mono_p + residual_n * scale_t.view(1, 1, 1, 1).clamp_min(1.0)
    pred = _unpad(pred_pad, pads)
    return pred.squeeze(0).squeeze(0).cpu().float().numpy()


def make_summary(panels_with_titles, out_path):
    W_max = max(p.shape[1] for p, _ in panels_with_titles)
    header_h = 42
    rows = []
    for p, t in panels_with_titles:
        if p.shape[1] < W_max:
            p = cv2.copyMakeBorder(p, 0, 0, 0, W_max - p.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        header = np.zeros((header_h, W_max, 3), dtype=np.uint8)
        cv2.putText(header, t, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.vstack([header, p]))
    sep = np.full((3, W_max, 3), 80, dtype=np.uint8)
    stacked = []
    for i, r in enumerate(rows):
        if i: stacked.append(sep)
        stacked.append(r)
    cv2.imwrite(out_path, np.vstack(stacked))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="artifacts/fusion_cache_v3")
    ap.add_argument("--out-dir", default="results/demo_phase9_qat")
    ap.add_argument("--per-dataset", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[init] loading 4 variants ...", flush=True)
    models = {}
    for label, ckpt, is_qat, pM, sz in VARIANTS:
        if not os.path.isfile(ckpt):
            print(f"[warn] missing {ckpt}"); continue
        m, inch = load_model(ckpt, device)
        models[label] = (m, inch, pM, sz)
        print(f"  loaded {label}: {pM}M params, {sz}, in_ch={inch}")

    for ds_name, cfg in DATASET_CFG.items():
        ds_dir = os.path.join(args.cache_root, ds_name, cfg["split"])
        files = sorted(glob.glob(os.path.join(ds_dir, "*.npz")))
        if not files: continue
        for f in files[:args.per_dataset]:
            name = Path(f).stem
            data = np.load(f)
            rgb_np = data["rgb"].astype(np.uint8)
            mono_np = data["mono_disp_aligned"].astype(np.float32)
            sgm_np = data["sgm_disp"].astype(np.float32)
            conf_np = data["confidence_map"].astype(np.float32)
            gt_np = np.nan_to_num(data["gt_disp"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            valid_np = data["valid_mask"].astype(bool) & np.isfinite(data["gt_disp"])
            disp_scale = float(data["disp_scale"])
            heur_np = data["fused_base"].astype(np.float32) if "fused_base" in data.files else mono_np
            sgm_valid_np = data["sgm_valid"].astype(bool) if "sgm_valid" in data.files else (sgm_np > 0)

            tau = cfg["tau"]; badk = cfg["bad_key"]
            m_mono = metric(mono_np, gt_np, valid_np, tau)
            m_sgm = metric(sgm_np, gt_np, valid_np & sgm_valid_np, tau)
            m_heur = metric(heur_np, gt_np, valid_np, tau)
            gt_v = gt_np[valid_np]
            vmin = float(np.percentile(gt_v, 2)) if gt_v.size else 0.0
            vmax = float(np.percentile(gt_v, 98)) if gt_v.size else 1.0

            panels = [
                (cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR), f"[{ds_name}] {name}  left"),
                (colorize(gt_np, vmin, vmax, invalid=~valid_np), "GT"),
                (colorize(mono_np, vmin, vmax), f"mono(DA2)   EPE={m_mono[0]:.3f}  {badk}={m_mono[1]:.2f}%"),
                (colorize(sgm_np, vmin, vmax, invalid=~sgm_valid_np), f"SGM (tuned)   EPE={m_sgm[0]:.3f}  {badk}={m_sgm[1]:.2f}%"),
                (colorize(heur_np, vmin, vmax), f"heuristic   EPE={m_heur[0]:.3f}  {badk}={m_heur[1]:.2f}%"),
            ]
            for label, ckpt, is_qat, pM, sz in VARIANTS:
                if label not in models: continue
                model, inch, pM2, sz2 = models[label]
                pred = run_one(model, device, rgb_np, mono_np, sgm_np, conf_np, sgm_valid_np, disp_scale)
                me = metric(pred, gt_np, valid_np, tau)
                title = f"EffViT {label}  [{pM2:.2f}M | {sz2}]   EPE={me[0]:.3f}  {badk}={me[1]:.2f}%"
                panels.append((colorize(pred, vmin, vmax), title))

            scene_dir = os.path.join(args.out_dir, ds_name, name)
            os.makedirs(scene_dir, exist_ok=True)
            make_summary(panels, os.path.join(scene_dir, "00_summary.png"))
            print(f"[{ds_name}/{name}] -> {scene_dir}/00_summary.png", flush=True)


if __name__ == "__main__":
    main()
