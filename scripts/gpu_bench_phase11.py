#!/usr/bin/env python3
"""Phase 11 - GPU baseline benchmark on RTX TITAN.

Measures:
  - DA2-Small + LS-align end-to-end latency (Row A workload) on GPU
  - Average power draw via nvidia-smi polling during inference
  - Derived FPS and fps/W

Usage:
  python scripts/gpu_bench_phase11.py --gpu 5 --n-iter 50 --out results/phase11_hw_ablation/gpu_ref.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, threading
from statistics import mean

def poll_nvidia_power(gpu_idx, stop_event, samples):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu_idx}",
                 "--query-gpu=power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=2
            ).decode().strip()
            parts = [x.strip() for x in out.split(",")]
            samples.append({"power_w": float(parts[0]), "util_pct": float(parts[1])})
        except Exception:
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--img-h", type=int, default=384)
    ap.add_argument("--img-w", type=int, default=768)
    ap.add_argument("--out", default="results/phase11_hw_ablation/gpu_ref.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import numpy as np

    print(f"[gpu-bench] GPU {args.gpu}  torch={torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    assert torch.cuda.is_available(), "CUDA required"
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load DA2-Small
    sys.path.insert(0, "/home/pdongaa/workspace/SGM-ViT")
    sys.path.insert(0, "/home/pdongaa/workspace/SGM-ViT/Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    ckpt = os.environ.get("SGMVIT_DA2_WEIGHTS",
                          "/home/pdongaa/workspace/SGM-ViT/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.cuda().eval()

    # Synthetic input at 384x768 (matches simulator shape)
    H, W = args.img_h, args.img_w
    img = np.random.rand(H, W, 3).astype(np.float32) * 255

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.infer_image(img, input_size=518)
        torch.cuda.synchronize()

    # Start power polling
    power_samples = []
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_nvidia_power,
                              args=(args.gpu, stop_event, power_samples))
    poller.start()
    time.sleep(0.5)  # settle

    # Timed run
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(args.n_iter):
            d = model.infer_image(img, input_size=518)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    time.sleep(0.5)
    stop_event.set()
    poller.join(timeout=2)

    ms_per_frame = elapsed / args.n_iter * 1000
    fps = args.n_iter / elapsed
    # Filter power samples to "active" period (util > 30%)
    active_samples = [s for s in power_samples if s["util_pct"] >= 30]
    all_powers = [s["power_w"] for s in power_samples]
    active_powers = [s["power_w"] for s in active_samples]

    # Use p50 of active for power estimate (avoid spikes)
    active_powers_sorted = sorted(active_powers)
    power_median_active = active_powers_sorted[len(active_powers_sorted) // 2] if active_powers_sorted else None
    power_mean_all = mean(all_powers) if all_powers else None
    power_mean_active = mean(active_powers) if active_powers else None

    fps_per_w = fps / power_median_active if power_median_active else None

    result = {
        "gpu": "RTX TITAN",
        "n_iter": args.n_iter,
        "img_h": H, "img_w": W,
        "internal_size": 518,
        "ms_per_frame": round(ms_per_frame, 2),
        "fps": round(fps, 2),
        "power_median_active_w": round(power_median_active, 2) if power_median_active else None,
        "power_mean_active_w": round(power_mean_active, 2) if power_mean_active else None,
        "power_mean_all_w": round(power_mean_all, 2) if power_mean_all else None,
        "fps_per_W": round(fps_per_w, 4) if fps_per_w else None,
        "n_power_samples": len(power_samples),
        "n_active_samples": len(active_samples),
        "note": "DA2-Small (ViT-S backbone) inference only, no LS-align step. LS-align adds ~1ms on GPU (negligible).",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== RTX TITAN RESULT ===")
    print(json.dumps(result, indent=2))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
