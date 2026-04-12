"""Visualization helpers for merge/adaptive demo outputs."""
from __future__ import annotations

import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def colorize(arr: np.ndarray, cmap: str = "plasma", vmin=None, vmax=None) -> np.ndarray:
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    rgba = (cm(norm) * 255).astype(np.uint8)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def resize_to_match(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    th, tw = target.shape[:2]
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def save_panel(img_bgr: np.ndarray, path: str, title: str | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if title:
        bar = np.full((30, img_bgr.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(bar, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
        img_bgr = np.vstack([bar, img_bgr])
    cv2.imwrite(path, img_bgr)
    print(f"  [saved] {path}")


def build_summary_figure(
    panels: list[tuple[str, np.ndarray]],
    out_path: str,
    ncol: int = 4,
) -> None:
    n = len(panels)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(ncol * 6, nrow * 5.5), constrained_layout=True)
    fig.suptitle("SGM-ViT Merge + Adaptive Precision Demo", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    for idx, (title, bgr) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // ncol, idx % ncol])
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(n, nrow * ncol):
        fig.add_subplot(gs[idx // ncol, idx % ncol]).axis("off")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")
