#!/usr/bin/env python3
"""
Generate Fig.1 (co-design architecture) and Fig.2 (flash attention tiling)
for the EdgeStereoDAv2 ICCAD paper.

Usage:
    python scripts/generate_arch_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")


# ===========================================================================
# Helpers
# ===========================================================================

def _rounded_box(ax, xy, w, h, label, color, fontsize=7, textcolor="white",
                 lw=1.2, edgecolor=None, alpha=1.0, sublabel=None, zorder=3):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    ec = edgecolor or color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.30, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=textcolor, zorder=zorder + 1, style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor, zorder=zorder + 1)
    return box


def _arrow(ax, xy_from, xy_to, color="#333", lw=1.2, style="-|>",
           connectionstyle="arc3,rad=0", zorder=2, linestyle="-"):
    arrow = FancyArrowPatch(xy_from, xy_to,
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=10,
                            connectionstyle=connectionstyle,
                            zorder=zorder, linestyle=linestyle)
    ax.add_patch(arrow)
    return arrow


def _label(ax, x, y, text, fontsize=5.5, color="#555", ha="center", va="center",
           rotation=0, fontweight="normal"):
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            rotation=rotation, fontweight=fontweight, zorder=10)


# ===========================================================================
# Color palette
# ===========================================================================
C_SGM      = "#5C6BC0"   # indigo
C_PKRN     = "#7E57C2"   # purple
C_CRU      = "#EF5350"   # red
C_VFE      = "#2196F3"   # blue
C_FLASH    = "#0288D1"   # dark blue
C_DPT      = "#26A69A"   # teal
C_ADCU     = "#FFA726"   # orange
C_FUSION   = "#66BB6A"   # green
C_MEM      = "#78909C"   # grey-blue
C_CTRL     = "#8D6E63"   # brown
C_INPUT    = "#90A4AE"   # light grey
C_GAS      = "#E91E63"   # pink
C_HW_BG    = "#ECEFF1"   # very light grey
C_ALG_BG   = "#FFF8E1"   # warm cream


# ===========================================================================
# Fig 1: Co-Design Architecture
# ===========================================================================

def fig1_architecture(out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(14, 9.5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 10.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Background panels
    # -----------------------------------------------------------------------
    # Algorithm pipeline (top)
    alg_bg = FancyBboxPatch((-0.3, 5.4), 14.6, 4.3,
                            boxstyle="round,pad=0.05",
                            facecolor=C_ALG_BG, edgecolor="#BCAAA4",
                            linewidth=1.5, alpha=0.5, zorder=0)
    ax.add_patch(alg_bg)
    _label(ax, 0.4, 9.45, "ALGORITHM PIPELINE", fontsize=8, color="#5D4037",
           fontweight="bold", ha="left")

    # Hardware architecture (bottom)
    hw_bg = FancyBboxPatch((-0.3, -0.3), 14.6, 5.4,
                           boxstyle="round,pad=0.05",
                           facecolor=C_HW_BG, edgecolor="#90A4AE",
                           linewidth=1.5, alpha=0.5, zorder=0)
    ax.add_patch(hw_bg)
    _label(ax, 0.4, 4.85, "HARDWARE ARCHITECTURE (28nm / 500MHz / 1.76mm²)",
           fontsize=8, color="#37474F", fontweight="bold", ha="left")

    # ===== ALGORITHM PIPELINE (top half, y=5.6..9.3) ======================

    bw = 1.8   # box width
    bh = 0.8   # box height

    # Row 1: Input → SGM → PKRN
    _rounded_box(ax, (0.0, 8.2), bw, bh, "Stereo Pair", C_INPUT,
                 sublabel="Left + Right", textcolor="#333")
    _rounded_box(ax, (2.5, 8.2), bw, bh, "SGM Engine", C_SGM,
                 sublabel="Numba-JIT")
    _rounded_box(ax, (5.0, 8.2), bw, bh, "PKRN Conf.", C_PKRN,
                 sublabel="1−c₁/c₂")

    _arrow(ax, (1.8, 8.6), (2.5, 8.6), C_SGM)
    _label(ax, 2.15, 8.8, "L+R", fontsize=5)
    _arrow(ax, (4.3, 8.6), (5.0, 8.6), C_PKRN)
    _label(ax, 4.65, 8.8, "cost vol.", fontsize=5)

    # CRU from PKRN
    _rounded_box(ax, (5.0, 6.8), bw, bh, "CRU", C_CRU,
                 sublabel="θ→prune mask")
    _arrow(ax, (5.9, 8.2), (5.9, 7.6), C_CRU)
    _label(ax, 6.15, 7.9, "conf map", fontsize=5)

    # SGM disparity output going down
    _arrow(ax, (3.4, 8.2), (3.4, 7.6), C_SGM, linestyle="--")

    # Row 2: DA2 ViT → GAS Sparse Attn → Token Re-assembly
    _rounded_box(ax, (0.0, 6.8), bw, bh, "DA2 ViT-S", C_VFE,
                 sublabel="12 blocks, D=384")
    _rounded_box(ax, (2.5, 6.8), bw, bh, "GAS Sparse", C_GAS,
                 sublabel="Gather-Attend-Scatter")

    _arrow(ax, (1.8, 7.2), (2.5, 7.2), C_VFE)
    _label(ax, 2.15, 7.4, "1369 tokens", fontsize=5)
    # CRU → GAS (prune mask)
    _arrow(ax, (5.0, 7.2), (4.3, 7.2), C_CRU)
    _label(ax, 4.65, 7.0, "prune mask", fontsize=5, color=C_CRU)

    # Row 3: Token Reassembly → DPT Decoder
    _rounded_box(ax, (7.5, 8.2), bw, bh, "Token\nRe-assembly", C_VFE,
                 sublabel="Gaussian fill", fontsize=6.5)
    _rounded_box(ax, (7.5, 6.8), bw, bh, "DPT Decoder", C_DPT,
                 sublabel="layers {2,5,8,11}")

    _arrow(ax, (4.3, 7.4), (7.5, 8.4), C_VFE,
           connectionstyle="arc3,rad=-0.15")
    _label(ax, 5.9, 8.05, "M kept tokens", fontsize=5)

    _arrow(ax, (8.4, 8.2), (8.4, 7.6), C_DPT)
    _label(ax, 8.7, 7.9, "N tokens", fontsize=5)

    # Row 4: ADCU + Fusion
    _rounded_box(ax, (10.0, 8.2), bw, bh, "ADCU", C_ADCU,
                 sublabel="scale+shift align")
    _rounded_box(ax, (12.2, 8.2), bw, bh, "Fusion", C_FUSION,
                 sublabel="α·SGM+(1−α)·DA2")

    _arrow(ax, (9.3, 7.2), (10.0, 8.4), C_DPT,
           connectionstyle="arc3,rad=-0.2")
    _label(ax, 9.4, 7.9, "d_rel", fontsize=5)

    _arrow(ax, (11.8, 8.6), (12.2, 8.6), C_ADCU)
    _label(ax, 12.0, 8.8, "d_abs", fontsize=5)

    # SGM disparity → Fusion (long arrow from top)
    _rounded_box(ax, (10.0, 6.8), bw, bh, "SGM Disp.", C_SGM,
                 sublabel="from SGM engine", textcolor="white", alpha=0.7)
    _arrow(ax, (3.4, 7.6), (3.4, 7.35), C_SGM, lw=0.8, linestyle="--")
    _arrow(ax, (3.4, 7.2), (10.0, 7.2), C_SGM, lw=0.8, linestyle="--",
           connectionstyle="arc3,rad=0")
    _arrow(ax, (11.8, 7.2), (12.9, 8.2), C_SGM, lw=0.8, linestyle="--",
           connectionstyle="arc3,rad=-0.15")

    # PKRN conf → ADCU (for alignment anchors)
    _arrow(ax, (6.8, 8.6), (10.0, 8.6), C_PKRN, lw=0.8, linestyle="--")
    _label(ax, 8.4, 8.45, "32 Harris pts", fontsize=5, color=C_PKRN)

    # PKRN conf → Fusion (same conf drives α)
    _arrow(ax, (6.8, 8.8), (12.8, 9.0), C_PKRN, lw=0.8, linestyle="--",
           connectionstyle="arc3,rad=-0.1")
    _label(ax, 10.2, 9.15, "conf→α", fontsize=5, color=C_PKRN)
    _arrow(ax, (12.8, 9.0), (12.8, 9.0), C_PKRN, lw=0.8)

    # Output
    _rounded_box(ax, (12.2, 6.8), bw, bh, "Fused Depth", C_FUSION,
                 sublabel="EPE=1.96", alpha=0.7)
    _arrow(ax, (13.1, 8.2), (13.1, 7.6), C_FUSION)

    # Label: "one conf map drives 3 functions"
    _label(ax, 5.9, 5.95, "▲ One PKRN confidence map drives: CRU pruning + ADCU calibration + Fusion blending",
           fontsize=6.5, color="#D32F2F", fontweight="bold")

    # ===== HARDWARE ARCHITECTURE (bottom half, y=-0.1..4.7) ===============

    hw_y0 = 0.2   # bottom row y
    hw_y1 = 2.0   # middle row y
    hw_y2 = 3.6   # top row y
    hw_bw = 2.2
    hw_bh = 1.1

    # --- Top row: compute engines ---
    # VFE
    _rounded_box(ax, (0.0, hw_y2), hw_bw, hw_bh, "VFE", C_VFE,
                 sublabel="32×32 PE Array\n1024 INT8 MACs", fontsize=7)
    # details inside VFE
    _label(ax, 1.1, hw_y2 + 0.15, "WS dataflow | ~90% util.", fontsize=4.5,
           color="white")

    # CRU
    _rounded_box(ax, (2.6, hw_y2), 1.6, hw_bh, "CRU", C_CRU,
                 sublabel="1369 comparators\n<0.3% area", fontsize=6.5)
    # details
    _label(ax, 3.4, hw_y2 + 0.15, "1-cycle latency", fontsize=4.5,
           color="white")

    # Flash Attention
    _rounded_box(ax, (4.5, hw_y2), hw_bw, hw_bh, "Flash Attn", C_FLASH,
                 sublabel="Br=64 × Bc=128\n183× mem reduction", fontsize=6.5)
    _label(ax, 5.6, hw_y2 + 0.15, "16KB tile (vs 22.5MB)", fontsize=4.5,
           color="white")

    # CSFE (DPT decoder)
    _rounded_box(ax, (7.0, hw_y2), hw_bw, hw_bh, "CSFE", C_DPT,
                 sublabel="256 MACs, OS mode\n1×1 + 3×3 conv", fontsize=6.5)
    _label(ax, 8.1, hw_y2 + 0.15, "bilinear upsample", fontsize=4.5,
           color="white")

    # ADCU
    _rounded_box(ax, (9.5, hw_y2), hw_bw, hw_bh, "ADCU", C_ADCU,
                 sublabel="Harris + NCC + LS\n2.7% area", fontsize=6.5)
    _label(ax, 10.6, hw_y2 + 0.15, "32 pts → scale,shift", fontsize=4.5,
           color="white")

    # Fusion HW
    _rounded_box(ax, (12.0, hw_y2), 1.8, hw_bh, "Fusion\nUnit", C_FUSION,
                 sublabel="α blend", fontsize=6.5)

    # --- Specialized units row ---
    sp_bw = 2.0
    sp_bh = 0.85

    _rounded_box(ax, (0.0, hw_y1), sp_bw, sp_bh, "PWL Softmax", "#5C6BC0",
                 sublabel="16-seg, <0.015% err", fontsize=6, textcolor="white")
    _rounded_box(ax, (2.2, hw_y1), sp_bw, sp_bh, "GELU Approx", "#5C6BC0",
                 sublabel="32-seg PWL", fontsize=6, textcolor="white")
    _rounded_box(ax, (4.4, hw_y1), sp_bw, sp_bh, "LayerNorm", "#5C6BC0",
                 sublabel="3-stage streaming", fontsize=6, textcolor="white")
    _rounded_box(ax, (6.6, hw_y1), sp_bw, sp_bh, "GAS Logic", C_GAS,
                 sublabel="Gather/Scatter idx", fontsize=6, textcolor="white")
    _rounded_box(ax, (8.8, hw_y1), sp_bw, sp_bh, "1/Z LUT", C_ADCU,
                 sublabel="4096-entry interp", fontsize=6, textcolor="white")
    _rounded_box(ax, (11.0, hw_y1), sp_bw, sp_bh, "Zero-Skip", "#78909C",
                 sublabel="~1.3× speedup", fontsize=6, textcolor="white")

    # --- Bottom row: memory hierarchy + control ---
    _rounded_box(ax, (0.0, hw_y0), 2.4, 1.1, "L0 Reg File", C_MEM,
                 sublabel="512B/PE\n0-cycle", fontsize=6)
    _rounded_box(ax, (2.7, hw_y0), 2.4, 1.1, "L1 Local SRAM", C_MEM,
                 sublabel="64KB/engine\n1-cycle", fontsize=6)
    _rounded_box(ax, (5.4, hw_y0), 2.8, 1.1, "L2 Global SRAM", C_MEM,
                 sublabel="512KB, 32 banks\n3-cycle, DMA", fontsize=6)
    _rounded_box(ax, (8.5, hw_y0), 2.4, 1.1, "L3 DRAM", C_MEM,
                 sublabel="LPDDR4x, 2GB\n50-cycle", fontsize=6)
    _rounded_box(ax, (11.2, hw_y0), 2.6, 1.1, "RISC-V\nController", C_CTRL,
                 sublabel="Scheduling + DMA", fontsize=6)

    # --- Arrows between HW blocks ---
    # VFE ↔ Flash Attn
    _arrow(ax, (2.2, hw_y2 + 0.55), (4.5, hw_y2 + 0.55), C_FLASH, lw=0.8)
    # CRU → GAS Logic
    _arrow(ax, (3.4, hw_y2), (7.3, hw_y1 + 0.85), C_CRU, lw=0.8,
           connectionstyle="arc3,rad=0.2")
    # VFE ↔ CSFE
    _arrow(ax, (2.2, hw_y2 + 0.3), (7.0, hw_y2 + 0.3), "#666", lw=0.6,
           linestyle="--", connectionstyle="arc3,rad=0.15")
    _label(ax, 4.6, hw_y2 + 1.15, "features @{2,5,8,11}", fontsize=4.5,
           color="#666")
    # CSFE → ADCU
    _arrow(ax, (9.2, hw_y2 + 0.55), (9.5, hw_y2 + 0.55), C_ADCU, lw=0.8)
    # ADCU → Fusion
    _arrow(ax, (11.7, hw_y2 + 0.55), (12.0, hw_y2 + 0.55), C_FUSION, lw=0.8)

    # Memory ↔ Compute (vertical dotted arrows)
    for cx in [1.1, 3.9, 5.6, 8.1, 10.6]:
        _arrow(ax, (cx, hw_y1), (cx, hw_y0 + 1.1), "#78909C", lw=0.5,
               linestyle=":", style="-")

    # --- Co-design mapping arrows (algorithm → hardware) ---
    map_style = "arc3,rad=0.0"
    map_color = "#B71C1C"
    map_lw = 0.9

    # Algorithm SGM → HW (external, off-chip)
    _label(ax, 0.3, 5.55, "← SGM runs off-chip (Numba-JIT / FPGA fabric)",
           fontsize=5.5, color="#5C6BC0", ha="left", fontweight="bold")

    # Algorithm DA2 ViT → HW VFE
    _arrow(ax, (0.9, 6.8), (0.9, 4.7), map_color, lw=map_lw, linestyle=":")
    _label(ax, 0.4, 5.4, "maps to", fontsize=4.5, color=map_color, rotation=90)

    # Algorithm CRU → HW CRU
    _arrow(ax, (5.9, 6.8), (3.4, 4.7), map_color, lw=map_lw, linestyle=":",
           connectionstyle="arc3,rad=0.15")

    # Algorithm DPT → HW CSFE
    _arrow(ax, (8.4, 6.8), (8.1, 4.7), map_color, lw=map_lw, linestyle=":")

    # Algorithm ADCU → HW ADCU
    _arrow(ax, (10.9, 8.2), (10.6, 4.7), map_color, lw=map_lw, linestyle=":")

    # Algorithm Fusion → HW Fusion
    _arrow(ax, (13.1, 6.8), (12.9, 4.7), map_color, lw=map_lw, linestyle=":")

    # --- Quantization annotation ---
    _label(ax, 13.8, hw_y1 + 0.5,
           "Quant:\nINT8 attn\nINT4 MLP\n(blk 0-7)",
           fontsize=5, color="#666", ha="center")

    # --- Title ---
    fig.suptitle("Fig. 1: EdgeStereoDAv2 Algorithm-Hardware Co-Design Architecture",
                 fontsize=11, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "fig1_architecture.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ===========================================================================
# Fig 2: Flash Attention Tiling
# ===========================================================================

def fig2_flash_attention(out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.2, 1.6, 1.0]})

    # -----------------------------------------------------------------------
    # Panel (a): Naive attention memory problem
    # -----------------------------------------------------------------------
    ax = axes[0]
    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Naive Attention: 22.5 MB", fontsize=9, fontweight="bold")

    # Q matrix
    _rounded_box(ax, (0, 4.5), 1.2, 1.5, "Q", "#1565C0", fontsize=10)
    _label(ax, 0.6, 4.3, "1370×64", fontsize=6, color="#1565C0")

    # × symbol
    _label(ax, 1.5, 5.25, "×", fontsize=14, color="#333")

    # K^T matrix
    _rounded_box(ax, (1.8, 4.5), 1.5, 1.2, "K^T", "#2E7D32", fontsize=10)
    _label(ax, 2.55, 4.3, "64×1370", fontsize=6, color="#2E7D32")

    # = symbol
    _label(ax, 3.6, 5.1, "=", fontsize=14, color="#333")

    # Full attention matrix (big red square - problem!)
    attn = FancyBboxPatch((3.9, 3.5), 2.0, 2.5,
                          boxstyle="round,pad=0.02",
                          facecolor="#FFCDD2", edgecolor="#E53935",
                          linewidth=2, zorder=3)
    ax.add_patch(attn)
    ax.text(4.9, 4.75, "S", fontsize=12, ha="center", va="center",
            fontweight="bold", color="#C62828", zorder=4)
    _label(ax, 4.9, 4.1, "1370×1370", fontsize=6, color="#C62828")
    _label(ax, 4.9, 3.7, "×6 heads", fontsize=5.5, color="#C62828")

    # Memory callout
    ax.annotate("22.5 MB\n(FP16)", xy=(5.9, 5.5), fontsize=8,
                fontweight="bold", color="#C62828",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                          edgecolor="#E53935"))

    # × sign for "exceeds SRAM"
    ax.plot([4.0, 5.8], [3.6, 5.9], color="#E53935", linewidth=3, alpha=0.4)
    ax.plot([4.0, 5.8], [5.9, 3.6], color="#E53935", linewidth=3, alpha=0.4)

    _label(ax, 3.0, 2.8, "X  Exceeds 512KB\non-chip SRAM", fontsize=7,
           color="#C62828", fontweight="bold")

    # Show V
    _rounded_box(ax, (0, 2.5), 1.2, 1.5, "V", "#E65100", fontsize=10)
    _label(ax, 0.6, 2.3, "1370×64", fontsize=6, color="#E65100")

    _label(ax, 1.5, 3.25, "×", fontsize=14, color="#333")
    _label(ax, 2.5, 3.25, "S", fontsize=10, color="#C62828")
    _label(ax, 3.5, 3.25, "→  O", fontsize=10, color="#333")

    # -----------------------------------------------------------------------
    # Panel (b): Tiled flash attention
    # -----------------------------------------------------------------------
    ax = axes[1]
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Tiled Flash Attention: 16 KB", fontsize=9,
                 fontweight="bold")

    # Q blocks (tall, split into Br=64 row blocks)
    q_colors = ["#1565C0", "#1976D2", "#1E88E5", "#2196F3", "#42A5F5"]
    n_q = 5  # show 5 of ~22 blocks
    q_h_each = 0.8
    for i in range(n_q):
        y = 6.0 - i * (q_h_each + 0.1)
        alpha = 1.0 if i == 1 else 0.35  # highlight current block
        _rounded_box(ax, (0, y), 0.8, q_h_each, f"Q[{i}]", q_colors[i % 5],
                     fontsize=6, alpha=alpha)
    _label(ax, 0.4, 1.8, "...", fontsize=12, color="#1565C0")
    _label(ax, 0.4, 1.2, "22 blocks\n(Br=64)", fontsize=5.5, color="#1565C0")

    # Current Q block highlighted with arrow
    _arrow(ax, (0.8, 5.6), (1.6, 5.6), "#1565C0", lw=1.5)
    _label(ax, 1.2, 5.85, "current", fontsize=5, color="#1565C0")

    # K/V blocks (wide, split into Bc=128 col blocks)
    kv_w_each = 0.7
    kv_colors = ["#2E7D32", "#388E3C", "#43A047", "#4CAF50"]
    n_kv = 4
    for j in range(n_kv):
        x = 1.8 + j * (kv_w_each + 0.15)
        alpha = 0.35
        if j == 2:
            alpha = 1.0  # current block pair
        _rounded_box(ax, (x, 6.5), kv_w_each, 0.5, f"K[{j}]",
                     kv_colors[j % 4], fontsize=5.5, alpha=alpha)
        _rounded_box(ax, (x, 5.8), kv_w_each, 0.5, f"V[{j}]",
                     "#E65100", fontsize=5.5, alpha=alpha)
    _label(ax, 5.7, 6.75, "... 11 blocks (Bc=128)", fontsize=5.5,
           color="#2E7D32", ha="left")

    # Current tile (highlighted)
    tile_x, tile_y = 3.5, 4.2
    tile = FancyBboxPatch((tile_x, tile_y), 1.6, 1.2,
                          boxstyle="round,pad=0.02",
                          facecolor="#C8E6C9", edgecolor="#2E7D32",
                          linewidth=2.5, zorder=5)
    ax.add_patch(tile)
    ax.text(tile_x + 0.8, tile_y + 0.75, "S_tile", fontsize=9,
            ha="center", va="center", fontweight="bold", color="#1B5E20",
            zorder=6)
    ax.text(tile_x + 0.8, tile_y + 0.25, "64×128", fontsize=7,
            ha="center", va="center", color="#2E7D32", zorder=6)

    # Arrow from Q block + K block → tile
    _arrow(ax, (1.6, 5.3), (tile_x, tile_y + 1.1), "#333", lw=1.0,
           connectionstyle="arc3,rad=-0.1")
    _arrow(ax, (3.85, 5.8), (tile_x + 0.8, tile_y + 1.2), "#333", lw=1.0,
           connectionstyle="arc3,rad=-0.1")

    # Memory callout
    ax.annotate("16 KB\nonly!", xy=(5.3, 4.8), fontsize=9,
                fontweight="bold", color="#2E7D32",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                          edgecolor="#4CAF50"))

    # Running statistics
    stat_x = 1.5
    stat_y = 2.5
    stat_box = FancyBboxPatch((stat_x, stat_y), 4.2, 1.2,
                              boxstyle="round,pad=0.03",
                              facecolor="#FFF3E0", edgecolor="#E65100",
                              linewidth=1.5, zorder=3)
    ax.add_patch(stat_box)
    ax.text(stat_x + 2.1, stat_y + 0.9, "Running Statistics (per row)",
            fontsize=7, ha="center", va="center", fontweight="bold",
            color="#BF360C", zorder=4)
    ax.text(stat_x + 2.1, stat_y + 0.45,
            "m[i] = running max    l[i] = running sum\n"
            "FP16 registers — updated each K/V block",
            fontsize=5.5, ha="center", va="center", color="#BF360C", zorder=4)

    _arrow(ax, (tile_x + 0.8, tile_y), (stat_x + 2.1, stat_y + 1.2),
           "#E65100", lw=1.0)

    # Output
    out_x, out_y = 6.5, 4.2
    _rounded_box(ax, (out_x, out_y), 1.5, 1.2, "O[i]", "#0D47A1",
                 sublabel="64×64\naccumulated", fontsize=7)
    _arrow(ax, (tile_x + 1.6, tile_y + 0.6), (out_x, out_y + 0.6),
           "#0D47A1", lw=1.2)
    _label(ax, 5.85, 4.55, "softmax\n× V_tile", fontsize=5, color="#0D47A1")

    # Schedule arrow (loop)
    ax.annotate("", xy=(3.2, 5.7), xytext=(4.5, 5.7),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0,
                                connectionstyle="arc3,rad=-0.5"))
    _label(ax, 3.85, 5.55, "stream K/V\nblocks", fontsize=5, color="#666")

    # Algorithm description
    _label(ax, 4.0, 0.8,
           "for each Q_block[i]:            (22 outer iterations)\n"
           "  for each K_block[j], V_block[j]:  (11 inner iterations)\n"
           "    S_tile = Q[i] × K[j]^T         (64×128 — 16KB)\n"
           "    update m[i], l[i]               (running max/sum)\n"
           "    O[i] += softmax(S_tile) × V[j]  (accumulate)",
           fontsize=5.5, color="#333", ha="center",
           fontweight="normal")
    rect = FancyBboxPatch((1.3, 0.15), 5.4, 1.3,
                          boxstyle="round,pad=0.05",
                          facecolor="#FAFAFA", edgecolor="#BDBDBD",
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect)

    # -----------------------------------------------------------------------
    # Panel (c): Comparison summary
    # -----------------------------------------------------------------------
    ax = axes[2]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("(c) Memory Comparison", fontsize=9, fontweight="bold")

    # Bar chart: naive vs tiled
    bar_x = [1.2, 3.2]
    bar_h = [5.5, 0.03]  # 22.5MB vs 16KB (normalized ~5.5 vs ~0.03)
    bar_colors_c = ["#FFCDD2", "#C8E6C9"]
    bar_edge = ["#E53935", "#4CAF50"]

    for i, (bx, bh_val, bc, be) in enumerate(
            zip(bar_x, bar_h, bar_colors_c, bar_edge)):
        rect = FancyBboxPatch((bx, 0.8), 1.2, bh_val,
                              boxstyle="round,pad=0.02",
                              facecolor=bc, edgecolor=be,
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)

    # Labels
    ax.text(1.8, 6.5, "22.5 MB", fontsize=10, ha="center", fontweight="bold",
            color="#C62828")
    ax.text(1.8, 0.5, "Naive", fontsize=7, ha="center", color="#C62828")

    ax.text(3.8, 1.2, "16 KB", fontsize=10, ha="center", fontweight="bold",
            color="#2E7D32")
    ax.text(3.8, 0.5, "Tiled FA", fontsize=7, ha="center", color="#2E7D32")

    # 183× label
    ax.annotate("183×\nreduction",
                xy=(2.8, 3.5), fontsize=11, fontweight="bold",
                color="#FF6F00", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                          edgecolor="#FF6F00", linewidth=1.5))

    # Arrow between bars
    ax.annotate("", xy=(3.2, 3.0), xytext=(2.4, 3.0),
                arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=2))

    # Additional stats
    _label(ax, 2.5, 0.1,
           "Tile: Br=64, Bc=128 | 6 heads | Zero accuracy loss",
           fontsize=5.5, color="#555")

    fig.suptitle("Fig. 2: Tiled Flash-Attention Engine — Memory Reduction from 22.5 MB to 16 KB",
                 fontsize=10, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "fig2_flash_attention.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating → {OUT_DIR}")
    fig1_architecture(OUT_DIR)
    fig2_flash_attention(OUT_DIR)
    print("Done.")
