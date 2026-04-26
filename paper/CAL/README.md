# CAL 4-page Letter — SA + APT Co-Design Draft

First draft of the IEEE CAL letter pivoted around two pillars:
1. Reconfigurable 32×32 systolic array (OS/WS/IS + LiteMLA direct mapping + depthwise sidecar)
2. Adaptive Precision Token (APT) — density × precision under one external PKRN signal

with **hardware–software co-design** as the umbrella narrative.

## Directory Layout

```
paper/CAL/
├── main.tex              # ~180 lines, SA+APT pillars
├── main.pdf              # 4 pages, compile target
├── README.md
├── IEEEtran.bst
├── figures/
│   ├── fig_arch_cal.{tex,pdf}         # simplified architecture
│   ├── fig_sa_utilization.{tex,pdf}   # per-phase PE util (PLACEHOLDER data)
│   ├── fig_apt_quadrant.{tex,pdf}     # token quadrant + hp sweep inset
│   └── fig_cosim_heatmap.{tex,pdf}    # SA × APT joint sweep (PLACEHOLDER data)
├── tables/
│   ├── table_complementarity.tex      # 40-image KITTI band breakdown
│   ├── table_hp_sweep.tex             # hp ∈ {25,50,75,100}% sweep
│   ├── table_apt_2x2.tex              # density × precision factorial
│   └── table_main_results.tex         # compact KITTI + SceneFlow
└── build/                             # .aux / .log scratch dir
```

`main.tex` declares `\graphicspath{{figures/}}` and references tables via `\input{tables/<name>}`.

## Build

```bash
cd /home/pdongaa/workspace/SGM-ViT/paper/CAL
export PATH=/home/pdongaa/.local/bin:$PATH

# (Re)compile each TikZ figure — only needed if the .tex source changed
cd figures && for f in fig_*.tex; do pdflatex -interaction=nonstopmode "$f"; done && cd ..

# Compile main (two passes for cross-refs; inline \thebibliography → no bibtex)
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

mv *.aux *.log build/ 2>/dev/null
```

Expected: `main.pdf`, 4 pages, letter/journal format, no unresolved refs.

## Asset Provenance

| File | Source |
|---|---|
| `main.tex` | NEW (does not inherit stale `paper/CAL-4page/` skeleton) |
| `tables/table_complementarity.tex` | Adapted from `paper/tcasi/`; last row relabeled APT+EffViT-B1 |
| `tables/table_hp_sweep.tex` | Copied from `paper/tcasi/` |
| `tables/table_apt_2x2.tex` | Adapted from `paper/tcasi/table_tm_wcaps_2x2.tex` (TM→density, W-CAPS→precision); SceneFlow column dropped to fit single column |
| `tables/table_main_results.tex` | NEW; KITTI + SceneFlow only |
| `figures/fig_arch_cal.tex` | NEW simplified arch (SA hub + APT spoke + external PKRN) |
| `figures/fig_sa_utilization.tex` | NEW; **placeholder bars** — TODO real data |
| `figures/fig_apt_quadrant.tex` | NEW; hp-sweep inset uses real `table_hp_sweep` numbers |
| `figures/fig_cosim_heatmap.tex` | NEW; **placeholder heatmap** — TODO real data |

## Open TODOs (before submission)

1. **Fig. 2 SA utilization** — replace placeholder bar values with measured per-phase PE utilization from `simulator/run_simulator.py` output (`simulator/results/simulation_results.json`, `per_engine_stats.*.busy_cycles / total_cycles`). One simulator run, no GPU.
2. **Fig. 4 joint SA × APT heatmap** — run the 3×4 sweep (SA ∈ {16, 32, 48} × APT ∈ {off, density-only, precision-only, full}). Driver: parameterize `simulator/run_simulator.py` over SA side length and APT config; collect KITTI D1 + FPS. ~1–2 GPU days. Without this, the co-design claim has no data; the placeholder is flagged in red in the tex.
3. **`table_apt_2x2.tex`** — two `$\ddagger$` cells are trend-estimated; end-to-end measurement outstanding (shared with TCAS-I).
4. **Reference list** — currently 19 entries; trim Eyeriss-v2, Gemmini, FPStereo first if page budget overflows.
5. **Contingency trims** if overflow: (a) merge Sec. VII into Sec. VI tail, (b) drop hp-sweep inset from Fig. 3.

## Relationship to other drafts

- `paper/tcasi/` — full 12-page journal submission; this CAL letter is a strict subset + re-pitch around SA + APT.
- `paper/CAL-4page/` — stale pre-pivot skeleton (2026-04-21, 5-SCU framing). Not touched; delete once this draft is stable.
