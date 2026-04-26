# TCAS-I 2026 Paper — EdgeStereoDAv2

IEEE TCAS-I / DAC 2026 submission: **EdgeStereoDAv2 — A Unified Accelerator with Learned EffViT-Depth Fusion for Edge Stereo**.

## Build

```bash
cd /home/pdongaa/workspace/SGM-ViT/paper/tcasi
export PATH=/home/pdongaa/.local/bin:$PATH
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex    # second pass for refs
```

No bibtex run required — bibliography is inlined via `\begin{thebibliography}` at the end of `main.tex`.

Output: `main.pdf` (12 pages, ~1.5 MB).

## Layout

### Top-level
| File | Role |
|---|---|
| `main.tex` | Paper source (~500 lines, IEEE journal format) |
| `main.pdf` | Compiled PDF (12 pages, 14 figures, 5 tables) |
| `IEEEtran.bst` | IEEE BibTeX style (kept for reference even if unused) |
| `table_sota_accuracy.tex` | Full-width SOTA accuracy comparison (14 rows) |
| `table_sota_accel.tex` | Accelerator efficiency comparison (7 rows) |

### Figures (14 total embedded in paper)
| File | What | Source |
|---|---|---|
| `fig_arch_v2.pdf` | F1 overall accelerator architecture | TikZ `fig_arch_v2.tex` |
| `fig_design_flow.pdf` | F2 algorithm-DSE-hardware co-design flow | TikZ `fig_design_flow.tex` |
| `fig_conf_dist.pdf` | F3 SGM PKRN confidence distribution | `gen_fig_conf_dist.py` |
| `fig_pareto.png` | F4 6-variant Pareto (params/GFLOPs vs avg EPE) | copy of `results/phase8_pareto/pareto_plot.png` |
| `fig_qat_delta.pdf` | F5 INT8 QAT vs FP32 delta bar chart | `gen_fig_qat_delta.py` |
| `fig_fusion_engine_dataflow.pdf` | F6 FusionEngineV2 internal dataflow (4 sub-cores + 16×3×3 PE grid) | TikZ `fig_fusion_engine_dataflow.tex` |
| `fig_weight_streamer.pdf` | F7 WeightStreamer + DRAM hierarchy + timing | TikZ `fig_weight_streamer.tex` |
| `fig_qualitative_demo.pdf` | F8 4-dataset × 5-method qualitative grid | `gen_fig_qualitative_demo.py` |
| `fig_sota_scatter.pdf` | F9 params-vs-D1 SOTA scatter | `gen_fig_sota_scatter.py` |
| `fig_roofline.png` | F10 roofline model | reused |
| `fig_fps_resolution.png` | F11 FPS vs resolution | reused |
| `fig_energy.png` | F12 energy breakdown | reused |
| `fig_area.png` | F13 die area breakdown | reused |
| `fig_ablation.png` | F14 ablation study | reused |

### Figure sources
- `fig_*.tex` (4 files): TikZ sources, each standalone-compilable via `xelatex`
- `gen_fig_*.py` (4 files): Matplotlib generation scripts that read from `/home/pdongaa/workspace/SGM-ViT/results/` and write PDF here

### `template/`
IEEE template reference material (not used in build):
- `IEEEtran_how-to.{tex,pdf}` — official IEEEtran usage guide
- `template_original.tex`, `template_preview.pdf` — original template distribution

## Regenerate figures

If `results/phase8_pareto/` or `results/phase9_sota/` data changes, regenerate matplotlib figures:

```bash
cd /home/pdongaa/workspace/SGM-ViT/paper/tcasi
python3 gen_fig_conf_dist.py
python3 gen_fig_qat_delta.py
python3 gen_fig_qualitative_demo.py
python3 gen_fig_sota_scatter.py
```

TikZ figures (`fig_*.tex`) compile standalone:

```bash
xelatex fig_arch_v2.tex             # → fig_arch_v2.pdf
xelatex fig_design_flow.tex
xelatex fig_fusion_engine_dataflow.tex
xelatex fig_weight_streamer.tex
```

## Review history

Located in `../review/phase10_iter{1,2,3}_*.md`:
- **Iter 1** review: 8 issues (4 blocking); fixed in `phase10_iter1_diff.md`
- **Iter 2** review: 1 blocking + 5 consistency issues; fixed in `phase10_iter2_diff.md`
- **Iter 3** review: "conditional-yes / submission-ready"; final polish applied

## History

This folder was renamed from `paper/cal/` to `paper/tcasi/` on 2026-04-20 as the target venue changed from IEEE Computer Architecture Letters to IEEE TCAS-I full journal paper.
