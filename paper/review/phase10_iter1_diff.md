# Phase 10, Iter 1 Review Response — Change Summary

Date: 2026-04-20
Author: anonymous (reviewer-response cycle 1)
Target: TCAS-I / DAC 2026 submission (`paper/tcasi/main.tex` + `paper/tcasi/main.pdf`)

Compile status: pdflatex × 2, exit 0. Page count 11 → 12. No undefined refs, no overfull hbox > 10 pt, residual Overfull is a 10.4 pt `\vbox` on the float-only p.8 (qualitative demos) which is structural and not a layout bug.

## Blocking fixes (4/4 addressed)

### B1 — FPS / latency / power contradictions (Tab II vs Tab III)
Files: `table_sota_accel.tex` (full rewrite), `main.tex` Tab~II (`tab:perf`)

- Rewrote Tab III into 3 clearly-labelled rows:
  - `Ours: B0-h24 INT8 (v1 pipeline, 518×518)`  → 172 ms / 5.81 FPS / 165 mW core
  - `Ours: B1-h24 INT8 (FusionEngineV2, 384×768)` → 25 ms / 40.0 FPS / 172 mW core
  - `Ours: Heuristic (FusionEngineV2, 384×768)` → 21.57 ms / 46.4 FPS / 141 mW core
- Power units now per-row with suffix (`mW (core)` vs `W (board)`) and a row-level dagger footnote explaining that FPGA rows are full-board power whereas ASIC rows are core power only. LPDDR4 and SGM engine power explicitly excluded.
- Removed the 140 W ASIC typo (was the 140 mW value reported in W).
- Added new column "Resolution" so 518×518 and 384×768 rows are never visually conflated.
- Tab II (`tab:perf`) reheaded to match: column headers now include (v1, 518×518) vs (FE-V2, 384×768); all ms/FPS numbers match Tab III; dual-report line for core vs SGM-inclusive area/power added to the static budget block; DAG event count and FE op count separated (86 FE ops vs ~2100 full-DAG events).

### B2 — Fig. 1 redraw (`fig_arch_v2.tex`)
- Rewrote the top-level architecture figure as a single-row 5-box pipeline: `SGM → CRM → GSU+DPC → ADCU → FusionEngineV2 → Metric Disparity`.
- Added PKRN "confidence fan-out" wire (dashed red) above the pipeline with explicit tag, fanning out to CRM, GSU, ADCU, FusionEngineV2.
- Added "Memory Subsystem" band below: DRAM → WeightStreamer → L2 → Strip Buffer → Unified SA, backed by a light gray fit-box and explicit 3 GB/s / 256 b labels.
- NEW modules (FusionEngineV2, WeightStreamer) use the red 0.5 mm bold border requested by the reviewer; `newStroke=#b85450`.
- All labels at ≥ `\footnotesize` (8 pt in the final PDF at the figure\*-width scale).
- Fig caption rewritten to drop the duplicated execution-order equation; now ≤70 words and references `Eq.~\ref{eq:order}` instead.
- Fixed a TikZ compile bug: reserved key `out` renamed to `outp`; nested `{\scriptsize ...\\...}` rewritten as `\\scriptsize ...\\scriptsize ...` (the nested form crashed the TeX-Live 2026 TikZ 3.1.11 parser on 3+ consecutive nodes).

### B3 — Amdahl argument restatement (§VIII.A, `sec:amdahl`)
- Split into two explicit points: (1) v1 pipeline encoder-sparsity ceiling with the corrected equation `S_max(s_e) = 1 / (0.3/s_e + 0.7)` — the old `(1-1)` typo is gone; (2) FusionEngineV2 decoder-throughput rebalance explanation.
- Stated new (encoder, decoder) wall-clock split as ~(40%, 60%) in FusionEngineV2 mode, computed from the summary.json measurements (SA 2.5 M, FusionEngine 10.0 M, WS 0.7 M cycles; pipelined total 12.5 M).
- Made the "Amdahl does not apply to the reformulation" claim explicit and un-hand-wavy.
- Contribution bullet in §I reworded: the "5.8 FPS per-frame ceiling" line replaced by a pointer to §VIII.A.

### B4 — LiteMLA precision path (§V, `sec:litemla_precision`)
- New subsection between "Two Execution Modes" and "Weight Layout" describing the SA reconfiguration into 8×8 FP16 sub-arrays (64 FP16 MACs at ~1/4 INT8 throughput) when executing LiteMLA.
- Notes the ~0.63 MB FP32 LiteMLA weight footprint and a dedicated 1 MB FP weight SRAM (0.08 mm² inside the 1.87 mm² core).
- Confines mixed-precision to ~8% of frame cycles.
- Cross-referenced from §III.D (sec:qat) so the two mentions are now consistent: "retained in FP16/FP32… reconfigures as described in §V".
- §IV.E die-area breakdown updated to list the 0.08 mm² FP buffer explicitly and to state the 2.17 mm² SGM-inclusive total.

## Non-blocking fixes applied

- **Issue 5 (area dual-report)**: abstract and Tab II both now report "1.87 mm² core / 2.17 mm² incl. SGM engine" and "172 mW core / 202 mW incl. SGM"; §IV.E breakdown matches.
- **Issue 7 ("8/8 dominance")**: abstract, §I third-contribution bullet, and §VII.A all reworded to "strict 8/8 dominance over the hand-crafted heuristic baseline; 6/8 metrics also improve over FP32 under INT8 QAT".
- **Fig 13 caption**: "31 mW is power not energy" wording fixed — now "31 mW average power (0.78 mJ/frame at 40 FPS)".
- **"Phase 10" internal references**: stripped from §IV.D header ("FusionEngineV2" not "FusionEngineV2 (Phase 10 Addition)") and from the Fig 1 caption ("new additions in this work" not "Phase-10 additions"). The `phase10` label remains only inside simulator JSON, which is not user-facing.
- **PKRN 0.5 vs 0.65 threshold**: clarified in §II.A bullet (i): 0.5 is the liberal-coverage threshold used only in Fig. 2; 0.65 is the strict high-confidence threshold used throughout the rest of the paper.
- **Fig 3 "70 ops" vs §VI.C "2100-op DAG"**: 70 replaced with the measured 86 FusionEngine ops; body text now explicitly says "86 FusionEngine ops (full per-frame DAG including SA-side events ~2100)".

## Not touched (per instructions)

- Issue 6 (silicon / post-PnR / FPGA measurement) — acknowledged as future work, no new data invented.
- Issue 8 (ViTCoD / SpAtten rows in Tab III) — skipped; no new external data fabricated.

## Remaining residual warnings

- 1× `Overfull \vbox (10.39 pt too high)` on p.8 — page 8 contains only the `fig_qualitative_demo.pdf` float, which is inherent to a large wide figure. Not a layout bug; IEEE accepts float-only pages.
- 2× "Text page 8 contains only floats" (same cause, benign).
- 4× font warnings for `OT1/ptm/m/scit` (IEEEtran's small-caps italic fallback) — benign, no visual effect.

## Files changed

- `paper/tcasi/main.tex`  (495 → 502 lines)
- `paper/tcasi/table_sota_accel.tex`  (27 lines, fully rewritten)
- `paper/tcasi/fig_arch_v2.tex`  (114 → 103 lines, fully rewritten)
- `paper/tcasi/fig_arch_v2.pdf`  (recompiled, 42.4 KB)
- `paper/tcasi/main.pdf`  (1.55 MB, 12 pages)

Backups preserved at `main.tex.bak_iter1`, `table_sota_accel.tex.bak_iter1`, `fig_arch_v2.tex.bak_iter1`.
