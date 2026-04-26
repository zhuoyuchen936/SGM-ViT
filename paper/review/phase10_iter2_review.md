# Review — Iter 2 (2026-04-20)

Second-round reviewer, fresh context. Paper now 12 pages.

## Blocking (1)

### Issue 1: Table II (`table_sota_accel.tex`) rows broken by single `\` instead of `\\`
- LaTeX parser emits errors: `Misplaced \noalign`, `Missing number`, `Illegal unit of measure`
- Table II mis-renders in the compiled PDF
- Fix: replace every line-ending single `\` with `\\` throughout `table_sota_accel.tex`

## Moderate/Minor (5)

### Issue 2: Heuristic KITTI baseline inconsistency (1.84 vs 1.87 EPE; 14.97% vs 15.08% D1)
- Tab I complementarity: 14.97% / 1.84; abstract + Tab III + Sec VII.A: 15.08% / 1.87
- Fix: pick one and propagate; or distinguish "original" vs "tuned SGM" heuristic explicitly

### Issue 3: "Phase N" / wrong-venue labels leak through figures
- `fig_design_flow.tex`: boxes "EffViT-Depth (Phase 7)", "Pareto (Phase 8)", "INT8 QAT (Phase 9)", "FusionEngineV2 (Phase 10)"; venue box "ICCAD / DAC 2026"
- `fig_arch_v2.tex` legend: "NEW (Ph.10)"
- `gen_fig_qualitative_demo.py` & `gen_fig_qat_delta.py`: suptitle contains "Phase 8 Pareto sweet-spot" / "Phase 9 vs Phase 8"
- Fix: replace Phase-N with descriptive labels (e.g. "Pareto search", "INT8 QAT", "unified FusionEngineV2"); set venue to "TCAS-I 2026" or remove

### Issue 4: SCU area fraction contradictory (3% vs 7.9% vs 8.8%)
- 0.148 / 1.87 = 7.9%; Sec IV.C + Tab III line 336 say 8.8%; abstract + Sec IX.A say 3%
- Fix: unify to 7.9% (or recompute). Abstract "three capabilities for 3% area" is numerically wrong.

### Issue 5: LiteMLA FP32 weight size inconsistent (0.63 MB vs 1.5 MB vs "4.85 MB conv")
- Sec V.B: 0.63 MB FP32, 1 MB FP SRAM (0.08 mm²)
- Sec V.C: 2.88 MB INT8 + 1.5 MB FP32 = 4.4 MB per frame
- Sec VIII.C: 1.5 MB FP32 attention
- Abstract: "4.85 MB conv weights hidden by WeightStreamer"
- Fix: measure real FP32 LiteMLA size (from actual effvit_qat.py output), unify all three. Explain if "4.85 MB" = per-frame streamed (INT8 conv + FP32 attn).

### Issue 6: FPS/W mislabeled (Tab III) + Sec VIII.C 70% limitation contradicts Sec VIII.A
- 46.4 FPS / 141 mW ≈ 329 FPS/W for heur; 40.0/172 ≈ 233 for B1 INT8. Table shows "233 / 232" — heur value wrong.
- Sec VIII.C says 70% DPT decoder cycle share is structural ceiling, but Sec VIII.A just said v2 rebalanced to 60%
- Fix: recompute heur FPS/W; clarify limitation refers to v1 baseline, v2 is 60%

## Verdict
- Blocking: 1
- Minor: 5
- Ready for submission? NO — needs 1 more iter focused on Issue 1 (Table row bug) + content sweep for Issues 2-6
