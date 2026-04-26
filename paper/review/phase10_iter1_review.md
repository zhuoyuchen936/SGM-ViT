# Review — Iter 1 (2026-04-20)

Reviewer role: senior IEEE TCAS-I / DAC 2026 reviewer
Paper: /home/pdongaa/workspace/SGM-ViT/paper/tcasi/main.pdf (495 lines, 11 pages, 14 figs, 5 tables)

## Overall impression
Solid, thoughtful algorithm-hardware co-design paper with an honest narrative (Amdahl discussion, capability-density reframing, in-house-split caveats). However, **not yet publishable**: Table II and the SOTA hardware table (Tab. III) directly contradict the abstract's headline 40 FPS / 25 ms claim, the top architecture figure is unreadable at print size, and several numerical and mode-of-execution statements do not reconcile across sections. Fixable in one iteration, but must be fixed.

## Strengths
- Complementarity framing (PKRN-stratified SGM vs DA2) well motivated; Table I is concrete
- Section 8.1 Amdahl → "capability density" reframing is refreshingly honest
- Explicit protocol footnote (in-house splits, not KITTI test server) avoids overclaim trap
- 12-op ISA table crisp; dual-mode story architecturally clean
- INT8 QAT ablation (Fig. 5, 6/8 improving over FP32) convincing if numbers reconcile

## Priority issues (ranked)

### Issue 1 (MAJOR): Headline FPS/latency contradicts SOTA hardware table
- Location: Abstract + Tab. II vs Tab. III (tab:sota_hw), Sec. 7.3
- Abstract says 40.0 FPS / 25 ms; Tab III says 4.75 FPS / 210 ms, 140 W(!)
- Power column units mixed W vs mW → **unit typo** for ASIC rows
- Fix: Re-align Tab III rows to the 384×768 FusionEngineV2 operating point. Separate "v1 pipeline" from "FusionEngineV2 main". Fix mW/W units. Add footnote for FPGA-board vs ASIC-core power.

### Issue 2 (MAJOR): Fig. 1 unreadable + too dense
- Location: Fig. 1 (fig_arch_v2.pdf), page 3
- Labels collide; caption duplicates Eq. 2; reviewer skims this first
- Fix: Redraw as single-row LTR pipeline (SGM → CRM → GSU+DPC → ADCU → FusionEngineV2), PKRN fan-out wire above, memory subsystem band below. Drop duplicate equation. Labels ≥8pt.

### Issue 3 (MODERATE): Amdahl argument contradicts itself
- Location: Sec. 8.1, Eq. 3
- Claims decoder=70% cycles → 1.43× ceiling, then says FusionEngineV2 "circumvents this ceiling by relocating 74% of decoder MACs." Amdahl is not about WHICH unit runs MACs.
- Fix: Restate: "FusionEngineV2 is not encoder-sparsity opt; it is decoder-throughput opt that re-balances the (encoder, decoder) split from (30%, 70%) to (X%, Y%)." Give the actual new split.

### Issue 4 (MODERATE): LiteMLA precision path unclear
- Location: Sec. 3.4 vs Sec. 5.1
- Sec 3.4 says LiteMLA FP32; Sec 5.1 says SA handles LiteMLA. 32×32 INT8 SA can't do FP32 matmul natively.
- Fix: Add paragraph explaining: dedicated FP16 lane in SA at 1/4 throughput, or offload to host, or add FP SRAM. Update area.

### Issue 5 (MODERATE): 1.87 mm² excludes SGM engine (hidden in footnote)
- Fix: Report both "1.87 mm² core / 2.17 mm² with SGM" in abstract + Tab II. Same for power.

### Issue 6 (MODERATE): No silicon, no FPGA measurement
- Fix: Add post-PnR (not just synthesis) number for FusionEngineV2/WeightStreamer; or commit to FPGA validation in revision plan.

### Issue 7 (MINOR): "8/8 dominance" phrasing slippery
- Fig 5 shows 2/8 INT8 regresses vs FP32. "8/8" only vs heuristic.
- Fix: "strict 8/8 dominance over hand-crafted heuristic baseline; 6/8 metrics also improve over FP32 under INT8 QAT"

### Issue 8 (MINOR): ViTCoD/SpAtten differentiation not quantitative
- Fix: Add ViTCoD and SpAtten rows to Tab III at comparable ViT-S workload (published numbers OK).

## Minor polish
- Fig 13 caption: "31 mW is power, not energy" — reword
- Strip "Phase N" internal phase numbers from final text
- Abstract says 1.69–2.3 mm², body says 1.87 — pick one
- Eq. 3 renders as "(1 - 1)" typo — fix
- "70 ops" (Fig 3) vs "2100-op DAG" (Sec 6.3) — clarify per-MBConv vs full frame
- Hyphenation: "cycle accurate" → "cycle-accurate", "stage sequential" → "stage-sequential"
- PKRN thresholds 0.5 vs 0.65 inconsistent

## Verdict
- Blocking: 4 (Issues 1, 2, 3, 4)
- Nice-to-have: 4
- One more iter → clear-accept territory for TCAS-I major revision
