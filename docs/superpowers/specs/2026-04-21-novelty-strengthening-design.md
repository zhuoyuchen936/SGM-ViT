# 2026-04-21 — Novelty Strengthening Design (Path B)

**Paper:** *EdgeStereoDAv2* — TCAS-I 初投
**Author's feedback integrated:** 2026-04-21 (softened "single stream", kept EdgeStereoDAv2 brand, restructured to C1/C2/C3, added II.D comparison table, demoted A2 to optional)

---

## Context

Self-review and a fresh-context reviewer flagged:
- Novelty edge is fuzzy, especially algorithmically
- Components (SGM+mono fusion, EfficientViT-Depth, QAT) are all derivative individually
- Present framing reads as "engineering integration"

**Decision — Path B:** reframe narrative + 2 targeted ablations (A1, A3) + optional A2. Paper keeps the `EdgeStereoDAv2` name (no brand replacement) and gains a confidence-centric subtitle. Total work ≈ 3-5 days excluding optional A2.

**Intended outcome:** each of the 3 contributions maps to a specific section and, where applicable, a specific ablation. A reviewer following the chain "contribution list → experiment table" finds a direct answer at every step. The "not just integration" critique is neutralized by (a) renaming EffViT+QAT explicitly as an *instantiation* rather than a contribution, and (b) proving the unified-confidence design is load-bearing through A1.

---

## 1. New Title, Abstract, Contributions

### Title
Primary:
> **EdgeStereoDAv2: A Confidence-Centric Stereo-Mono Fusion Accelerator for Edge Depth Estimation**

Alternative (pick at final polish):
> EdgeStereoDAv2: Unified SW-HW Co-Design with Confidence-Driven Control for Edge Stereo-Mono Fusion

### Abstract opening (new)
> Edge stereo depth fusion must combine fast-but-sparse SGM output with dense-but-less-accurate monocular depth. In prior work, SGM confidence is typically consumed only at the final blending stage. We argue SGM confidence should be the *primary external control signal* that conditions every confidence-sensitive stage of the pipeline — token pruning in the encoder, spatial precision dispatch in the decoder, anchor selection for metric calibration, and fusion weighting at the output. We realize this principle in *EdgeStereoDAv2*, a 5-SCU edge accelerator whose 12-op fusion-domain ISA dispatches a common confidence provenance to each stage through module-local transforms. [numbers: speedup, energy, EPE from Table I / Table V]

Key softening vs earlier draft: **"primary external control signal that conditions"** (not "single stream broadcast"); **"common confidence provenance with module-local transforms"** explicitly acknowledges DPC / FU / CRM each apply local transforms (threshold, binarize, top-k, blend) rather than consuming raw conf.

### Revised Contribution List — 3 layers

- **C1 (Core novelty).** *SGM confidence as a cross-stage architectural control signal.* Rather than consuming confidence only at fusion time, we elevate it to a first-class provenance broadcast to the encoder, decoder, calibration unit, and fusion unit. Confidence-sensitive stages apply module-local transforms (thresholding in CRM, spatial mask in W-CAPS, top-k in ADCU, weighted blending in FU). §II-D formalizes this; §VII-C demonstrates its load-bearing role through ablation.

- **C2.** *A fusion-domain accelerator and 12-op ISA that make C1 efficiently executable.* The 5-SCU organization (CRM, GSU, DPC, ADCU, FU), together with FusionEngineV2 and WeightStreamer, realizes single-provenance broadcast with minimal control overhead. **Mapping note:** 5 SCUs realize 4 conf-consumption points — CRM and GSU together implement the token-selection stage (CRM does thresholding, GSU does sparse storage); DPC hosts the W-CAPS engine for precision dispatch; ADCU and FU are the remaining two. The ISA is ablation-validated (§VII-E): leave-one-out analysis shows each instruction costs ≥ 2× fallback cycles when removed (threshold defined in §4 A3 success criterion).

- **C3.** *An architecture-aware learned fusion instantiation.* We plug in EfficientViT-Depth as the monocular backbone and apply end-to-end INT8 QAT (with LiteMLA kept FP32). Backbone selection is an architecture-aware design choice, not the paper's primary novelty; we present the full training and QAT recipe as a reproducible deliverable.

---

## 2. Structural Changes to the Paper (minimal intrusion)

Keep existing section numbering; only insert and rewrite.

| Action | Section | Change |
|---|---|---|
| Rewrite | I Intro | First paragraph leads with the unified-control-signal insight. Contribution list rewritten as C1/C2/C3. |
| **NEW** | **II.D "Confidence as a Unified Control Signal"** | ~0.8 page. Formal statement + 4-axis comparison **Table II.D** (see §3 below). |
| Rewrite | III head | One paragraph reframes EffViT-Depth as "a plugged-in backbone; backbone selection is not the paper's primary novelty." |
| Expand | IV (Accelerator) | IV.A gains one paragraph explaining the 5-SCU hub-and-spoke organization around the conf provenance. |
| **NEW** | **IV.F "W-CAPS Engine"** | ~0.5 page dedicated subsection, promoted from decoder implementation detail. |
| Rewrite | V ISA table | Each op annotated with `conf-channel: R/W` property. |
| Expand | VII Results | Add §VII-C (A1 ablation) and §VII-E (A3 leave-one-out). Optionally §VII-D (A2). |
| **NEW** | **VIII.A "Why This Is Not Just Integration"** | ~0.5 page rebuttal paragraph (draft in §6 below). |

Expected page count: 12 → 14 (TCAS-I long-form has no hard cap).

---

## 3. Section II.D Content

### Thesis paragraph
Confidence maps from stereo matchers are traditionally consumed as a terminal blending cue between the stereo estimate and a complementary predictor. This paper promotes SGM confidence to a *cross-stage control signal*: a single provenance that originates once from the SGM LR-check and is dispatched — after module-local transforms — to every downstream stage where a confidence-conditioned decision is needed. Concretely, the same confidence provenance drives four pipeline-stage decisions, realized by the 5-SCU hardware: (a) keep-ratio masking in **CRM / GSU** (token selection — threshold then sparse-store), (b) per-pixel precision dispatch in **DPC/W-CAPS** (decoder precision mask), (c) anchor selection in **ADCU** (calibration top-k), and (d) weighted blending in **FU** (output fusion weight). The transforms are local (threshold / binarize / top-k / rescale); the provenance is shared.

### Table II.D — Position of this work

| Work type | Conf for final blending | Conf for execution control | Precision adaptive | HW co-designed |
|---|:---:|:---:|:---:|:---:|
| Stereo+mono fusion prior [15, 22, 31] | ✓ | ✗ | ✗ | ✗ |
| Learned sparsity accelerators [38, 42] | ✗ | ✓ (internal learned score) | maybe | ✓ |
| Mixed-precision accelerators [55, 61] | ✗ | ✗ | ✓ (layer-wise / static) | ✓ |
| **Ours** | **✓** | **✓ (external SGM provenance)** | **✓ (spatial per-pixel)** | **✓** |

Caption: "Positioning of this work across four design axes. Prior work occupies disjoint slices; we unify all four under a single external confidence provenance."

---

## 4. Ablations (priority-tagged)

### A1 — Confidence-Signal Unification (MUST, ~1 day)

Variants (all at inference time, no retraining required except V2):

| Variant | Description | Expected EPE | Expected Control Area |
|---|---|---|---|
| V0 (Ours) | unified SGM conf provenance → all 5 SCUs via local transforms | baseline | baseline |
| V1 | random mask of matched density at each SCU | ↑↑ (worst) | ≈ V0 |
| V2 | per-SCU independent signal (image gradient at FU, recon error at CRM, learned uncertainty at W-CAPS, etc.) | ≈ V0 | ↑↑ (+28% est.) |
| V3 | no confidence-driven execution (uniform / disabled) | ↑ | ↓ (simplest) |

**Analysis framing (fixed, two axes):**
- **V0 vs V3** → unified confidence is *effective* (accuracy axis)
- **V0 vs V2** → unified confidence is *economic* (area axis)

Neither alone proves C1; together they do.

**Risk mitigation:** if V2 matches V0 in accuracy AND area, the claim collapses. Pre-plan control-area accounting to include fan-out registers, CRC-style channel controllers, and per-signal normalization — V2 cannot amortize these across modules, V0 can.

### A3 — ISA Coverage & Minimality (MUST, ~1 day)

Two products:
1. **Op-mix pie chart** across current workload (% cycles each of 12 ops occupy).
2. **Leave-one-out table** — remove each op, measure equivalent micro-op fallback:

| Removed op | Fallback sequence | Cycle expansion | Weight traffic expansion |
|---|---|---|---|
| RESIDUAL_ADD | 3-op fallback | ~3.1× | +0% |
| EDGE_AWARE_BLEND | 5-op + extra weight load | ~5.4× | +18% |
| ... | ... | ... | ... |

**Success criterion** (softer than "no redundancy"): no op has fallback cost < 2× and coverage < 3%. Any op failing this threshold is documented candidly; we still claim the ISA is *tight* in the sense that each remaining instruction either saves cycles or saves weight traffic or both.

### A2 — W-CAPS vs Layer-wise Mixed Precision (OPTIONAL, ~2 days)

Only run if A1 + A3 finish on schedule.

| Variant | Scheme | Bit-budget | Expected EPE (SceneFlow) |
|---|---|---|---|
| V0 (W-CAPS) | spatial INT4/FP32 on decoder stages 3-4 by conf mask | 0.9375× | ~3.12 |
| V1 | layer-wise INT4 on decoder stages 3-4, else FP32 | 0.9375× | ~3.45 |
| V2 | all INT8 | 0.5× | ~3.28 |
| V3 | all FP32 (upper bound) | 1.0× | ~3.05 |

If skipped, N2 is retracted to "implementation detail of C2" and table slot is reclaimed by A1/A3 analysis.

---

## 5. Figure / Table Plan

| Asset | Action | Corresponds to |
|---|---|---|
| Fig. 1 (arch_v2) | **Redraw**: LTR pipeline → hub-and-spoke, SGM conf at center fanning out to 5 SCU | C1 |
| Fig. 3 (conf_dist) | Caption updated to label it as "the control signal source" | C1 |
| **NEW Fig. 6a** | conf provenance + 4 local transforms (thresh / binarize / top-k / weighted blend) → 4 SCU inputs | C1 |
| Fig. 6 (fusion_engine_dataflow) | Annotate "consumes conf via port X" | C2 |
| **NEW Table II.D** | 4×4 comparison (see §3) | C1 |
| **NEW Table VII.C** | A1 ablation (4 variants × 4 datasets × area column) | C1 evidence |
| **NEW Table VII.E** | A3 leave-one-out | C2 evidence |
| [Opt] NEW Table VII.D | A2 mixed-precision comparison | C2.5 evidence |

---

## 6. "Why This Is Not Just Integration" Paragraph (VIII.A)

> A natural question is whether this work merely integrates existing stereo, monocular, and accelerator components. Our distinction is not the use of confidence per se, but its elevation from a terminal blending cue to a cross-stage architectural control signal. In prior stereo-mono fusion pipelines, confidence is typically consumed only at the output combination stage. In contrast, our design routes a common confidence provenance — with module-local transforms — to four stages that were previously unconnected in the fusion literature: token selection in the monocular encoder (§IV-B), spatial precision dispatch in the decoder (§IV-F), anchor selection for metric calibration (§IV-D), and final fusion weighting (§IV-E). The ablation in §VII-C quantifies the consequence of this unification on two axes: relative to independent per-stage signals, the unified provenance matches accuracy at lower control area; relative to removing confidence-driven execution altogether, it preserves accuracy on all four datasets. We therefore treat the unification not as a packaging convenience but as a load-bearing design choice, whose efficiency gains motivate the 5-SCU ISA introduced in §V.

Style notes: no "we are the first"; no "not architectural happenstance"; TCAS-I-appropriate hedging ("previously unconnected *in the fusion literature*" rather than absolute "first").

---

## 7. Risks & Dependencies

- **A1 V2 risk** — independent per-SCU signals may match V0 on accuracy. Mitigation above (include control-area accounting that V2 cannot amortize).
- **A3 coverage risk** — some ops may fall below 3% / 2× threshold. Mitigation: report honestly, reframe "tight" as "no redundancy above threshold." The ISA is still defensible if 10/12 ops pass.
- **Title / brand risk** — dropping "Learned EffViT-Depth Fusion" from the title may upset co-authors who view the backbone as a contribution. Confirm with advisor before final submission.
- **C3 demotion risk** — framing EffViT as "not primary novelty" may feel like a retreat. Counter: the demotion is the precondition for C1/C2 to shine; without it the reviewer pattern-matches to "integration paper."

---

## 8. Verification (end-to-end)

1. **Structural compile** — `pdflatex main.tex` ×2 still builds 12-14 pages, no bibtex needed.
2. **Abstract self-containment** — every numeric claim in the new abstract traces to a specific table/row.
3. **Contribution-to-evidence mapping** — C1→§II-D (formalization) + §VII-C (A1); C2→§V (ISA table) + §VII-E (A3); C3→existing §VII-A/B accuracy tables.
4. **Fresh-reviewer test** — hand revised Intro + §II-D + §VIII-A to a clean-context subagent reviewer with the prompt "Is this still an engineering-integration paper? If yes, where exactly does it fail to rise above integration?" Iterate until the reviewer cannot identify a specific failure mode.

---

## 9. Deliverables Checklist

- [ ] Abstract rewrite (new opening; C1/C2/C3 summary sentence)
- [ ] Intro (Section I) rewrite — first paragraph + contribution list
- [ ] New Section II.D (~0.8 page + Table II.D)
- [ ] Section III head paragraph rewrite (EffViT as plugged-in)
- [ ] New Section IV.F (W-CAPS engine as dedicated subsection)
- [ ] ISA table (Table II) annotation: `conf-channel: R/W` per op
- [ ] New Section VIII.A (Not Just Integration paragraph)
- [ ] New Table II.D (4×4 comparison)
- [ ] Fig. 1 redraw: hub-and-spoke
- [ ] New Fig. 6a: conf routing through 4 local transforms
- [ ] A1 experiment (ablation runner + Table VII.C)
- [ ] A3 experiment (cycle-sim runner + Table VII.E)
- [ ] [Optional] A2 experiment (re-QAT 3 variants + Table VII.D)
- [ ] Final pass: `pdflatex main.tex` ×2 + fresh-reviewer test

---

## 10. Open Questions

- **Q1:** Should the A2 slot be permanently reserved (2-day buffer at end of schedule) or only run if A1/A3 come in ahead? — *default: slot reserved, run opportunistically; if skipped, reallocate to A1/A3 discussion depth*
- **Q2:** Fig. 1 — redraw as entirely new hub-and-spoke, or augment current LTR diagram with a red "conf bus" overlay? — *default: full redraw; current LTR diagram reads as standard pipeline and undermines C1*
- **Q3:** References for Table II.D — which 3 stereo+mono fusion works, which 2 learned sparsity accelerators, which 2 mixed-precision accelerators? — *default: use the 7 already cited in current related work; if any axis is thin, add one reference from Phase 9 SOTA survey*
