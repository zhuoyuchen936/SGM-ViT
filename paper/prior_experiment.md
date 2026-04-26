# Prior Experiments and Eliminated Routes

## Overview

This note records the routes explored before the current mainline converged.
Its purpose is twofold:

- preserve negative results and transition-stage findings for paper writing;
- explain why some older code paths were removed from the active repository.

The current accepted direction is:

- `token merge` instead of hard pruning;
- `decoder-aware W-CAPS` as the current precision co-design line;
- `edge_aware_residual` heuristic fusion as the default output fusion;
- `FusionNet` only as a constrained search branch, not as the default backend.

The ongoing learned-fusion search after this convergence point is tracked in
`paper/fusion_net_experiment_log.md`.

## Current Accepted Direction

The working project story is now:

1. `merge` is the safe sequence-compression direction for dense DA2-style prediction.
2. `decoder-side precision control` is materially safer than encoder-side aggressive precision reduction.
3. `heuristic fusion` is still the strongest stable deployment choice.
4. `learned fusion` remains exploratory and must beat heuristic fusion on both visual quality and boundary-aware metrics before promotion.

## Eliminated Directions

### 1. Hard Token Pruning

**Initial goal**

Use SGM confidence to remove high-confidence tokens early and cut ViT attention cost.

**Method**

- build a token-level confidence map from SGM / PKRN;
- prune high-confidence tokens;
- optionally rely on fusion to hide monocular degradation.

**Representative result**

| Route | Protocol | Dense-only result | Fused / system result | Interpretation |
| --- | --- | --- | --- | --- |
| Dense DA2 + align | KITTI default operating point | `EPE 2.7095` | reference | dense baseline |
| Hard pruning (`theta=0.65`) | KITTI default operating point | `EPE 4.0178` | fusion only partially recovers | clear dense regression |
| Hard pruning | hardware estimate | `18.6%` token pruning, `33.8%` attention reduction | compute improves, prediction quality degrades | not safe for dense output |

**Why it was eliminated**

- DPT-style dense prediction needs spatially unique token features.
- Hard token removal breaks this assumption.
- Fusion can hide some damage in high-confidence regions, but that does not make pruning safe for dense monocular prediction.

**Paper wording**

`Hard token pruning harms dense monocular depth quality; confidence-guided fusion only partially recovers end-task performance by replacing damaged regions with SGM output.`

### 2. Two-Pass Sparse / Hole-Filling Recovery

**Initial goal**

Recover quality lost by pruning through a second pass or explicit hole repair.

**Method**

- first pass on kept tokens;
- second pass or extra recovery path for pruned tokens;
- optional reassembly / hole filling to repair missing context.

**Representative result**

| Route | Protocol | Result | Interpretation |
| --- | --- | --- | --- |
| Two-pass GAS / recovery path | pilot ablation | small accuracy gain at clear extra compute / control complexity | poor tradeoff |

**Why it was eliminated**

- the extra pass weakens the original compute-saving argument;
- quality recovery was too small relative to added complexity;
- merge solves the dense-structure problem more directly.

**Paper wording**

`Recovery-by-second-pass was explored but offered limited quality gain relative to its added complexity, and was superseded by merge-based dense-preserving compression.`

### 3. Encoder-Side Adaptive Precision / Activation Proxy

**Initial goal**

Use the same confidence prior to assign different precision budgets to merged representative tokens.

**Method**

- build merge groups;
- score groups by confidence-derived sensitivity;
- fake-quantize representative activations to `INT8/INT4`.

**Representative result**

20-image pilot from `CAPS_MERGE_V1.md`:

| Config | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Dense DA2 + align | `2.1183` | `20.3378` | `1.9403` | `17.1010` |
| Merge FP32 | `2.2804` | `22.7984` | `1.9845` | `17.7505` |
| Merge uniform INT8 proxy | `2.2776` | `22.7213` | `1.9777` | `17.6146` |
| Merge uniform INT4 proxy | `4.9292` | `49.2395` | `3.0197` | `27.2703` |
| CAPS-Merge hp=25% | `3.4132` | `35.3633` | `2.3935` | `21.5472` |
| CAPS-Merge hp=50% | `2.8921` | `30.0311` | `2.1957` | `19.4927` |

**Why it was eliminated**

- the confidence prior is useful, but encoder activation-only INT4 is still too destructive;
- even adaptive high-precision masking did not recover merge FP32;
- the route was valuable as a sensitivity-study prototype, not as the final algorithm.

**Paper wording**

`Confidence-aware precision scheduling is promising, but encoder-side activation proxy quantization remained too aggressive to serve as the final dense-prediction path.`

### 4. Early High-Freedom FusionNet

**Initial goal**

Replace heuristic fusion with a lightweight learned residual network.

**Method**

- take aligned mono disparity, SGM disparity, fused base, confidence, validity, disagreement, and detail cues;
- predict a residual correction over heuristic fusion.

**Representative result**

| Route | Protocol | Result | Interpretation |
| --- | --- | --- | --- |
| Early unconstrained FusionNet | smoke-stage multi-dataset experiments | sometimes improved global metrics, but often worsened `boundary_epe`, flat-region noise, and visual edge cleanliness | not stable enough to replace heuristic fusion |

**Why it was eliminated**

- the network had too much freedom in anchor regions where heuristic fusion was already stable;
- visual quality regressed through edge pollution and noisy interiors;
- the problem is not solved by simply giving the network more capacity.

**Paper wording**

`Learned residual fusion requires stronger update-region control and supervision design; unconstrained residual prediction degraded boundary fidelity even when some scalar metrics improved.`

## Absorbed Transitional Directions

### 1. CAPS-Merge v1

This route is not the final algorithm, but it contributed two lasting ideas:

- confidence should schedule precision spatially, not only layer-wise;
- group sensitivity is a useful abstraction once merge groups already exist.

It is now treated as a historical bridge from pruning-era thinking to the current decoder-aware precision line.

### 2. Decoder Coarse INT4 + High-Precision Mask

This route produced the key architectural finding that the decoder is more tolerant than the encoder, especially on coarse stages.

Representative results from `DECODER_CAPS_V1.md`:

| Config | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Merge FP32 | `2.2815` | `22.6471` | `1.9809` | `17.6641` |
| Decoder coarse INT6 proxy | `2.2899` | `22.8032` | `1.9833` | `17.7227` |
| Decoder coarse INT4 proxy | `2.4859` | `25.6197` | `2.0490` | `18.6254` |
| Decoder coarse INT4 CAPS hp=50% | `2.3661` | `23.7907` | `1.9938` | `17.8464` |
| Decoder coarse INT4 CAPS hp=75% | `2.3015` | `22.5928` | `1.9789` | `17.4629` |

This route is now absorbed into the `decoder-aware W-CAPS` mainline story rather than kept as a separate branch name.

### 3. Constrained FusionNet-v1b

This route did not become the default backend, but it contributed necessary discipline to the learned-fusion search:

- stronger residual clipping;
- stronger anchor-region protection;
- explicit logging of `boundary_epe`, `flat_region_noise`, and residual magnitude.

It is best understood as a search constraint baseline, not as the final fusion method.

## Representative Results

### Dense / Prune / Merge

| Route | Protocol | Key numbers |
| --- | --- | --- |
| Dense DA2 + align | KITTI dense-only baseline | `EPE 2.7095` |
| Hard pruning (`theta=0.65`) | KITTI dense-only default point | `EPE 4.0178` |
| Hard pruning (`theta=0.65`) | hardware-side operating point | `18.6%` token pruning, `33.8%` attention reduction |
| Merge FP32 | 20-image pilot | `Dense EPE 2.2804`, `Fused EPE 1.9845` |

### Encoder Precision Route

| Route | Protocol | Key numbers |
| --- | --- | --- |
| Merge uniform INT8 proxy | 20-image pilot | nearly identical to merge FP32 |
| Merge uniform INT4 proxy | 20-image pilot | `Dense EPE 4.9292`, clearly destructive |
| CAPS-Merge hp=50% | 20-image pilot | improves over uniform INT4, but still behind merge FP32 |

### Decoder Precision Route

| Route | Protocol | Key numbers |
| --- | --- | --- |
| Decoder all INT4 proxy | 20-image pilot | `Dense EPE 2.7004`, `Fused EPE 2.1198` |
| Decoder coarse INT4 CAPS hp=50% | 20-image pilot | `Dense EPE 2.3661`, `Fused EPE 1.9938` |
| Decoder coarse INT4 CAPS hp=75% | 20-image pilot | almost recovers merge FP32 and slightly improves fused metrics |

### Heuristic Fusion vs Learned Fusion

| Route | Protocol | Key numbers / status |
| --- | --- | --- |
| `edge_aware_residual` heuristic fusion | current mainline | strongest stable default backend |
| Early unconstrained FusionNet | smoke-stage pilot | qualitative only: edge pollution and flat-region noise regressions |
| Constrained FusionNet-v1b | smoke-stage pilot | more stable than early net, but still not a consistent heuristic replacement |

## Takeaways for Paper

The paper should now follow these conclusions consistently:

1. `Pruning harms dense prediction.` Dense-only quality must be analyzed separately from fused end-task quality.
2. `Merge is the correct compression direction.` It preserves spatial positions and is structurally better matched to dense DPT-style decoding.
3. `Decoder-side precision reduction is gentler than encoder-side precision reduction.` The coarse decoder is the first viable low-bit target.
4. `Learned fusion is not yet solved.` A residual fusion network needs stronger update-region modeling and better teacher/supervision design before it can replace heuristic fusion.

In other words, the final paper should present pruning, two-pass recovery, encoder activation proxy quantization, and early learned fusion as necessary but discarded stages of the design search, not as parallel equally-viable options.
