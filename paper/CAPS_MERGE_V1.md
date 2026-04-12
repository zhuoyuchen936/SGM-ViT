# CAPS-Merge-v1

## Goal

Use the same SGM/PKRN confidence prior to jointly schedule:

1. token routing via confidence-guided merge,
2. arithmetic precision on the merged representatives,
3. final SGM-DA2 fusion.

The motivation is straightforward. Current results already show:

- hard pruning damages dense DA2 severely,
- merge is consistently better than pruning,
- INT8 is stable, while aggressive INT4 causes visible loss,
- the decoder is still the main bottleneck, so encoder-only sparsity is not enough.

This suggests the next step should not be "more pruning". It should be
"preserve spatial structure with merge, then spend high precision only where the
geometry prior says it is still necessary."

## Reference Mapping

### Revolver

Most relevant idea:

- precision should follow sensitivity, not remain fixed end-to-end,
- high precision is reserved for sensitive phases,
- low precision is used for robust phases.

Mapping to SGM-ViT:

- replace temporal phases with spatial confidence groups,
- replace phase-aware precision with confidence-aware precision,
- replace "prefill vs decode" with "low-confidence/edge groups vs high-confidence flat groups".

### MEGA.mini

Most relevant idea:

- most data can use cheap low precision,
- only a small outlier subset needs the expensive path.

Mapping to SGM-ViT:

- most merge representatives can use low precision,
- only sensitivity-ranked representatives should keep the high-precision path.

### Matryoshka Quantization

Most relevant idea:

- one model should support multiple operating points instead of storing fully separate variants.

Mapping to SGM-ViT:

- longer term, CAPS should evolve toward a single nested-precision checkpoint,
- short term, we first validate that the confidence prior can pick the right regions for high precision.

## CAPS-Merge-v1 Algorithm

### Step 1: Merge

Build merge groups exactly as in the current confidence-guided token merge:

- keep `K = round(N * keep_ratio)` representative tokens,
- choose the `K` lowest-confidence tokens as representatives,
- assign each remaining token to its nearest representative on the token grid.

This preserves the full decoder grid while shortening the attention sequence.

### Step 2: Group Sensitivity

For each merge group `g`, compute:

- `mean_conf(g)`
- `std_conf(g)`
- `conf_range(g) = max_conf(g) - min_conf(g)`
- `mean_radius(g)` = average normalized distance from members to representative

Define the sensitivity score:

`score(g) = w1*(1 - mean_conf(g)) + w2*std_conf(g) + w3*conf_range(g) + w4*mean_radius(g)`

Default weights in the prototype:

- `w1 = 1.0`
- `w2 = 0.5`
- `w3 = 0.5`
- `w4 = 0.25`

Interpretation:

- low confidence means SGM is unreliable, so DA2 needs more precision,
- high variance or range indicates a likely boundary/mixed region,
- larger radius means the group is spatially coarser and more likely to lose detail.

### Step 3: Precision Assignment

Choose the top `ceil(K * hp_ratio)` groups by sensitivity as high precision.

Prototype precision assignment:

- high-sensitivity groups: `INT8`
- remaining groups: `INT4`

Prefix tokens use the high-precision path.

### Step 4: Merge Attention with Adaptive Precision

For each transformer block after `merge_layer`:

1. gather `CLS + representatives`,
2. fake-quantize representative activations by assigned precision,
3. run standard attention,
4. fake-quantize attention outputs again by assigned precision,
5. scatter the representative outputs back to the full token grid,
6. run FFN densely.

Important:

- this prototype only adapts the attention activation precision on representatives,
- it does not yet implement weight-level dual precision or decoder adaptive precision,
- it is meant to validate the routing signal, not to claim final hardware numbers.

## Why This Is Better Than Plain Mixed Precision

Static mixed precision answers:

`Which layers are usually robust?`

CAPS answers:

`Which spatial groups in this image are robust right now?`

That is a much better fit for SGM-ViT because the stereo confidence prior is already
spatial and sample-dependent.

## Prototype Scope

The prototype should compare:

- dense FP32 baseline,
- merge FP32 baseline,
- merge with uniform low precision on all representatives,
- merge with confidence-aware adaptive precision.

Primary metric:

- dense-only EPE / D1 after alignment.

Secondary metric:

- fused EPE / D1.

Fixed operating point for the first prototype:

- `keep_ratio = 0.814`
- `merge_layer = 0`
- `encoder = vits`
- `input_size = 518`
- `high_precision_bits = 8`
- `low_precision_bits = 4`

## Expected Outcomes

### Positive outcome

If adaptive precision improves over uniform low precision at the same keep ratio,
then the confidence prior is not only useful for routing but also useful for
precision allocation.

That would justify a stronger paper claim:

`A single stereo confidence prior jointly schedules token routing, arithmetic precision, and output fusion.`

### Negative outcome

If adaptive precision does not improve over uniform low precision, then either:

- confidence alone is not enough for precision scheduling,
- the score needs boundary-aware features from DA2 itself,
- or activation-only quantization is too weak a proxy and weight-level adaptive precision is needed.

That would still be useful, because it tells us where the current co-design boundary is.

## Current Prototype Result

20-image pilot, fixed `keep_ratio = 0.814`:

| Config | Rep Count | HP Count | Precision Proxy | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
|--------|-----------|----------|-----------------|-----------|--------------|-----------|--------------|
| Dense DA2 + align | 1369 | 1369 | 1.0000 | 2.1183 | 20.3378 | 1.9403 | 17.1010 |
| Merge FP32 | 1114 | 1114 | 1.0000 | 2.2804 | 22.7984 | 1.9845 | 17.7505 |
| Merge uniform INT8 proxy | 1114 | 1114 | 1.0000 | 2.2776 | 22.7213 | 1.9777 | 17.6146 |
| Merge uniform INT4 proxy | 1114 | 0 | 0.5000 | 4.9292 | 49.2395 | 3.0197 | 27.2703 |
| CAPS-Merge hp=25% | 1114 | 278 | 0.6248 | 3.4132 | 35.3633 | 2.3935 | 21.5472 |
| CAPS-Merge hp=50% | 1114 | 557 | 0.7500 | 2.8921 | 30.0311 | 2.1957 | 19.4927 |

Immediate interpretation:

- uniform INT8 is almost identical to merge FP32, so the prototype itself is numerically stable at 8-bit,
- uniform INT4 is far too destructive for dense prediction,
- CAPS is clearly better than uniform INT4 at the same keep ratio,
- increasing the high-precision fraction from 25% to 50% gives a strong recovery,
- but CAPS still does not close the gap to merge FP32.

This means the confidence prior is useful for precision allocation, but the current
activation-only INT8/INT4 prototype is still too aggressive to serve as the final algorithm.
The next step should therefore focus on either:

- a less aggressive low-precision target,
- weight-level residual precision instead of pure activation fake quantization,
- or decoder-aware adaptive precision where the larger cycle budget exists.

## Next Hardware-Oriented Extensions

### Extension A: Weight-level CAPS

Store:

- low-precision base weights,
- a compact high-precision residual for sensitive groups/layers.

This is the closest SGM-ViT analogue to the MPRE idea in Revolver.

### Extension B: Decoder CAPS

Apply the same confidence prior to decoder feature tiles:

- low-confidence or high-gradient regions stay INT8,
- high-confidence smooth regions use INT4.

This is the most promising path for real end-to-end speedup because the decoder is the current bottleneck.

### Extension C: Nested Precision Checkpoint

Train or fine-tune one nested-precision checkpoint so that deployment can move between:

- dense INT8,
- merge INT8,
- merge + CAPS,
- future decoder CAPS,

without maintaining fully separate model copies.
