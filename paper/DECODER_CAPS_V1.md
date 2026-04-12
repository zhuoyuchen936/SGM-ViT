# Decoder-CAPS-v1

## Motivation

The current SGM-ViT bottleneck is no longer the encoder alone. The DPT decoder still
accounts for most of the cycle budget, and the previous merge experiments already
showed that preserving token structure is necessary for dense prediction.

This suggests a second-stage co-design question:

`Can the same stereo prior also decide where decoder feature maps still need high precision?`

## Core Idea

Keep the encoder in merge FP32 mode, then apply spatially adaptive precision inside the
decoder:

- sensitive spatial regions use high precision,
- reliable or smooth regions use low precision.

Unlike the previous CAPS-Merge prototype, this version targets the actual runtime
bottleneck in the pipeline.

## Why Decoder-Aware Precision Helps

Encoder routing protects global token interactions, but the decoder determines how much
local geometric detail survives into the final depth map.

This is especially important for the failure mode observed in stereo:

- an object may have rich texture,
- SGM may still flatten its internal relief,
- fusion then over-trusts the flattened stereo estimate,
- the final map loses fine intra-object shape.

Therefore, decoder precision should not depend on confidence alone. It should also
preserve image regions likely to contain fine internal structure.

## Sensitivity Map

Decoder-CAPS-v1 builds a spatial sensitivity score:

`score = w_conf * (1 - conf) + w_tex * texture + w_var * local_variance`

where:

- `conf` is the PKRN confidence,
- `texture` is the image gradient magnitude,
- `local_variance` is a local intensity variance map.

Interpretation:

- `1 - conf` preserves difficult stereo regions,
- `texture` protects local relief and boundaries,
- `local_variance` helps avoid smoothing away repeated micro-structure inside objects.

## Spatial Precision Assignment

At each decoder stage:

- resize the sensitivity map to the current feature-map resolution,
- select the top `hp_ratio` pixels as the high-precision region,
- fake-quantize the remaining pixels to low precision.

Prototype settings:

- high precision: `INT8`
- low precision: `INT4`
- fixed keep ratio: `0.814`
- encoder: merge FP32

## Experimental Questions

The prototype is meant to answer three questions:

1. Is decoder adaptive precision numerically more stable than encoder-only adaptive precision?
2. Is confidence-only scheduling enough?
3. Does adding texture/detail cues better preserve intra-object geometry?

## Expected Interpretation

If `conf+texture` beats `confidence-only` at the same decoder precision budget, then:

- stereo confidence alone is not enough for fine geometry,
- a detail-aware auxiliary cue is needed,
- and the best story for the paper becomes:

`The stereo prior schedules reliability, while the image-detail prior preserves local shape.`

That is stronger than a pure SGM-confidence routing story because it explains how to
avoid the well-known stereo over-smoothing failure inside objects.

## If It Works

The next paper-level algorithm should likely become:

1. confidence-guided merge in the encoder,
2. detail-aware adaptive precision in the decoder,
3. detail-aware fusion that trusts DA2 residuals inside textured high-confidence objects.

## If It Fails

If decoder-CAPS still does not help enough, the likely issue is that fake activation
quantization is too weak a proxy. Then the next step should shift to:

- weight-level adaptive precision in decoder convolutions,
- or explicit residual fusion that lets DA2 contribute only the high-frequency disparity component.

## Current Prototype Result

20-image pilot, fixed `keep_ratio = 0.814`:

| Config | Precision Proxy | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
|--------|-----------------|-----------|--------------|-----------|--------------|
| Dense DA2 + align | 1.0000 | 2.1023 | 19.7227 | 1.9323 | 16.8056 |
| Merge FP32 | 1.0000 | 2.2792 | 22.1252 | 1.9776 | 17.4454 |
| Decoder uniform INT8 proxy | 1.0000 | 2.2885 | 22.6608 | 1.9821 | 17.6810 |
| Decoder uniform INT4 proxy | 0.5000 | 2.6898 | 29.3354 | 2.1227 | 19.8247 |
| Decoder CAPS conf hp=50% | 0.7498 | 2.5002 | 25.9051 | 2.0130 | 18.0132 |
| Decoder CAPS conf+tex hp=50% | 0.7498 | 2.5325 | 26.2126 | 2.0302 | 18.1657 |

Interpretation:

- decoder-side low precision is much less destructive than encoder-side uniform INT4,
- confidence-aware decoder precision clearly improves over uniform decoder INT4,
- but it still does not recover merge FP32,
- naive `confidence + texture` is slightly worse than confidence-only.

This last point is important. It suggests that plain image texture is not the right proxy
for the "SGM flattens intra-object relief" failure mode. A better trigger is likely a
`detail disagreement` signal rather than texture alone.

## Better Detail-Preservation Trigger

The current texture cue is too broad because textured regions are not always geometrically
meaningful. A more targeted next-step trigger should be:

`detail_score = texture_energy - stereo_relief_energy`

where:

- `texture_energy` comes from image gradients or local variance,
- `stereo_relief_energy` comes from local disparity gradients or curvature in SGM.

If texture is high but stereo relief is low, that is exactly the "SGM flattened the object
interior" case. Those regions should receive:

- higher decoder precision,
- or stronger DA2 residual contribution during fusion.

This is a better next direction than simply increasing the texture weight.

## Stage Sweep Result

20-image pilot, fixed `keep_ratio = 0.814`:

| Config | Precision Proxy | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
|--------|-----------------|-----------|--------------|-----------|--------------|
| Dense DA2 + align | 1.0000 | 2.0971 | 19.7694 | 1.9264 | 16.7988 |
| Merge FP32 | 1.0000 | 2.2675 | 21.3637 | 1.9729 | 17.0809 |
| Decoder all INT4 proxy | 0.5000 | 2.7004 | 29.7484 | 2.1198 | 19.8598 |
| Decoder coarse INT4 proxy | 0.7500 | 2.4731 | 25.5968 | 2.0496 | 18.6112 |
| Decoder fine INT4 proxy | 0.7500 | 2.6002 | 27.5872 | 2.1003 | 19.3220 |
| Decoder all CAPS hp=50% | 0.7500 | 2.4999 | 25.8739 | 2.0162 | 18.1015 |
| Decoder coarse CAPS hp=50% | 0.8750 | 2.3789 | 23.8042 | 2.0141 | 18.0573 |
| Decoder fine CAPS hp=50% | 0.8750 | 2.4383 | 25.2106 | 2.0025 | 17.9361 |

The main finding is:

- decoder-side low precision is indeed much gentler than encoder-side low precision,
- but the decoder is not uniformly tolerant,
- lowering precision in the coarse decoder is safer than lowering precision in the fine decoder,
- the best current tradeoff is `coarse CAPS hp=50%`, not the all-stage or fine-stage variant.

This matters for the next hardware story. The right question is no longer
"can the decoder be quantized?" but rather:

`which decoder stages can absorb low precision without destroying dense geometry?`

The current evidence suggests that lower-resolution coarse decoder stages are the
better first target, while the high-resolution fine decoder and output head remain more sensitive.

## Coarse Precision Sweep Result

To turn the stage-sweep observation into a practical design rule, we fixed the
stage policy to `coarse_only` and swept both the low-precision bit-width and
the high-precision spatial ratio.

20-image pilot, fixed `keep_ratio = 0.814`:

| Config | Precision Proxy | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
|--------|-----------------|-----------|--------------|-----------|--------------|
| Dense DA2 + align | 1.0000 | 2.1194 | 20.1670 | 1.9410 | 17.0249 |
| Merge FP32 | 1.0000 | 2.2815 | 22.6471 | 1.9809 | 17.6641 |
| Decoder coarse INT6 proxy | 0.8750 | 2.2899 | 22.8032 | 1.9833 | 17.7227 |
| Decoder coarse INT6 CAPS hp=25% | 0.9062 | 2.3034 | 23.3233 | 1.9913 | 17.9288 |
| Decoder coarse INT6 CAPS hp=50% | 0.9375 | 2.3056 | 23.1338 | 1.9890 | 17.8496 |
| Decoder coarse INT6 CAPS hp=75% | 0.9688 | 2.2839 | 22.4968 | 1.9817 | 17.5853 |
| Decoder coarse INT4 proxy | 0.7500 | 2.4859 | 25.6197 | 2.0490 | 18.6254 |
| Decoder coarse INT4 CAPS hp=25% | 0.8125 | 2.4500 | 25.0197 | 2.0243 | 18.2664 |
| Decoder coarse INT4 CAPS hp=50% | 0.8750 | 2.3661 | 23.7907 | 1.9938 | 17.8464 |
| Decoder coarse INT4 CAPS hp=75% | 0.9375 | 2.3015 | 22.5928 | 1.9789 | 17.4629 |

This sweep sharpens the decoder story considerably:

- `coarse INT6` is already almost indistinguishable from `Merge FP32` on dense-only metrics.
- `coarse INT4` is too aggressive when applied uniformly, but recovers well once the high-precision mask reaches `50%` to `75%`.
- At matched proxy `0.875`, static `INT6` is still better than adaptive `INT4`, which means bit-width selection matters more than spatial adaptation once the low-bit path becomes too coarse.
- At proxy `0.9375`, `coarse INT4 CAPS hp=75%` nearly recovers `Merge FP32` and even edges it on fused metrics.

The main engineering conclusion is now:

`The coarse decoder is the first viable adaptive-precision target.`

More specifically:

- a mild bit-width drop (`INT6`) on coarse decoder stages is already close to lossless,
- an aggressive bit-width drop (`INT4`) needs spatial protection,
- and the fine decoder should not be the primary quantization target.

This is a much stronger and cleaner result than the encoder-side adaptive-precision
prototype, where even confidence-aware precision could not recover the merge baseline.

## Revised Next Step

The next decoder-focused experiment should therefore not be another broad stage sweep.
It should move one level closer to a paper-quality algorithm:

1. keep `encoder = merge FP32`,
2. quantize only `coarse decoder stages`,
3. use `INT6` as the stable default operating point,
4. treat `INT4 + spatial protection` as the aggressive operating point,
5. only after that consider weight-level or kernel-level realization.

This reframes the decoder contribution from:

`Can adaptive precision help the decoder at all?`

to:

`How far can coarse decoder stages be pushed before dense geometry starts to break?`

## Weight-Aware Coarse INT4 Result

The previous decoder experiments still used activation-side fake quantization.
To move closer to a hardware-plausible interpretation, we added a dual-path
prototype for coarse decoder stages:

- FP32 branch: original decoder kernels
- low-precision branch: weight-only PTQ decoder copy (`INT4`)
- spatial mixer: a confidence-guided high-precision mask chooses which branch
  survives at each coarse-stage output

This is not yet a fused custom kernel, but it is a more realistic proxy than
only quantizing activations after each stage.

20-image pilot, fixed `keep_ratio = 0.814`, `stage_policy = coarse_only`:

| Config | Precision Proxy | Dense EPE | Dense D1 (%) | Fused EPE | Fused D1 (%) |
|--------|-----------------|-----------|--------------|-----------|--------------|
| Dense DA2 + align | 1.0000 | 2.1091 | 20.1242 | 1.9368 | 16.9971 |
| Merge FP32 | 1.0000 | 2.2926 | 22.8386 | 1.9855 | 17.7455 |
| Decoder coarse INT4 activation proxy | 0.7500 | 2.4755 | 25.6735 | 2.0477 | 18.6418 |
| Decoder coarse INT4 activation CAPS hp=75% | 0.9375 | 2.2857 | 22.5648 | 1.9751 | 17.4593 |
| Decoder coarse INT4 weight-aware proxy | 0.7500 | 2.3027 | 22.7942 | 1.9828 | 17.7890 |
| Decoder coarse INT4 weight-aware CAPS hp=50% | 0.8750 | 2.2828 | 22.0645 | 1.9723 | 17.3541 |
| Decoder coarse INT4 weight-aware CAPS hp=75% | 0.9375 | 2.2828 | 22.5873 | 1.9771 | 17.6124 |

This is the strongest decoder result so far.

The key observation is not just that coarse decoder quantization is tolerable.
It is that the more realistic weight-aware implementation remains very close to
`Merge FP32`, and in fact outperforms the earlier activation-only proxy at the
same coarse `INT4` operating points.

That matters for the paper story because it means the earlier decoder result was
not merely an artifact of activation-side fake quantization.

The current best tradeoff is:

`coarse INT4 weight-aware CAPS hp=50%`

because it slightly improves over `Merge FP32` on dense-only D1 and fused D1
while still operating at a reduced coarse-stage precision budget.

At this point, the main decoder claim can be sharpened to:

`Coarse decoder stages can absorb INT4 weight quantization when protected by a confidence-guided high-precision mask.`

This is much stronger than the original encoder-side adaptive-precision result,
and is a plausible foundation for a future kernel-level realization.
