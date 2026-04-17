# SGM-ViT

SGM-ViT is a research codebase for stereo-guided monocular depth estimation.
It combines classical Semi-Global Matching (SGM), Depth Anything V2, token
merge, decoder adaptive precision, and confidence-guided fusion into one
algorithm-system co-design workflow.

The current default mainline is:

- `Merge FP32`
- `Merge + decoder W-CAPS INT4`
- `edge_aware_residual` fusion

This repository contains code, paper drafts, and self-generated figures. It
does not include large datasets, pretrained checkpoints, or experiment outputs.

## Overview

![SGM-ViT architecture](paper/figures/fig1_architecture.png)

The software pipeline is:

`stereo pair -> SGM disparity + PKRN confidence -> DA2 / merge / W-CAPS -> disparity alignment -> fusion`

At a high level, the repository studies three linked ideas:

- `confidence as a systems signal`: the same SGM confidence map is reused for
  alignment, routing, and fusion
- `token merge instead of hard pruning`: reduce ViT attention cost without
  removing spatial positions from dense prediction
- `decoder weight-aware adaptive precision`: keep selected decoder regions in
  higher precision while quantizing the rest

## Highlights

- Unified four-dataset evaluation for `KITTI`, `ETH3D`, `Middlebury`, and
  `SceneFlow`
- Merge-centric demo with GT comparison and shared-scale disparity rendering
- SGM precomputation pipeline for ETH3D / SceneFlow / Middlebury
- FusionNet training and evaluation hooks for lightweight learned fusion
- Hardware and simulator code for architecture-level analysis

## Current Results Snapshot

The repository currently tracks merge + adaptive precision as the main story.
Representative fused results from the latest local summaries are:

| Dataset | Protocol | Dense DA2 + align | Best Merge FP32 | Best W-CAPS INT4 |
| --- | --- | --- | --- | --- |
| KITTI | fused EPE / D1 | `1.9420 / 15.99` | `1.9568 / 16.38` | `1.9962 / 16.94` |
| ETH3D | fused EPE / D1 | `0.8329 / 21.85` | `0.8311 / 22.25` | `0.8374 / 22.08` |
| Middlebury | fused all-valid EPE / bad2 | `2.1115 / 25.93` | `2.1202 / 25.73` | `2.1270 / 25.83` |
| SceneFlow | fused EPE / bad3 | `4.8440 / 39.65` | `4.9042 / 38.77` | `5.4118 / 41.85` |

The main qualitative takeaway so far is:

- merge is consistently safer than hard token pruning for dense prediction
- decoder-side low precision is noticeably more tolerant than encoder-side
  approximation
- confidence-guided fusion remains the strongest accuracy lever

## Important Research Finding

One negative result is already established and should be treated as part of the
project record:

- hard `token pruning` is not safe for dense DA2 prediction; it noticeably
  hurts dense-only quality, and fusion only partially hides that damage

Because of that, pruning is treated as an archived route rather than an active
entry point. The route history, negative findings, and representative results
are summarized in `paper/prior_experiment.md`.

## Repository Layout

```text
SGM-ViT/
|- core/                  # inference, fusion, merge, adaptive precision, viz
|- scripts/               # dataset eval, sweeps, demos, preprocessing
|- SGM/                   # classical stereo implementation
|- hardware/              # hardware-oriented modeling code
|- simulator/             # architecture and system analysis utilities
|- paper/                 # paper drafts and self-generated figures
|- asserts/               # small bundled demo assets
|- demo.py                # merge + W-CAPS demo entry point
|- pyproject.toml
`- requirements.txt
```

## What Is Not Included

The following are intentionally excluded from version control:

- datasets
- pretrained checkpoints
- local experiment outputs under `results/`
- local backup artifacts under `artifacts/`
- third-party reference PDFs under `paper/ref/`

## Setup

Python `3.9+` is expected.

Install dependencies with either:

```bash
pip install -e .
```

or:

```bash
pip install -r requirements.txt
```

The code assumes local access to:

- the `Depth-Anything-V2` source tree at `./Depth-Anything-V2`
- DA2 checkpoints under `Depth-Anything-V2/checkpoints/`
- stereo datasets such as KITTI / ETH3D / SceneFlow / Middlebury

Common environment variables:

- `SGMVIT_KITTI_ROOT`
- `SGMVIT_ETH3D_ROOT`
- `SGMVIT_SCENEFLOW_ROOT` (default: `.../sceneflow_official/extracted/driving`)
- `SGMVIT_MIDDLEBURY_ROOT`
- `SGMVIT_DA2_WEIGHTS`

## Dataset Preparation

The evaluation and training scripts expect four stereo datasets stored
on the NFS share.  Only SceneFlow setup is documented here in detail;
KITTI, ETH3D, and Middlebury follow standard community layouts.

### SceneFlow (Driving subset)

The official SceneFlow dataset is hosted at
<https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>.

Three subsets are available: **Driving**, **Monkaa**, and **FlyingThings3D**.
Currently only the Driving subset (35mm / scene_forwards / fast, 300 frames)
is used by the code.

After downloading and extracting the Driving tarballs, the directory layout
should be:

```text
sceneflow_official/extracted/driving/       <- SGMVIT_SCENEFLOW_ROOT
  frames_cleanpass/
    35mm_focallength/scene_forwards/fast/{left,right}/*.png
  frames_finalpass/                          (not used by current code)
  disparity/
    35mm_focallength/scene_forwards/fast/left/*.pfm
  camera_data/                               (not used by current code)
  sgm_hole/                                  (pre-computed, see below)
    35mm_focallength/scene_forwards/fast/left/
      35mm_focallength_scene_forwards_fast_left_<NNNN>.pfm
      35mm_focallength_scene_forwards_fast_left_<NNNN>_mismatches.npy
      35mm_focallength_scene_forwards_fast_left_<NNNN>_occlusion.npy
```

The `sgm_hole/` directory is **not** part of the official download.
Generate it with:

```bash
python scripts/precompute_sgm_hole.py --dataset sceneflow
```

The default root path is
`/nfs/usrhome/pdongaa/dataczy/sceneflow_official/extracted/driving`.
Override with `SGMVIT_SCENEFLOW_ROOT` or the `--sceneflow-root` CLI flag.

The Monkaa and FlyingThings3D subsets are extracted alongside Driving at
`sceneflow_official/extracted/{monkaa,flyingthings3d}/` for future use.

## Quick Start

Run the merge + W-CAPS demo:

```bash
python demo.py
```

Inspect demo options:

```bash
python demo.py --help
```

Run the unified four-dataset evaluation:

```bash
python scripts/eval_merge_adaptive.py --dataset all
```

Useful examples:

```bash
python scripts/eval_merge_adaptive.py --dataset kitti
python scripts/eval_merge_adaptive.py --dataset sceneflow --max-samples 50
python scripts/run_four_dataset_demos.py
python scripts/precompute_sgm_hole.py --dataset all
```

## Main Entry Points

- `demo.py`: merge + W-CAPS visualization pipeline
- `scripts/eval_merge_adaptive.py`: default four-dataset evaluation entry
- `scripts/run_four_dataset_demos.py`: generate one demo per dataset
- `scripts/precompute_sgm_hole.py`: rebuild SGM caches for ETH3D / SceneFlow / Middlebury
- `scripts/eval_fusion_net.py` and `scripts/train_fusion_net.py`: lightweight learned fusion experiments

## Scope And Limitations

This is a research prototype, not a polished benchmark release. Expect:

- path assumptions tied to local dataset storage
- external dependency on `Depth-Anything-V2`
- limited automated testing
- evolving evaluation protocols as research directions change

The goal of this repository is to preserve the current algorithm and paper
workflow in a reproducible codebase, while keeping heavyweight assets outside
the repository itself.
