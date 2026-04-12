# SGM-ViT

SGM-ViT is a research codebase for combining classical stereo matching with
Depth Anything V2. The current mainline focuses on two algorithm-system
directions:

- `token merge`: reduce ViT attention sequence length without deleting spatial
  positions
- `decoder weight-aware adaptive precision (W-CAPS)`: keep selected decoder
  regions in higher precision while using low-bit weights elsewhere

The project also studies confidence-guided fusion between SGM disparity and
aligned monocular disparity, plus an FPGA-oriented hardware/simulator story.

## Current Status

This repository is a research prototype, not a packaged release. The main
software path is working end to end:

`stereo pair -> SGM disparity + PKRN confidence -> DA2 / merge / W-CAPS -> disparity alignment -> fusion`

Current merged results stored locally show:

- `KITTI`: fused dense and fused merge are both strong; best merge is close to
  dense fused quality while reducing attention cost
- `ETH3D` and `Middlebury`: merge is consistently better than aggressive token
  removal for dense prediction
- `SceneFlow`: fusion helps, but the dataset remains much harder than KITTI and
  ETH3D

One negative result is already established and should be treated as part of the
project record:

- hard `token pruning` is not safe for dense DA2 prediction; it noticeably
  hurts dense-only quality, and fusion only partially hides that damage

Because of that, the default research storyline in this repository is now:

- `Merge FP32`
- `Merge + decoder W-CAPS INT4`
- `edge_aware_residual` fusion

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

Not versioned in this repository:

- datasets
- pretrained checkpoints
- experiment outputs under `results/`
- local backup artifacts under `artifacts/`
- local reference PDFs under `paper/ref/`

## Environment Setup

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

Several scripts also read dataset roots from environment variables:

- `SGMVIT_KITTI_ROOT`
- `SGMVIT_ETH3D_ROOT`
- `SGMVIT_SCENEFLOW_ROOT`
- `SGMVIT_MIDDLEBURY_ROOT`
- `SGMVIT_DA2_WEIGHTS`

## Main Entry Points

### Demo

Run the merge + W-CAPS demo:

```bash
python demo.py
```

The demo supports:

- GT disparity visualization when provided
- shared-scale disparity rendering across `SGM / GT / aligned / fused`
- heuristic or lightweight network-based fusion backends

Inspect available flags with:

```bash
python demo.py --help
```

### Unified Four-Dataset Evaluation

The default evaluation entry point is:

```bash
python scripts/eval_merge_adaptive.py --dataset all
```

It evaluates:

- `SGM`
- `Dense DA2 + align`
- `Merge FP32`
- `Merge + decoder W-CAPS INT4 hp25 / hp50 / hp75`

Current default fusion is:

- `edge_aware_residual`

Useful examples:

```bash
python scripts/eval_merge_adaptive.py --dataset kitti
python scripts/eval_merge_adaptive.py --dataset sceneflow --max-samples 50
python scripts/run_four_dataset_demos.py
```

### Dataset-Specific Utilities

The repository also includes scripts for:

- SGM precomputation for ETH3D / SceneFlow / Middlebury
- fusion ablations
- confidence analysis
- decoder precision studies
- FusionNet training and evaluation
- architecture and paper figure generation

## Notes On Scope

This repository intentionally keeps some older branches of the research history
for reference, including pruning-related scripts. They are no longer the
default storyline, and new results should be interpreted through the merge +
adaptive precision pipeline first.

If you clone this repository on a new machine, expect to provide your own:

- dataset storage
- model checkpoints
- optional precomputed SGM caches

The repository should contain the code and paper materials needed to understand
and reproduce the workflow, but not the heavy assets themselves.
