"""Phase 10 — End-to-end pipeline wrapper.

Chains 6 stages:
  Stage 0: DMA load (stereo images + conf + weights)
  Stage 1: SGM compute
  Stage 2: DA2 ViT-S encoder
  Stage 3: DA2 decoder (either custom RefineNet OR EffViT backbone + FPN head, depending on config)
  Stage 4: Mono-SGM alignment (robust LS)
  Stage 5: Fusion (either heuristic FU OR EffViT-head FPN)

Two workload variants:
  'baseline':  encoder + custom decoder + heuristic FU
  'effvit_bX_h24':  encoder + EffViT backbone replacement + FPN head (FE) — for DA2+EffViT-fusion
                    actually EffViT replaces the full decoder+fusion path.

The returned WorkloadDAG carries the whole pipeline end-to-end.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator.core.workload_dag import WorkloadDAG
from simulator.core.event_simulator import build_workload as build_baseline_workload
from simulator.core.workload_effvit import build_effvit_dag
from simulator.core.event_simulator import SimConfig


# ---------------------------------------------------------------------------
# SGM cycle model (approximation)
# ---------------------------------------------------------------------------

def sgm_cycles(H: int, W: int, D: int = 192, pe_count: int = 32, overhead_factor: float = 2.5) -> int:
    """Approximate SGM compute cycles.

    cost-volume + aggregation (4 directions) + LR-check + filling ≈ 2.5× base.
    Base cost-computation: H*W*D / pe_count cycles.
    """
    base = (H * W * D) / max(pe_count, 1)
    return max(1, int(math.ceil(base * overhead_factor)))


# ---------------------------------------------------------------------------
# Alignment cycle model
# ---------------------------------------------------------------------------

def alignment_cycles(H: int, W: int, pe_count: int = 32, iters: int = 8) -> int:
    """Robust Huber LS over all pixels, ~8 iterations (Levenberg-Marquardt-ish).

    Each iteration = forward residual + weighting + normal equations (2x2 for scale+shift).
    Cycles ≈ iters * H * W / pe_count + small 2x2 solve overhead.
    """
    return max(1, int(math.ceil(iters * H * W / pe_count)) + 200)


# ---------------------------------------------------------------------------
# Prefix (SGM + DMA) ops added to any pipeline
# ---------------------------------------------------------------------------

def _add_prefix_ops(dag: WorkloadDAG, img_h: int, img_w: int, sgm_disp_range: int = 192) -> tuple[int, int]:
    """Add Stage 0 (DMA) + Stage 1 (SGM) ops.

    Returns the op ids of (dma_end, sgm_end) for chaining by the caller.
    """
    dma_id = dag.add_op("stage0_dma_load", "dma",
                       input_bytes=img_h * img_w * 3 * 2,  # L+R RGB
                       metadata={"stage": 0, "note": "stereo RGB + initial conf buf"})

    sgm_cyc = sgm_cycles(img_h, img_w, sgm_disp_range)
    sgm_id = dag.add_op("stage1_sgm_compute", "systolic_array",
                       flops=img_h * img_w * sgm_disp_range * 4 * 2,  # cost + 4-dir aggregation
                       input_bytes=img_h * img_w * 3 * 2,
                       output_bytes=img_h * img_w * 4,  # disparity + conf + hole_mask packed
                       metadata={"stage": 1, "sa_op_type": "sgm_compute", "fixed_cycles": sgm_cyc})
    dag.add_edge(dma_id, sgm_id)
    return dma_id, sgm_id


def _add_alignment_op(dag: WorkloadDAG, img_h: int, img_w: int, after_op_ids: list[int]) -> int:
    """Add Stage 4 (alignment) op, chained after given predecessors."""
    align_cyc = alignment_cycles(img_h, img_w)
    align_id = dag.add_op("stage4_mono_sgm_align", "systolic_array",
                         flops=img_h * img_w * 8 * 4,  # 8 iter × 4 ops/px
                         input_bytes=img_h * img_w * 8,
                         output_bytes=img_h * img_w * 4,
                         metadata={"stage": 4, "sa_op_type": "alignment", "fixed_cycles": align_cyc})
    for pid in after_op_ids:
        dag.add_edge(pid, align_id)
    return align_id


# ---------------------------------------------------------------------------
# Public: build full end-to-end workload
# ---------------------------------------------------------------------------

def build_pipeline_workload(
    config_name: str,
    sim_config: SimConfig,
    img_h: int = 384,
    img_w: int = 768,
    sgm_disp_range: int = 192,
    variant: str = "b1",
    head_ch: int = 24,
    in_channels: int = 7,
) -> WorkloadDAG:
    """Build an end-to-end DAG.

    config_name ∈ {'baseline', 'effvit_b1_h24', 'effvit_b0_h24'}
    """
    if config_name == "baseline":
        # Build original ViT-S + custom decoder + heuristic FU workload
        dag = build_baseline_workload(sim_config, img_h=img_h, img_w=img_w)
        # Re-tag DMA + existing fusion stages for breakdown.
        for op in dag.operations.values():
            if op.engine == "dma":
                op.metadata.setdefault("stage", 0)
            elif op.name.startswith("patch_embed") or "encoder" in op.name or "attn_" in op.name or "mlp_" in op.name:
                op.metadata.setdefault("stage", 2)
            elif "decoder" in op.name or "refine" in op.name or "output" in op.name:
                op.metadata.setdefault("stage", 3)
            elif op.engine == "adcu":
                op.metadata.setdefault("stage", 4)
            elif op.engine == "fu":
                op.metadata.setdefault("stage", 5)
            else:
                op.metadata.setdefault("stage", 2)
        # Add SGM stage 1 in front by creating a new DAG? Instead insert at topological start.
        sgm_cyc = sgm_cycles(img_h, img_w, sgm_disp_range)
        sgm_id = dag.add_op("stage1_sgm_compute", "systolic_array",
                           flops=img_h * img_w * sgm_disp_range * 4 * 2,
                           input_bytes=img_h * img_w * 3 * 2,
                           output_bytes=img_h * img_w * 4,
                           metadata={"stage": 1, "sa_op_type": "sgm_compute", "fixed_cycles": sgm_cyc})
        # Make SGM a successor of the existing DMA
        for op_id, op in list(dag.operations.items()):
            if op.engine == "dma":
                dag.add_edge(op_id, sgm_id)
        return dag

    # --- EffViT path ---
    variant = "b0" if "b0" in config_name else "b1"
    dag = WorkloadDAG()
    # Stage 0 + 1
    dma_id, sgm_id = _add_prefix_ops(dag, img_h=img_h, img_w=img_w, sgm_disp_range=sgm_disp_range)
    # Mark stages (reset since we built from scratch)
    dag.operations[dma_id].metadata["stage"] = 0
    dag.operations[sgm_id].metadata["stage"] = 1

    # Stage 2: DA2 ViT-S encoder (same shape as baseline; use build_baseline to pull those ops only)
    # Simplification: reuse baseline builder and filter to encoder-only ops
    full_baseline = build_baseline_workload(sim_config, img_h=img_h, img_w=img_w)
    encoder_ops_idx_map = {}
    patch_op_id_in_new = None
    encoder_last_id_in_new = None
    for op_id, op in full_baseline.operations.items():
        # Encoder = block_* (12 ViT blocks) + patch_embed + CRM merge planning.
        # Exclude decoder_*, output_*, adcu_*, fusion_*, dma_*.
        if op.engine == "dma":
            continue
        name = op.name.lower()
        is_decoder_or_fusion = (
            name.startswith("decoder_") or name.startswith("output_")
            or name.startswith("adcu_") or name.startswith("fusion_")
            or op.engine in ("adcu", "fu", "dpc")
        )
        if is_decoder_or_fusion:
            continue
        is_encoder = (
            name.startswith("block_") or name.startswith("patch_")
            or op.engine == "crm" or op.engine == "gsu"
        )
        if not is_encoder:
            continue
        new_id = dag.add_op(op.name, op.engine,
                           flops=op.flops, weight_bytes=op.weight_bytes,
                           input_bytes=op.input_bytes, output_bytes=op.output_bytes,
                           metadata={**op.metadata, "stage": 2})
        encoder_ops_idx_map[op_id] = new_id
        if op.name.startswith("patch_"):
            patch_op_id_in_new = new_id
        encoder_last_id_in_new = new_id
    # Rebuild encoder edges
    for old_id, op in full_baseline.operations.items():
        if old_id not in encoder_ops_idx_map:
            continue
        new_id = encoder_ops_idx_map[old_id]
        for pred in op.predecessors:
            if pred in encoder_ops_idx_map:
                dag.add_edge(encoder_ops_idx_map[pred], new_id)
    # Chain SGM → patch_embed
    if patch_op_id_in_new is not None:
        dag.add_edge(sgm_id, patch_op_id_in_new)

    # Stage 3 + 5: EffViT backbone + FPN head (replaces DA2 decoder + fusion)
    #   In our real algorithm design, EffViT fusion takes (RGB + mono + sgm + conf + sgm_valid)
    #   as input, NOT the DA2 encoder feature. DA2 encoder is used for mono_disp separately.
    #   For end-to-end pipeline modeling we treat them as sequential: encoder → effvit fusion.
    effvit_dag = build_effvit_dag(variant=variant, head_ch=head_ch, in_channels=in_channels,
                                   img_h=img_h, img_w=img_w, include_prefix_ops=False)
    effvit_id_map = {}
    effvit_first_id = None
    effvit_last_id = None
    for old_id, op in effvit_dag.operations.items():
        new_id = dag.add_op(op.name, op.engine,
                           flops=op.flops, weight_bytes=op.weight_bytes,
                           input_bytes=op.input_bytes, output_bytes=op.output_bytes,
                           metadata={**op.metadata, "stage": 3 if "head" not in op.name.lower() else 5})
        # Route head_conv / residual_conv to stage 5
        if any(k in op.name for k in ("head_conv", "residual_conv", "up1_0", "up2_1", "up3_2", "up4_3")):
            dag.operations[new_id].metadata["stage"] = 5
        effvit_id_map[old_id] = new_id
        if effvit_first_id is None:
            effvit_first_id = new_id
        effvit_last_id = new_id
    # Rebuild EffViT internal edges
    for old_id, op in effvit_dag.operations.items():
        new_id = effvit_id_map[old_id]
        for pred in op.predecessors:
            if pred in effvit_id_map:
                dag.add_edge(effvit_id_map[pred], new_id)
    # Chain encoder_last → effvit_first
    if encoder_last_id_in_new is not None and effvit_first_id is not None:
        dag.add_edge(encoder_last_id_in_new, effvit_first_id)

    # Stage 4: alignment, chained after encoder + SGM (needs mono + sgm both)
    align_id = _add_alignment_op(dag, img_h, img_w,
                                 after_op_ids=[encoder_last_id_in_new] if encoder_last_id_in_new else [sgm_id])
    # Chain alignment → EffViT (so alignment result feeds fusion)
    if effvit_first_id is not None:
        dag.add_edge(align_id, effvit_first_id)

    return dag


def dag_stage_summary(dag: WorkloadDAG) -> dict:
    """Count ops/flops/weights per stage."""
    stages: dict[int, dict] = {}
    for op in dag.operations.values():
        s = op.metadata.get("stage", -1)
        d = stages.setdefault(s, {"num_ops": 0, "flops": 0, "weight_bytes": 0, "engines": {}})
        d["num_ops"] += 1
        d["flops"] += op.flops
        d["weight_bytes"] += op.weight_bytes
        d["engines"][op.engine] = d["engines"].get(op.engine, 0) + 1
    return stages


if __name__ == "__main__":
    sc = SimConfig(clock_freq_mhz=500, process_node_nm=28)
    for cfg in ["baseline", "effvit_b0_h24", "effvit_b1_h24"]:
        dag = build_pipeline_workload(cfg, sc, img_h=384, img_w=768)
        print(f"\n=== {cfg} ===")
        stages = dag_stage_summary(dag)
        for s in sorted(stages.keys()):
            d = stages[s]
            print(f"  stage {s}: {d['num_ops']} ops, {d['flops']/1e9:.2f} GFLOPs, "
                  f"{d['weight_bytes']/1e6:.2f} MB weight, engines={d['engines']}")
