"""
simulator/core/event_simulator.py
==================================
Event-driven cycle-accurate simulator for EdgeStereoDAv2.

Step 14 fixes (code review):
  C2: MLP decomposed into fc1 + gelu + fc2 + layernorm
  I5: Attention decomposed into qkv + per-flash-tile (qk + softmax + av) + out_proj
  I4: Output tag dual-path applied to output_conv1 when active
  I6: Decoder Stage 1 resize operations included
  C3: Scheduler dispatches ready ops in deterministic order (by op_id)
  S1: Stats reset uses clean assignment (in caller)

The build_workload() now produces a more accurate DAG that captures:
  * flash-attention tiling and softmax sidecar cycles (previously bypassed)
  * MLP hidden-dim matmul and GELU sidecar cycles (previously underestimated ~50%)
  * decoder Stage 1 resize ops (bilinear or conv transpose)
  * output head dual-path when 'output' tag is in active_tags
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from hardware.base_module import Event, ModuleState, ModuleStats
from hardware.pe_array.unified_sa import UnifiedSystolicArray, SAConfig
from hardware.scu.crm import ConfidenceRoutingModule, CRMConfig
from hardware.scu.gsu import GatherScatterUnit, GSUConfig
from hardware.scu.dpc import DualPrecisionController, DPCConfig, affected_tags
from hardware.scu.adcu import AbsoluteDisparityCU, ADCUConfig
from hardware.scu.fu import FusionUnit, FUConfig

from simulator.core.event_queue import EventQueue
from simulator.core.workload_dag import WorkloadDAG, Operation
from simulator.core.memory_controller import L2Controller
from simulator.core.scheduler import OperationScheduler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Simulator configuration."""
    clock_freq_mhz: int = 500
    process_node_nm: int = 28

    # SA
    pe_rows: int = 32
    pe_cols: int = 32

    # Flash attention tiling
    flash_tile_br: int = 64
    flash_tile_bc: int = 128

    # Memory
    l2_size_bytes: int = 512 * 1024
    l2_num_banks: int = 32
    l2_bandwidth_bytes_per_cycle: int = 256
    l2_read_latency: int = 3
    l2_write_latency: int = 3
    dram_bandwidth_bytes_per_cycle: int = 4
    dram_latency_cycles: int = 50

    # Sparsity
    keep_ratio: float = 1.0
    merge_layer: int = 0
    stage_policy: str = "coarse_only"

    # GSU chunking (Spec C streaming rule 1)
    gsu_chunk_size: int = 64

    @property
    def num_macs(self) -> int:
        return self.pe_rows * self.pe_cols


# ---------------------------------------------------------------------------
# Workload Builder Helpers
# ---------------------------------------------------------------------------

def _add_dual_path_stage(
    dag: WorkloadDAG,
    tag: str,
    stage_h: int, stage_w: int,
    channels: int,
    input_ids: list[int],
    active_tags: set[str],
    sa_op_meta: dict,
    sa_flops: int,
    sa_weight_bytes: int,
) -> int:
    """
    Build a dual-path decoder stage (HP conv + DPC mask + LP conv + FU blend)
    when tag is in active_tags; otherwise build a single-path stage.

    Returns the op_id of the stage's final output (blend if dual, or conv if single).
    """
    if tag in active_tags:
        # HP conv (original precision)
        hp_id = dag.add_op(
            f"decoder_{tag}_hp", "systolic_array",
            flops=sa_flops, weight_bytes=sa_weight_bytes,
            metadata=dict(sa_op_meta),
        )
        for dep in input_ids:
            dag.add_edge(dep, hp_id)

        # DPC mask generation (independent of HP conv)
        dpc_id = dag.add_op(
            f"decoder_{tag}_mask", "dpc",
            metadata={"stage_tag": tag, "stage_h": stage_h, "stage_w": stage_w},
        )
        for dep in input_ids:
            dag.add_edge(dep, dpc_id)

        # LP conv (low precision)
        lp_id = dag.add_op(
            f"decoder_{tag}_lp", "systolic_array",
            flops=sa_flops, weight_bytes=sa_weight_bytes,
            metadata=dict(sa_op_meta),
        )
        dag.add_edge(hp_id, lp_id)  # serialize on SA

        # FU blend (needs HP result, LP result, and mask)
        blend_id = dag.add_op(
            f"decoder_{tag}_blend", "fu",
            metadata={
                "fu_op_type": "alpha_blend",
                "total_pixels": stage_h * stage_w * channels,
            },
        )
        dag.add_edge(lp_id, blend_id)
        dag.add_edge(dpc_id, blend_id)
        return blend_id

    # Single-path
    single_id = dag.add_op(
        f"decoder_{tag}", "systolic_array",
        flops=sa_flops, weight_bytes=sa_weight_bytes,
        metadata=dict(sa_op_meta),
    )
    for dep in input_ids:
        dag.add_edge(dep, single_id)
    return single_id


def _add_attention_block(
    dag: WorkloadDAG,
    block_idx: int,
    input_id: int,
    M_attn: int,
    embed_dim: int,
    num_heads: int,
    flash_br: int,
    flash_bc: int,
) -> int:
    """
    Add attention block as QKV + per-flash-tile (Q*K^T + softmax + attn*V) + out_proj.

    Fixes I5: previously a single matmul op, now captures flash tiling and softmax.

    Returns the final op_id (output projection).
    """
    head_dim = embed_dim // num_heads

    # QKV projection: (M, D) x (D, 3D)
    qkv_flops = M_attn * embed_dim * 3 * embed_dim * 2
    qkv_id = dag.add_op(
        f"block_{block_idx}_qkv", "systolic_array",
        flops=qkv_flops,
        weight_bytes=embed_dim * 3 * embed_dim,
        metadata={"sa_op_type": "matmul", "M": M_attn, "K": embed_dim, "N": 3 * embed_dim},
    )
    dag.add_edge(input_id, qkv_id)

    # Flash attention tiles: iterate Q tiles, each streams all KV tiles
    n_q_tiles = max(1, math.ceil(M_attn / flash_br))
    n_kv_tiles = max(1, math.ceil(M_attn / flash_bc))

    # For simulation efficiency, aggregate all heads into one op per Q tile.
    # This preserves flash-tile granularity without exploding op count.
    prev_tile_end = qkv_id
    for q_idx in range(n_q_tiles):
        # Q*K^T for all KV tiles and all heads
        qkt_flops = num_heads * flash_br * flash_bc * head_dim * 2 * n_kv_tiles
        qkt_id = dag.add_op(
            f"block_{block_idx}_qkt_q{q_idx}", "systolic_array",
            flops=qkt_flops,
            metadata={"sa_op_type": "matmul", "M": flash_br, "K": head_dim, "N": flash_bc * n_kv_tiles},
        )
        dag.add_edge(prev_tile_end, qkt_id)

        # Softmax on the attention tile (sidecar, but happens on SA)
        softmax_id = dag.add_op(
            f"block_{block_idx}_softmax_q{q_idx}", "systolic_array",
            flops=num_heads * flash_br * flash_bc * 5,  # exp approx + normalize
            metadata={"sa_op_type": "sidecar", "sidecar": "softmax",
                      "num_elements": num_heads * flash_br * flash_bc},
        )
        dag.add_edge(qkt_id, softmax_id)

        # attn*V for all KV tiles and all heads
        av_flops = num_heads * flash_br * head_dim * 2 * flash_bc * n_kv_tiles
        av_id = dag.add_op(
            f"block_{block_idx}_av_q{q_idx}", "systolic_array",
            flops=av_flops,
            metadata={"sa_op_type": "matmul", "M": flash_br, "K": flash_bc * n_kv_tiles, "N": head_dim},
        )
        dag.add_edge(softmax_id, av_id)
        prev_tile_end = av_id

    # Output projection
    out_flops = M_attn * embed_dim * embed_dim * 2
    out_id = dag.add_op(
        f"block_{block_idx}_out_proj", "systolic_array",
        flops=out_flops,
        weight_bytes=embed_dim * embed_dim,
        metadata={"sa_op_type": "matmul", "M": M_attn, "K": embed_dim, "N": embed_dim},
    )
    dag.add_edge(prev_tile_end, out_id)
    return out_id


def _add_mlp_block(
    dag: WorkloadDAG,
    block_idx: int,
    input_id: int,
    seq_len: int,
    embed_dim: int,
    mlp_ratio: float = 4.0,
) -> int:
    """
    Add MLP block as fc1 + gelu + fc2 + layernorm.

    Fixes C2: previously a single matmul op underestimating cycles by ~50%.

    Returns the final op_id (layernorm).
    """
    mlp_hidden = int(embed_dim * mlp_ratio)

    # fc1: (seq_len, embed_dim) x (embed_dim, mlp_hidden)
    fc1_flops = seq_len * embed_dim * mlp_hidden * 2
    fc1_id = dag.add_op(
        f"block_{block_idx}_mlp_fc1", "systolic_array",
        flops=fc1_flops,
        weight_bytes=embed_dim * mlp_hidden,
        metadata={"sa_op_type": "matmul", "M": seq_len, "K": embed_dim, "N": mlp_hidden},
    )
    dag.add_edge(input_id, fc1_id)

    # GELU (sidecar on SA)
    gelu_id = dag.add_op(
        f"block_{block_idx}_mlp_gelu", "systolic_array",
        flops=seq_len * mlp_hidden * 4,  # PWL GELU ~4 ops/element
        metadata={"sa_op_type": "sidecar", "sidecar": "gelu",
                  "num_elements": seq_len * mlp_hidden},
    )
    dag.add_edge(fc1_id, gelu_id)

    # fc2: (seq_len, mlp_hidden) x (mlp_hidden, embed_dim)
    fc2_flops = seq_len * mlp_hidden * embed_dim * 2
    fc2_id = dag.add_op(
        f"block_{block_idx}_mlp_fc2", "systolic_array",
        flops=fc2_flops,
        weight_bytes=mlp_hidden * embed_dim,
        metadata={"sa_op_type": "matmul", "M": seq_len, "K": mlp_hidden, "N": embed_dim},
    )
    dag.add_edge(gelu_id, fc2_id)

    # LayerNorm (sidecar)
    ln_id = dag.add_op(
        f"block_{block_idx}_mlp_ln", "systolic_array",
        flops=seq_len * embed_dim * 3,
        metadata={"sa_op_type": "sidecar", "sidecar": "layernorm",
                  "num_elements": seq_len * embed_dim},
    )
    dag.add_edge(fc2_id, ln_id)
    return ln_id


# ---------------------------------------------------------------------------
# Workload Builder
# ---------------------------------------------------------------------------

def build_workload(
    config: SimConfig,
    img_h: int = 518,
    img_w: int = 518,
    embed_dim: int = 384,
    num_heads: int = 6,
    depth: int = 12,
    decoder_features: int = 64,
    patch_size: int = 14,
) -> WorkloadDAG:
    """
    Build the operation DAG for one frame.

    Covers Spec B full decoder taxonomy + accurate attention/MLP decomposition.
    """
    dag = WorkloadDAG()

    patch_h = img_h // patch_size
    patch_w = img_w // patch_size
    seq_len = patch_h * patch_w + 1  # +1 for CLS
    N_patches = patch_h * patch_w

    is_merge = config.keep_ratio < 1.0
    merge_seq = max(1, int(seq_len * config.keep_ratio)) if is_merge else seq_len

    active_tags = affected_tags(config.stage_policy)

    # ---- DMA load ----
    dma_id = dag.add_op("dma_load", "dma", metadata={"note": "image + conf + weights"})

    # ---- Patch embedding ----
    pe_flops = N_patches * patch_size * patch_size * 3 * embed_dim * 2
    patch_embed_id = dag.add_op(
        "patch_embed", "systolic_array",
        flops=pe_flops,
        weight_bytes=patch_size * patch_size * 3 * embed_dim,
        input_bytes=img_h * img_w * 3,
        output_bytes=seq_len * embed_dim,
        metadata={"sa_op_type": "matmul", "M": N_patches, "K": patch_size*patch_size*3, "N": embed_dim},
    )
    dag.add_edge(dma_id, patch_embed_id)

    # ---- CRM (merge planning, parallel with patch_embed) ----
    crm_id = None
    if is_merge:
        crm_id = dag.add_op(
            "crm_merge_plan", "crm",
            metadata={
                "crm_mode": "MERGE",
                "image_h": img_h, "image_w": img_w,
                "grid_h": patch_h, "grid_w": patch_w,
                "keep_ratio": config.keep_ratio,
            },
        )
        dag.add_edge(dma_id, crm_id)

    # ---- Encoder blocks ----
    prev_block_end = patch_embed_id

    for i in range(depth):
        is_sparse = is_merge and i >= config.merge_layer

        if is_sparse:
            # GSU gather (single op; internal chunking modeled by GSU's cycle estimate)
            gather_id = dag.add_op(
                f"block_{i}_gather", "gsu",
                metadata={"gsu_mode": "GATHER",
                          "num_reps": merge_seq, "num_tokens": N_patches},
            )
            dag.add_edge(prev_block_end, gather_id)
            if i == config.merge_layer and crm_id is not None:
                dag.add_edge(crm_id, gather_id)
            attn_input = gather_id
        else:
            attn_input = prev_block_end

        M_attn = merge_seq if is_sparse else seq_len

        # I5: Attention decomposed (QKV + flash tiles + out_proj)
        out_proj_id = _add_attention_block(
            dag, i, attn_input, M_attn, embed_dim, num_heads,
            config.flash_tile_br, config.flash_tile_bc,
        )

        if is_sparse:
            scatter_id = dag.add_op(
                f"block_{i}_scatter", "gsu",
                metadata={"gsu_mode": "SCATTER_MERGE",
                          "num_tokens": N_patches, "num_reps": merge_seq},
            )
            dag.add_edge(out_proj_id, scatter_id)
            mlp_input = scatter_id
        else:
            mlp_input = out_proj_id

        # C2: MLP decomposed (fc1 + gelu + fc2 + ln)
        mlp_output_id = _add_mlp_block(dag, i, mlp_input, seq_len, embed_dim)

        prev_block_end = mlp_output_id

    encoder_done = prev_block_end
    F = decoder_features

    # ---- Decoder Stage 1: Feature extraction (4 taps) ----
    # I6: include resize_layers as separate ops
    tap_scales = [(1, 1), (2, 2), (4, 4), (8, 8)]
    tap_ids = []

    for idx, (sh, sw) in enumerate(tap_scales):
        # Projection: 1x1 conv at patch_h x patch_w
        proj_flops = patch_h * patch_w * embed_dim * F * 2
        proj_weight = embed_dim * F
        proj_meta = {"sa_op_type": "conv1x1", "M": patch_h * patch_w,
                     "K": embed_dim, "N": F}

        proj_tag = f"proj_{idx + 1}"
        proj_id = _add_dual_path_stage(
            dag, proj_tag,
            stage_h=patch_h, stage_w=patch_w, channels=F,
            input_ids=[encoder_done], active_tags=active_tags,
            sa_op_meta=proj_meta,
            sa_flops=proj_flops, sa_weight_bytes=proj_weight,
        )

        # Resize layer (bilinear for scale > 1; identity for scale == 1)
        if sh > 1:
            target_h, target_w = patch_h * sh, patch_w * sw
            resize_id = dag.add_op(
                f"decoder_resize_{idx + 1}", "fu",
                metadata={"fu_op_type": "upsample_2x",
                          "H": patch_h, "W": patch_w, "target_h": target_h,
                          "channels": F},
            )
            dag.add_edge(proj_id, resize_id)
            tap_ids.append(resize_id)
        else:
            tap_ids.append(proj_id)

    # ---- Decoder Stage 2: Layer RN (4 ops) ----
    rn_ids = []
    for idx in range(4):
        sh, sw = tap_scales[idx]
        h, w = patch_h * sh, patch_w * sw
        rn_flops = h * w * F * F * 2

        rn_id = _add_dual_path_stage(
            dag, f"rn_{idx + 1}",
            stage_h=h, stage_w=w, channels=F,
            input_ids=[tap_ids[idx]], active_tags=active_tags,
            sa_op_meta={"sa_op_type": "conv1x1", "M": h * w, "K": F, "N": F},
            sa_flops=rn_flops, sa_weight_bytes=F * F,
        )
        rn_ids.append(rn_id)

    # ---- Decoder Stage 3: RefineNet cascade ----
    # path_4 uses layer_4_rn (scale 8 -> 148x148 output)
    # path_3 uses path_4 + layer_3_rn
    # path_2 uses path_3 + layer_2_rn
    # path_1 uses path_2 + layer_1_rn
    refine_configs = [
        ("path_4", (4, 4), [rn_ids[3]]),
        ("path_3", (2, 2), [rn_ids[2]]),
        ("path_2", (1, 1), [rn_ids[1]]),
        ("path_1", (1, 1), [rn_ids[0]]),
    ]

    prev_path = None
    for tag, (sh, sw), extra_deps in refine_configs:
        h, w = patch_h * sh, patch_w * sw
        # Each refinenet: 2 RCU x 2 3x3 conv
        refine_flops = 2 * 2 * h * w * F * F * 9 * 2
        refine_weight = 2 * 2 * F * F * 9

        deps = list(extra_deps)
        if prev_path is not None:
            deps.append(prev_path)

        prev_path = _add_dual_path_stage(
            dag, tag,
            stage_h=h, stage_w=w, channels=F,
            input_ids=deps, active_tags=active_tags,
            sa_op_meta={"sa_op_type": "conv3x3", "C_in": F, "C_out": F, "H": h, "W": w},
            sa_flops=refine_flops, sa_weight_bytes=refine_weight,
        )

    # ---- Decoder Stage 4: Output head ----
    # I4: apply dual-path to output_conv1 when 'output' is in active_tags
    out_h, out_w = patch_h * 8, patch_w * 8
    output_conv1_flops = out_h * out_w * F * F * 9 * 2
    output_conv1_id = _add_dual_path_stage(
        dag, "output",
        stage_h=out_h, stage_w=out_w, channels=F,
        input_ids=[prev_path], active_tags=active_tags,
        sa_op_meta={"sa_op_type": "conv3x3", "C_in": F, "C_out": F, "H": out_h, "W": out_w},
        sa_flops=output_conv1_flops, sa_weight_bytes=F * F * 9,
    )

    # Bilinear upsample to full resolution
    upsample_id = dag.add_op(
        "decoder_upsample", "fu",
        metadata={"fu_op_type": "upsample_2x", "H": out_h, "W": out_w, "channels": F},
    )
    dag.add_edge(output_conv1_id, upsample_id)

    # output_conv2 (1x1): always single-path
    output_conv2_flops = img_h * img_w * F * 1 * 2
    output_conv2_id = dag.add_op(
        "decoder_output_conv2", "systolic_array",
        flops=output_conv2_flops, weight_bytes=F * 1,
        metadata={"sa_op_type": "conv1x1", "M": img_h * img_w, "K": F, "N": 1},
    )
    dag.add_edge(upsample_id, output_conv2_id)

    decoder_done = output_conv2_id

    # ---- ADCU ----
    adcu_id = dag.add_op(
        "adcu_calibration", "adcu",
        metadata={"image_h": img_h, "image_w": img_w},
    )
    dag.add_edge(decoder_done, adcu_id)

    # ---- Fusion ----
    fusion_id = dag.add_op(
        "fusion_pipeline", "fu",
        metadata={"fu_op_type": "fusion_pipeline", "H": img_h, "W": img_w},
    )
    dag.add_edge(adcu_id, fusion_id)

    return dag


# ---------------------------------------------------------------------------
# Event-Driven Simulator
# ---------------------------------------------------------------------------

class EventDrivenSimulator:
    """
    Main event-driven simulator with event-skip architecture.
    """

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

        self.modules = {
            "systolic_array": UnifiedSystolicArray(SAConfig(
                rows=self.config.pe_rows, cols=self.config.pe_cols,
                flash_tile_br=self.config.flash_tile_br,
                flash_tile_bc=self.config.flash_tile_bc,
            )),
            "crm": ConfidenceRoutingModule(),
            "gsu": GatherScatterUnit(GSUConfig(chunk_size=self.config.gsu_chunk_size)),
            "dpc": DualPrecisionController(),
            "adcu": AbsoluteDisparityCU(),
            "fu": FusionUnit(),
        }

        self.mem_ctrl = L2Controller(
            num_banks=self.config.l2_num_banks,
            read_latency=self.config.l2_read_latency,
            write_latency=self.config.l2_write_latency,
        )

        self.event_queue = EventQueue()
        self.scheduler: Optional[OperationScheduler] = None
        self.current_cycle: int = 0

    def build_workload(self, **kwargs) -> WorkloadDAG:
        return build_workload(self.config, **kwargs)

    def simulate_frame(self, **workload_kwargs) -> dict[str, Any]:
        """Run event-driven simulation for one frame."""
        dag = self.build_workload(**workload_kwargs)
        dag.topological_order()  # validate no cycles

        # S1: clean stats reset
        for mod in self.modules.values():
            mod.state = ModuleState.IDLE
            mod.stats = ModuleStats()
            mod.current_op = None

        self.event_queue = EventQueue()
        self.scheduler = OperationScheduler(dag)
        self.current_cycle = 0

        # DMA treated as cycle-0 completion
        for op_id, op in dag.operations.items():
            if op.engine == "dma":
                self.scheduler.completed.add(op_id)
                self.scheduler.dispatched.add(op_id)

        self.scheduler.dispatch_ready(0, self.modules, self.event_queue)

        max_iterations = 1_000_000
        iteration = 0

        while self.event_queue and not self.scheduler.all_done:
            iteration += 1
            if iteration > max_iterations:
                break

            next_cycle = self.event_queue.peek_cycle()
            if next_cycle is None:
                break

            gap = next_cycle - self.current_cycle
            if gap > 0:
                for mod in self.modules.values():
                    mod.accumulate_gap(gap)
            self.current_cycle = next_cycle

            events = self.event_queue.drain_cycle(next_cycle)

            for event in events:
                if event.action == "sched:op_complete":
                    op_id = event.data.get("op_id")
                    if op_id is not None:
                        self.scheduler.mark_complete(
                            op_id, next_cycle, self.modules, self.event_queue,
                        )
                else:
                    module = self.modules.get(event.module_id)
                    if module:
                        new_events = module.handle_event(event, next_cycle)
                        self.event_queue.push_many(new_events)

        return self._collect_results(self.current_cycle, dag, iteration)

    def _collect_results(self, total_cycles: int, dag: WorkloadDAG,
                          iterations: int) -> dict[str, Any]:
        freq = self.config.clock_freq_mhz * 1e6
        latency_s = total_cycles / freq if freq > 0 else 0
        fps = 1.0 / latency_s if latency_s > 0 else 0

        engine_cycles = {}
        engine_flops = {}
        for name, mod in self.modules.items():
            engine_cycles[name] = mod.stats.busy_cycles
            engine_flops[name] = mod.stats.total_macs * 2

        total_flops = sum(engine_flops.values())

        # Legacy engine mapping for backward compat
        legacy_breakdown = {
            "VFE": {
                "cycles": (engine_cycles.get("systolic_array", 0)
                           + engine_cycles.get("crm", 0)
                           + engine_cycles.get("gsu", 0)),
                "flops": engine_flops.get("systolic_array", 0),
            },
            "CSFE": {
                "cycles": (engine_cycles.get("dpc", 0)
                           + engine_cycles.get("fu", 0)),
                "flops": engine_flops.get("fu", 0),
            },
            "ADCU": {
                "cycles": engine_cycles.get("adcu", 0),
                "flops": engine_flops.get("adcu", 0),
            },
        }
        for k in legacy_breakdown:
            c = legacy_breakdown[k]["cycles"]
            legacy_breakdown[k]["fraction"] = c / total_cycles if total_cycles > 0 else 0

        detailed = {name: mod.stats.to_dict() for name, mod in self.modules.items()}

        return {
            "total_cycles": total_cycles,
            "latency_ms": round(latency_s * 1000, 2),
            "fps": round(fps, 2),
            "total_flops": total_flops,
            "sparsity": {
                "keep_ratio": self.config.keep_ratio,
                "merge_layer": self.config.merge_layer,
                "stage_policy": self.config.stage_policy,
                "enabled": self.config.keep_ratio < 1.0,
            },
            "engine_breakdown": legacy_breakdown,
            "detailed_breakdown": detailed,
            "dag_summary": dag.summary(),
            "scheduler": self.scheduler.progress() if self.scheduler else {},
            "memory": self.mem_ctrl.stats_dict(),
            "simulation_iterations": iterations,
        }
