"""
hardware/pe_array/unified_sa.py
===============================
Unified Systolic Array for the EdgeStereoDAv2 accelerator.

Replaces the separate ``mhsa_pe.SystolicPEArray`` (WS only) and
``conv_pe.ConvolutionEngine`` (OS only) with a single configurable
array that supports Weight-Stationary, Output-Stationary, and
Input-Stationary dataflows via a 2-bit mode register.

Colocated **sidecar units** (Softmax, GELU, LayerNorm) share the SA
control bus but have independent datapaths.  They are exclusive with
SA MAC compute -- the SA stalls while a sidecar runs.

Event contract (Spec D):
  Per tile:  sa:weight_load_done -> sa:tile_compute_done -> sa:tile_writeback_done
  Per op:    sa:op_complete
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from hardware.base_module import (
    Event,
    HardwareModule,
    ModuleState,
    PipelineStage,
)
from hardware.interfaces import ConfigPort, MemoryPort, StreamPort


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DataflowMode(Enum):
    WS = "weight_stationary"
    OS = "output_stationary"
    IS = "input_stationary"


@dataclass
class SAConfig:
    """Parameterized systolic array configuration."""
    rows: int = 32
    cols: int = 32
    bitwidth: int = 8          # operand precision (INT8)
    acc_width: int = 32        # accumulator precision (INT32)
    reconfigure_cycles: int = 4  # mode switch cost

    # Tiling defaults (can be overridden per-op)
    default_tile_m: int = 64
    default_tile_n: int = 128
    default_tile_k: int = 64

    # Flash Attention tiling
    flash_tile_br: int = 64
    flash_tile_bc: int = 128

    # Sidecar configurations
    softmax_segments: int = 16
    softmax_pipeline_stages: int = 4
    gelu_segments: int = 32
    gelu_pipeline_stages: int = 2
    layernorm_pipeline_stages: int = 3

    @property
    def num_macs(self) -> int:
        return self.rows * self.cols

    @property
    def pipeline_depth(self) -> int:
        """Systolic pipeline fill/drain depth."""
        return self.rows + self.cols - 1

    @property
    def mac_area_um2(self) -> float:
        """Area per MAC at 28nm reference."""
        return 400.0

    @property
    def mac_power_uw(self) -> float:
        """Power per MAC at 28nm/500MHz reference."""
        return 15.0


# ---------------------------------------------------------------------------
# Sidecar Unit Models
# ---------------------------------------------------------------------------

@dataclass
class SidecarSpec:
    """Specification for a colocated sidecar unit."""
    name: str
    pipeline_stages: int
    throughput_elements_per_cycle: int
    area_28nm_mm2: float
    power_28nm_mw: float

    def cycle_estimate(self, num_elements: int) -> int:
        """Cycles to process num_elements through this sidecar."""
        streaming = max(1, math.ceil(num_elements / self.throughput_elements_per_cycle))
        return streaming + self.pipeline_stages  # fill + streaming


class SoftmaxSidecar:
    """PWL softmax approximation sidecar (from mhsa_pe.PiecewiseLinearSoftmax)."""

    def __init__(self, num_segments: int = 16, input_range: float = 8.0):
        self.num_segments = num_segments
        self.input_range = input_range
        self.breakpoints = np.linspace(-input_range, 0, num_segments + 1)
        self.slopes = np.zeros(num_segments)
        self.intercepts = np.zeros(num_segments)
        for i in range(num_segments):
            x0, x1 = self.breakpoints[i], self.breakpoints[i + 1]
            y0, y1 = np.exp(x0), np.exp(x1)
            self.slopes[i] = (y1 - y0) / (x1 - x0)
            self.intercepts[i] = y0 - self.slopes[i] * x0

    def exp_approx(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        xc = np.clip(x, -self.input_range, 0)
        for i in range(self.num_segments):
            mask = (xc >= self.breakpoints[i]) & (xc < self.breakpoints[i + 1])
            result[mask] = self.slopes[i] * xc[mask] + self.intercepts[i]
        result[xc >= 0] = 1.0
        return result

    def forward(self, scores: np.ndarray) -> np.ndarray:
        max_val = scores.max(axis=-1, keepdims=True)
        shifted = scores - max_val
        exp_vals = self.exp_approx(shifted)
        return exp_vals / (exp_vals.sum(axis=-1, keepdims=True) + 1e-10)

    @staticmethod
    def spec() -> SidecarSpec:
        return SidecarSpec(
            name="softmax",
            pipeline_stages=4,
            throughput_elements_per_cycle=32,
            area_28nm_mm2=0.012,
            power_28nm_mw=2.0,
        )


class GELUSidecar:
    """PWL GELU approximation sidecar (from mhsa_pe.GELUApprox)."""

    def __init__(self, num_segments: int = 32, input_range: float = 4.0):
        self.num_segments = num_segments
        self.input_range = input_range
        x_pts = np.linspace(-input_range, input_range, num_segments + 1)
        self.breakpoints = x_pts
        self.slopes = np.zeros(num_segments)
        self.intercepts = np.zeros(num_segments)
        for i in range(num_segments):
            x0, x1 = x_pts[i], x_pts[i + 1]
            y0 = self._exact(x0)
            y1 = self._exact(x1)
            self.slopes[i] = (y1 - y0) / (x1 - x0)
            self.intercepts[i] = y0 - self.slopes[i] * x0

    @staticmethod
    def _exact(x: float) -> float:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        xc = np.clip(x, -self.input_range, self.input_range)
        for i in range(self.num_segments):
            mask = (xc >= self.breakpoints[i]) & (xc < self.breakpoints[i + 1])
            result[mask] = self.slopes[i] * xc[mask] + self.intercepts[i]
        result[x >= self.input_range] = x[x >= self.input_range]
        return result

    @staticmethod
    def spec() -> SidecarSpec:
        return SidecarSpec(
            name="gelu",
            pipeline_stages=2,
            throughput_elements_per_cycle=32,
            area_28nm_mm2=0.008,
            power_28nm_mw=1.2,
        )


class LayerNormSidecar:
    """Streaming LayerNorm sidecar."""

    @staticmethod
    def spec() -> SidecarSpec:
        return SidecarSpec(
            name="layernorm",
            pipeline_stages=3,
            throughput_elements_per_cycle=32,
            area_28nm_mm2=0.010,
            power_28nm_mw=1.5,
        )


# ---------------------------------------------------------------------------
# Cycle Estimation Helpers
# ---------------------------------------------------------------------------

def _matmul_tile_count(M: int, K: int, N: int,
                       tile_m: int, tile_n: int, tile_k: int) -> int:
    nm = math.ceil(M / tile_m)
    nn = math.ceil(N / tile_n)
    nk = math.ceil(K / tile_k)
    return nm * nn * nk


def _conv3x3_tile_count(H: int, W: int, C_in: int, C_out: int,
                        tile_h: int = 8, tile_w: int = 8,
                        tile_co: int = 16) -> int:
    nh = math.ceil(H / tile_h)
    nw = math.ceil(W / tile_w)
    nco = math.ceil(C_out / tile_co)
    return nh * nw * nco


# ---------------------------------------------------------------------------
# Unified Systolic Array
# ---------------------------------------------------------------------------

class UnifiedSystolicArray(HardwareModule):
    """
    Unified SA: WS for matmul, OS for conv, with sidecar units.
    """

    def __init__(self, config: SAConfig | None = None):
        self.config = config or SAConfig()
        self._current_mode: DataflowMode = DataflowMode.WS

        # Sidecar units
        self.softmax = SoftmaxSidecar(self.config.softmax_segments)
        self.gelu = GELUSidecar(self.config.gelu_segments)
        self.layernorm = LayerNormSidecar()
        self._sidecar_specs = {
            "softmax": self.softmax.spec(),
            "gelu": self.gelu.spec(),
            "layernorm": self.layernorm.spec(),
        }

        super().__init__(name="systolic_array")

    def _declare_ports(self) -> dict[str, Any]:
        return {
            "weight_load": MemoryPort(
                name="weight_load",
                data_width_bits=256,
                max_burst_len=16,
                is_read=True, is_write=False,
            ),
            "activation_in": StreamPort(
                name="activation_in",
                data_width_bits=256,
            ),
            "result_out": StreamPort(
                name="result_out",
                data_width_bits=256,
            ),
            "config": ConfigPort(name="sa_config", data_width_bits=32),
        }

    # -- Mode management -----------------------------------------------------

    def configure_mode(self, mode: DataflowMode, cycle: int) -> list[Event]:
        """Switch dataflow mode.  Returns events if reconfiguration needed."""
        if self._current_mode == mode:
            return []
        self._current_mode = mode
        self.state = ModuleState.RECONFIGURE
        done_cycle = cycle + self.config.reconfigure_cycles
        return [self._emit(done_cycle, "sa:reconfigure_done")]

    @property
    def mode(self) -> DataflowMode:
        return self._current_mode

    # -- Cycle estimation (parameterized, no hardcoded dims) -----------------

    def estimate_matmul_cycles(
        self, M: int, K: int, N: int,
        tile_m: int | None = None,
        tile_n: int | None = None,
        tile_k: int | None = None,
        utilization: float = 0.85,
    ) -> dict[str, Any]:
        """Estimate cycles for a WS matmul (M,K) x (K,N)."""
        tm = tile_m or self.config.default_tile_m
        tn = tile_n or self.config.default_tile_n
        tk = tile_k or self.config.default_tile_k

        num_tiles = _matmul_tile_count(M, K, N, tm, tn, tk)
        total_macs = M * K * N

        # Per-tile cycle breakdown
        weight_load_per_tile = max(1, math.ceil(
            (tk * tn * self.config.bitwidth // 8) /
            self.ports["weight_load"].bytes_per_beat
        ))
        skew_cycles = self.config.rows - 1
        compute_per_tile = tk  # one MAC per cycle per PE column
        de_skew_cycles = self.config.cols - 1
        writeback_per_tile = max(1, math.ceil(
            (tm * tn * self.config.acc_width // 8) /
            self.ports["result_out"].bytes_per_beat
        ))

        # First tile: full pipeline fill (serial)
        first_tile_cycles = (
            weight_load_per_tile + skew_cycles
            + compute_per_tile + de_skew_cycles
            + writeback_per_tile
        )
        # Subsequent tiles with double-buffering:
        #   While tile[i] writes back, tile[i+1] loads weights + computes.
        #   Bottleneck = max(weight_load + compute, writeback)
        steady_state_per_tile = max(
            weight_load_per_tile + compute_per_tile,
            writeback_per_tile,
        )

        if num_tiles <= 1:
            raw_cycles = first_tile_cycles
        else:
            raw_cycles = first_tile_cycles + (num_tiles - 1) * steady_state_per_tile

        effective_cycles = max(1, int(raw_cycles / utilization))

        return {
            "op_type": "matmul",
            "dims": {"M": M, "K": K, "N": N},
            "tiles": {"m": tm, "n": tn, "k": tk},
            "num_tiles": num_tiles,
            "total_macs": total_macs,
            "weight_load_per_tile": weight_load_per_tile,
            "compute_per_tile": compute_per_tile,
            "writeback_per_tile": writeback_per_tile,
            "first_tile_cycles": first_tile_cycles,
            "steady_state_per_tile": steady_state_per_tile,
            "raw_cycles": raw_cycles,
            "utilization": utilization,
            "effective_cycles": effective_cycles,
        }

    def estimate_conv_cycles(
        self, C_in: int, C_out: int, H: int, W: int,
        kernel_size: int = 3,
        tile_h: int = 8, tile_w: int = 8, tile_co: int = 16,
        utilization: float = 0.75,
    ) -> dict[str, Any]:
        """Estimate cycles for an OS conv (C_in,H,W) with (C_out,C_in,K,K) kernel."""
        if kernel_size == 1:
            # 1x1 conv = matmul: (C_out, C_in) x (C_in, H*W)
            return self.estimate_matmul_cycles(
                M=H * W, K=C_in, N=C_out, utilization=utilization,
            )

        total_macs = C_out * C_in * kernel_size * kernel_size * H * W
        num_tiles = _conv3x3_tile_count(H, W, C_in, C_out, tile_h, tile_w, tile_co)

        # Per-tile: output tile of (tile_co, tile_h, tile_w)
        # Each output pixel needs C_in * K*K MACs
        macs_per_tile = tile_co * tile_h * tile_w * C_in * kernel_size * kernel_size
        compute_per_tile = max(1, math.ceil(macs_per_tile / self.config.num_macs))

        # Weight load per tile: (tile_co, C_in, K, K) bytes
        weight_bytes = tile_co * C_in * kernel_size * kernel_size * (self.config.bitwidth // 8)
        weight_load_per_tile = max(1, math.ceil(
            weight_bytes / self.ports["weight_load"].bytes_per_beat
        ))

        # Input with halo: (C_in, tile_h+K-1, tile_w+K-1)
        input_bytes = C_in * (tile_h + kernel_size - 1) * (tile_w + kernel_size - 1) * (self.config.bitwidth // 8)
        input_load_per_tile = max(1, math.ceil(
            input_bytes / self.ports["activation_in"].bytes_per_beat
        ))

        writeback_bytes = tile_co * tile_h * tile_w * (self.config.acc_width // 8)
        writeback_per_tile = max(1, math.ceil(
            writeback_bytes / self.ports["result_out"].bytes_per_beat
        ))

        fetch_per_tile = max(weight_load_per_tile, input_load_per_tile)
        per_tile_cycles = fetch_per_tile + compute_per_tile + writeback_per_tile
        raw_cycles = num_tiles * per_tile_cycles
        effective_cycles = max(1, int(raw_cycles / utilization))

        return {
            "op_type": f"conv{kernel_size}x{kernel_size}",
            "dims": {"C_in": C_in, "C_out": C_out, "H": H, "W": W},
            "tiles": {"h": tile_h, "w": tile_w, "co": tile_co},
            "num_tiles": num_tiles,
            "total_macs": total_macs,
            "weight_load_per_tile": weight_load_per_tile,
            "compute_per_tile": compute_per_tile,
            "writeback_per_tile": writeback_per_tile,
            "per_tile_cycles": per_tile_cycles,
            "raw_cycles": raw_cycles,
            "utilization": utilization,
            "effective_cycles": effective_cycles,
        }

    def estimate_sidecar_cycles(self, sidecar_name: str, num_elements: int) -> int:
        """Estimate cycles for a sidecar operation (LN, softmax, GELU)."""
        spec = self._sidecar_specs[sidecar_name]
        return spec.cycle_estimate(num_elements)

    def estimate_attention_cycles(
        self, seq_len: int, embed_dim: int, num_heads: int,
        utilization: float = 0.85,
    ) -> dict[str, Any]:
        """
        Estimate cycles for full multi-head attention (QKV proj + attn + out proj).

        Uses flash attention tiling for the attention matrix.
        """
        head_dim = embed_dim // num_heads
        N = seq_len

        # QKV projection: (N, D) x (D, 3D)
        qkv = self.estimate_matmul_cycles(N, embed_dim, 3 * embed_dim, utilization=utilization)

        # Per-head attention with flash tiling
        br = self.config.flash_tile_br
        bc = self.config.flash_tile_bc
        n_q_tiles = math.ceil(N / br)
        n_kv_tiles = math.ceil(N / bc)
        total_attn_tiles = n_q_tiles * n_kv_tiles * num_heads

        # Q*K^T tile: (br, head_dim) x (head_dim, bc) -> (br, bc)
        qkt_per_tile = self.estimate_matmul_cycles(
            br, head_dim, bc, tile_m=br, tile_n=bc, tile_k=head_dim,
            utilization=utilization,
        )
        # softmax per tile
        softmax_per_tile = self.estimate_sidecar_cycles("softmax", br * bc)
        # attn*V tile: (br, bc) x (bc, head_dim) -> (br, head_dim)
        av_per_tile = self.estimate_matmul_cycles(
            br, bc, head_dim, tile_m=br, tile_n=head_dim, tile_k=bc,
            utilization=utilization,
        )

        attn_cycles_per_tile = (
            qkt_per_tile["effective_cycles"]
            + softmax_per_tile
            + av_per_tile["effective_cycles"]
        )
        total_attn_cycles = total_attn_tiles * attn_cycles_per_tile

        # LayerNorm after attention
        ln_cycles = self.estimate_sidecar_cycles("layernorm", N * embed_dim)

        # Output projection: (N, D) x (D, D)
        out_proj = self.estimate_matmul_cycles(N, embed_dim, embed_dim, utilization=utilization)

        total = qkv["effective_cycles"] + total_attn_cycles + ln_cycles + out_proj["effective_cycles"]

        return {
            "qkv_cycles": qkv["effective_cycles"],
            "attention_cycles": total_attn_cycles,
            "attention_tiles": total_attn_tiles,
            "layernorm_cycles": ln_cycles,
            "out_proj_cycles": out_proj["effective_cycles"],
            "total_cycles": total,
            "total_macs": (
                qkv["total_macs"]
                + total_attn_tiles * (qkt_per_tile["total_macs"] + av_per_tile["total_macs"])
                + out_proj["total_macs"]
            ),
        }

    def estimate_mlp_cycles(
        self, seq_len: int, embed_dim: int, mlp_ratio: float = 4.0,
        utilization: float = 0.90,
    ) -> dict[str, Any]:
        """Estimate cycles for MLP block: fc1 + GELU + fc2 + LayerNorm."""
        hidden = int(embed_dim * mlp_ratio)
        N = seq_len

        fc1 = self.estimate_matmul_cycles(N, embed_dim, hidden, utilization=utilization)
        gelu_cycles = self.estimate_sidecar_cycles("gelu", N * hidden)
        fc2 = self.estimate_matmul_cycles(N, hidden, embed_dim, utilization=utilization)
        ln_cycles = self.estimate_sidecar_cycles("layernorm", N * embed_dim)

        total = fc1["effective_cycles"] + gelu_cycles + fc2["effective_cycles"] + ln_cycles

        return {
            "fc1_cycles": fc1["effective_cycles"],
            "gelu_cycles": gelu_cycles,
            "fc2_cycles": fc2["effective_cycles"],
            "layernorm_cycles": ln_cycles,
            "total_cycles": total,
            "total_macs": fc1["total_macs"] + fc2["total_macs"],
        }

    # -- Event-driven interface (Spec D) ------------------------------------

    # Cap per-tile event emission to bound event queue size (C1 fix)
    MAX_TILE_EVENTS: int = 8

    def accept_op(self, op: dict, cycle: int) -> list[Event]:
        if not self.is_idle:
            return []

        self.current_op = op
        op_type = op.get("sa_op_type", "matmul")

        # Check if mode switch needed
        events: list[Event] = []
        if op_type in ("conv3x3",) and self._current_mode != DataflowMode.OS:
            events.extend(self.configure_mode(DataflowMode.OS, cycle))
            cycle += self.config.reconfigure_cycles
        elif op_type in ("matmul", "conv1x1") and self._current_mode != DataflowMode.WS:
            events.extend(self.configure_mode(DataflowMode.WS, cycle))
            cycle += self.config.reconfigure_cycles

        # Estimate cycle breakdown
        if op_type in ("matmul", "conv1x1"):
            est = self.estimate_matmul_cycles(
                M=op.get("M", 1), K=op.get("K", 1), N=op.get("N", 1),
            )
        elif op_type == "conv3x3":
            est = self.estimate_conv_cycles(
                C_in=op.get("C_in", 1), C_out=op.get("C_out", 1),
                H=op.get("H", 1), W=op.get("W", 1), kernel_size=3,
            )
        elif op_type == "sidecar":
            # Sidecar op (softmax / gelu / layernorm): use the appropriate
            # sidecar cycle estimator.  Sidecars are exclusive with MAC compute.
            sc_name = op.get("sidecar", "layernorm")
            num_el = op.get("num_elements", 1)
            sc_cycles = self.estimate_sidecar_cycles(sc_name, num_el)
            est = {"effective_cycles": sc_cycles, "num_tiles": 1,
                   "total_macs": 0,
                   "weight_load_per_tile": 0,
                   "compute_per_tile": sc_cycles,
                   "writeback_per_tile": 0}
        else:
            est = {"effective_cycles": op.get("cycles", 1), "num_tiles": 1,
                   "total_macs": op.get("flops", 0) // 2}

        # C1: per-tile events for the first MAX_TILE_EVENTS tiles, then op_complete.
        # This captures pipeline fill granularity without exploding the event queue.
        self.state = ModuleState.FETCH
        num_tiles = int(est.get("num_tiles", 1))
        per_tile_wl = int(est.get("weight_load_per_tile", 1))
        per_tile_compute = int(est.get("compute_per_tile", est.get("effective_cycles", 1)))
        per_tile_wb = int(est.get("writeback_per_tile", 1))
        total_cycles = int(est.get("effective_cycles", 1))

        op_id = op.get("id")
        total_macs = int(est.get("total_macs", 0))

        # First tile: full pipeline fill
        first_tile_cycles = per_tile_wl + per_tile_compute + per_tile_wb

        # Emit events for the first N tiles (fill phase observable)
        tiles_to_emit = min(num_tiles, self.MAX_TILE_EVENTS)
        current = cycle
        for tile_idx in range(tiles_to_emit):
            if tile_idx == 0:
                wl_done = current + per_tile_wl
                comp_done = wl_done + per_tile_compute
                wb_done = comp_done + per_tile_wb
            else:
                # Steady-state overlap: compute overlaps with prev writeback
                wl_done = current + per_tile_wl
                comp_done = wl_done + per_tile_compute
                wb_done = comp_done + per_tile_wb

            events.append(self._emit(wl_done, "sa:weight_load_done",
                                      {"op_id": op_id, "tile": tile_idx}))
            events.append(self._emit(comp_done, "sa:tile_compute_done",
                                      {"op_id": op_id, "tile": tile_idx,
                                       "macs": total_macs // max(num_tiles, 1)}))
            events.append(self._emit(wb_done, "sa:tile_writeback_done",
                                      {"op_id": op_id, "tile": tile_idx}))

            if tile_idx == 0:
                current = wb_done  # first tile serial
            else:
                # Steady state: per-tile = max(wl+compute, wb)
                current += max(per_tile_wl + per_tile_compute, per_tile_wb)

        # Always emit op_complete at total cycle budget
        op_done = cycle + total_cycles
        events.append(self._emit(op_done, "sa:op_complete",
                                  {"op_id": op_id,
                                   "total_cycles": total_cycles,
                                   "total_macs": total_macs,
                                   "num_tiles": num_tiles}))

        self.stats.total_macs += total_macs
        return events

    def handle_event(self, event: Event, cycle: int) -> list[Event]:
        action = event.action

        if action == "sa:reconfigure_done":
            self.state = ModuleState.IDLE
            return []

        if action == "sa:weight_load_done":
            self.state = ModuleState.COMPUTE
            return []

        if action == "sa:tile_compute_done":
            self.state = ModuleState.WRITEBACK
            return []

        if action == "sa:tile_writeback_done":
            # Tile done writing back; next tile can advance (steady-state overlap)
            self.state = ModuleState.COMPUTE
            return []

        if action == "sa:op_complete":
            self.state = ModuleState.IDLE
            self.stats.ops_completed += 1
            self.current_op = None
            return [self._emit(cycle, "sched:op_complete", event.data)]

        return []

    # -- Behavioral compute (for verification) --------------------------------

    def compute_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Behavioral matmul: C = A @ B (INT8 -> INT32)."""
        return A.astype(np.int32) @ B.astype(np.int32)

    # -- Spec / estimation ---------------------------------------------------

    def describe(self) -> dict:
        cfg = self.config
        return {
            "name": "UnifiedSystolicArray",
            "array_size": f"{cfg.rows}x{cfg.cols}",
            "num_macs": cfg.num_macs,
            "bitwidth": cfg.bitwidth,
            "acc_width": cfg.acc_width,
            "dataflow_modes": ["WS", "OS", "IS"],
            "reconfigure_cycles": cfg.reconfigure_cycles,
            "pipeline_depth": cfg.pipeline_depth,
            "flash_attention": {
                "tile_br": cfg.flash_tile_br,
                "tile_bc": cfg.flash_tile_bc,
            },
            "sidecars": {
                name: {
                    "pipeline_stages": s.pipeline_stages,
                    "throughput": f"{s.throughput_elements_per_cycle} elem/cycle",
                    "area_mm2": s.area_28nm_mm2,
                }
                for name, s in self._sidecar_specs.items()
            },
            "ports": {k: v.describe() for k, v in self.ports.items()},
        }

    def estimate_area_mm2(self, node_nm: int = 28) -> dict[str, float]:
        scale = (node_nm / 28) ** 2
        cfg = self.config
        mac_area = cfg.num_macs * cfg.mac_area_um2 * 1e-6 * scale
        skew_buf = 0.01 * scale  # skew/de-skew registers
        control = 0.005 * scale

        sidecar_area = sum(s.area_28nm_mm2 * scale for s in self._sidecar_specs.values())

        total = mac_area + skew_buf + control + sidecar_area
        return {
            "mac_array_mm2": mac_area,
            "skew_buffers_mm2": skew_buf,
            "control_mm2": control,
            "sidecars_mm2": sidecar_area,
            "total_mm2": total,
            "node_nm": node_nm,
        }

    def estimate_power_mw(
        self, node_nm: int = 28, freq_mhz: int = 500, utilization: float = 0.8,
    ) -> dict[str, float]:
        node_scale = (node_nm / 28) ** 1.1
        freq_scale = freq_mhz / 500
        cfg = self.config

        mac_power = cfg.num_macs * cfg.mac_power_uw * 1e-3 * node_scale * freq_scale * utilization
        sidecar_power = sum(s.power_28nm_mw * node_scale * freq_scale * utilization
                            for s in self._sidecar_specs.values())
        control_power = 2.0 * node_scale * freq_scale
        leakage = (mac_power + sidecar_power + control_power) * 0.1

        total = mac_power + sidecar_power + control_power + leakage
        return {
            "mac_array_mw": mac_power,
            "sidecars_mw": sidecar_power,
            "control_mw": control_power,
            "leakage_mw": leakage,
            "total_mw": total,
            "node_nm": node_nm,
            "freq_mhz": freq_mhz,
        }
