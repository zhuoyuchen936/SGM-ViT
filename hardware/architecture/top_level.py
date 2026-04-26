"""
hardware/architecture/top_level.py
==================================
EdgeStereoDAv2 Top-Level Accelerator (SA + 5 SCUs).

Major revision: replaces separate VFE/CSFE/ADCU engine classes with
a unified systolic array and five special compute units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hardware.pe_array.unified_sa import UnifiedSystolicArray, SAConfig
from hardware.scu.crm import ConfidenceRoutingModule, CRMConfig
from hardware.scu.gsu import GatherScatterUnit, GSUConfig
from hardware.scu.dpc import DualPrecisionController, DPCConfig
from hardware.scu.adcu import AbsoluteDisparityCU, ADCUConfig
from hardware.scu.fu import FusionUnit, FUConfig
from hardware.architecture.interconnect import L2Arbiter, ArbiterConfig
from hardware.architecture.memory_hierarchy import MemoryHierarchySpec


# ---------------------------------------------------------------------------
# Top-Level Configuration
# ---------------------------------------------------------------------------

@dataclass
class AcceleratorConfig:
    """Top-level accelerator configuration."""
    # Process technology
    process_node_nm: int = 28
    clock_freq_mhz: int = 500

    # SA
    sa: SAConfig = None
    # SCUs
    crm: CRMConfig = None
    gsu: GSUConfig = None
    dpc: DPCConfig = None
    adcu: ADCUConfig = None
    fu: FUConfig = None
    # Interconnect
    arbiter: ArbiterConfig = None

    def __post_init__(self):
        if self.sa is None:
            self.sa = SAConfig()
        if self.crm is None:
            self.crm = CRMConfig()
        if self.gsu is None:
            self.gsu = GSUConfig()
        if self.dpc is None:
            self.dpc = DPCConfig()
        if self.adcu is None:
            self.adcu = ADCUConfig()
        if self.fu is None:
            self.fu = FUConfig()
        if self.arbiter is None:
            self.arbiter = ArbiterConfig()

    @property
    def peak_tops(self) -> float:
        return self.sa.num_macs * self.clock_freq_mhz * 1e6 * 2 / 1e12


# ---------------------------------------------------------------------------
# Top-Level Accelerator
# ---------------------------------------------------------------------------

class EdgeStereoDAv2Accelerator:
    """
    Top-level accelerator: Unified SA + 5 SCUs + L2 arbiter.

    Architecture:
      sa:       UnifiedSystolicArray (WS/OS configurable, 32x32 default)
      crm:      ConfidenceRoutingModule (SCU-1, merge planner)
      gsu:      GatherScatterUnit (SCU-2, data steering)
      dpc:      DualPrecisionController (SCU-3, decoder dual-path)
      adcu:     AbsoluteDisparityCU (SCU-4, depth calibration)
      fu:       FusionUnit (SCU-5, pixel ops + bilinear)
      arbiter:  L2Arbiter (32-bank crossbar)
      mem:      MemoryHierarchySpec (L0-L3 with SCU L1s)
    """

    def __init__(self, config: AcceleratorConfig | None = None):
        self.config = config or AcceleratorConfig()

        # Instantiate all modules
        self.sa = UnifiedSystolicArray(self.config.sa)
        self.crm = ConfidenceRoutingModule(self.config.crm)
        self.gsu = GatherScatterUnit(self.config.gsu)
        self.dpc = DualPrecisionController(self.config.dpc)
        self.adcu = AbsoluteDisparityCU(self.config.adcu)
        self.fu = FusionUnit(self.config.fu)
        self.arbiter = L2Arbiter(self.config.arbiter)
        self.mem = MemoryHierarchySpec(self.config.process_node_nm)

        # Module registry for iteration
        self.modules = {
            "systolic_array": self.sa,
            "crm": self.crm,
            "gsu": self.gsu,
            "dpc": self.dpc,
            "adcu": self.adcu,
            "fu": self.fu,
            "l2_arbiter": self.arbiter,
        }

    # -- Full specification ---------------------------------------------------

    def full_spec(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "accelerator": "EdgeStereoDAv2",
            "process": f"{cfg.process_node_nm}nm",
            "clock": f"{cfg.clock_freq_mhz} MHz",
            "peak_tops": f"{cfg.peak_tops:.3f} TOPS (INT8)",
            "sa": self.sa.describe(),
            "scu": {
                "crm": self.crm.describe(),
                "gsu": self.gsu.describe(),
                "dpc": self.dpc.describe(),
                "adcu": self.adcu.describe(),
                "fu": self.fu.describe(),
            },
            "interconnect": self.arbiter.describe(),
            "memory": {
                "l2_budget": self.mem.spec_c_worst_case_budget(),
                "scu_l1_total_bytes": self.mem.total_l1_scu_bytes(),
            },
        }

    # -- Area breakdown -------------------------------------------------------

    def area_breakdown(self, node_nm: int | None = None) -> dict[str, Any]:
        node = node_nm or self.config.process_node_nm
        scale = (node / 28) ** 2

        sa_area = self.sa.estimate_area_mm2(node)
        crm_area = self.crm.estimate_area_mm2(node)
        gsu_area = self.gsu.estimate_area_mm2(node)
        dpc_area = self.dpc.estimate_area_mm2(node)
        adcu_area = self.adcu.estimate_area_mm2(node)
        fu_area = self.fu.estimate_area_mm2(node)
        arbiter_area = self.arbiter.estimate_area_mm2(node)

        # SA L1 SRAM
        sa_l1 = self.mem.levels["L1_SA"].capacity_bytes / 1024 * 0.001 * scale

        # SCU L1 SRAMs
        scu_l1_total = self.mem.total_l1_scu_bytes() / 1024 * 0.001 * scale

        # Control processor
        ctrl = 0.050 * scale

        # IO pads + interconnect
        io_inter = 0.400 * scale

        components = {
            "SA (MACs + sidecars)": sa_area["total_mm2"],
            "SA L1 SRAM": sa_l1,
            "CRM": crm_area["total_mm2"],
            "GSU": gsu_area["total_mm2"],
            "DPC": dpc_area["total_mm2"],
            "ADCU": adcu_area["total_mm2"],
            "FU": fu_area["total_mm2"],
            "SCU L1 SRAMs": scu_l1_total,
            "L2 SRAM + crossbar": arbiter_area["total_mm2"],
            "Control Processor": ctrl,
            "IO + Interconnect": io_inter,
        }

        total = sum(components.values())
        components["Total"] = total
        components["node_nm"] = node

        return components

    # -- Power estimate -------------------------------------------------------

    def power_estimate(self, node_nm: int | None = None) -> dict[str, Any]:
        node = node_nm or self.config.process_node_nm
        freq = self.config.clock_freq_mhz

        sa_pwr = self.sa.estimate_power_mw(node, freq)
        crm_pwr = self.crm.estimate_power_mw(node, freq)
        gsu_pwr = self.gsu.estimate_power_mw(node, freq)
        dpc_pwr = self.dpc.estimate_power_mw(node, freq)
        adcu_pwr = self.adcu.estimate_power_mw(node, freq)
        fu_pwr = self.fu.estimate_power_mw(node, freq)
        arb_pwr = self.arbiter.estimate_power_mw(node, freq)

        io_power = 30.0  # mW, relatively constant

        dynamic = (sa_pwr["total_mw"] + crm_pwr["total_mw"] + gsu_pwr["total_mw"]
                   + dpc_pwr["total_mw"] + adcu_pwr["total_mw"] + fu_pwr["total_mw"]
                   + arb_pwr["total_mw"])
        leakage = dynamic * 0.1
        total = dynamic + leakage + io_power

        return {
            "sa_mw": sa_pwr["total_mw"],
            "crm_mw": crm_pwr["total_mw"],
            "gsu_mw": gsu_pwr["total_mw"],
            "dpc_mw": dpc_pwr["total_mw"],
            "adcu_mw": adcu_pwr["total_mw"],
            "fu_mw": fu_pwr["total_mw"],
            "arbiter_mw": arb_pwr["total_mw"],
            "io_mw": io_power,
            "dynamic_mw": dynamic,
            "leakage_mw": leakage,
            "total_mw": total,
            "node_nm": node,
            "freq_mhz": freq,
        }

    # -- Performance estimate -------------------------------------------------

    def estimate_frame_cycles(
        self,
        seq_len: int = 1370,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 12,
        keep_ratio: float = 1.0,
        stage_policy: str = "coarse_only",
        image_h: int = 518,
        image_w: int = 518,
    ) -> dict[str, Any]:
        """
        Estimate total frame cycles for the full pipeline.

        This is a quick analytical estimate (not event-driven simulation).
        """
        patch_h = image_h // 14
        patch_w = image_w // 14
        merge_seq = max(1, int(seq_len * keep_ratio)) if keep_ratio < 1.0 else seq_len

        # Patch embedding
        patch_embed = self.sa.estimate_matmul_cycles(
            seq_len - 1, 14 * 14 * 3, embed_dim,
        )

        # CRM (if merge)
        crm_est = self.crm.estimate_total_cycles(
            mode="MERGE" if keep_ratio < 1.0 else "PRUNE",
            image_h=image_h, image_w=image_w,
            grid_h=patch_h, grid_w=patch_w,
            keep_ratio=keep_ratio,
        )

        # Encoder: per-block attention + MLP
        block_attn = self.sa.estimate_attention_cycles(merge_seq, embed_dim, num_heads)
        block_mlp = self.sa.estimate_mlp_cycles(seq_len, embed_dim)

        # GSU per block (if merge)
        if keep_ratio < 1.0:
            gsu_gather = self.gsu.estimate_gather_cycles(merge_seq)
            gsu_scatter = self.gsu.estimate_scatter_merge_cycles(seq_len - 1)
            gsu_per_block = gsu_gather["total_cycles"] + gsu_scatter["total_cycles"]
        else:
            gsu_per_block = 0

        encoder_per_block = block_attn["total_cycles"] + block_mlp["total_cycles"] + gsu_per_block
        encoder_total = encoder_per_block * depth

        # Decoder: sum of all stages (simplified: proj + rn + refine + output)
        decoder_stages = [
            ("proj", patch_h, patch_w, embed_dim, 64),
            ("rn", patch_h, patch_w, 64, 64),
            ("refine", patch_h, patch_w, 64, 64),
        ]
        decoder_total = 0
        for stage_name, h, w, c_in, c_out in decoder_stages:
            for scale in [1, 2, 4, 8]:
                sh, sw = h * scale, w * scale
                est = self.sa.estimate_conv_cycles(c_in, c_out, sh, sw, kernel_size=3)
                decoder_total += est["effective_cycles"]

        # DPC overhead
        dpc_est = self.dpc.estimate_total_cycles(stage_policy, patch_h, patch_w)

        # ADCU
        adcu_est = self.adcu.estimate_total_cycles(image_h, image_w)

        # Fusion
        fusion_est = self.fu.estimate_fusion_pipeline_cycles(image_h, image_w)

        # Total (mostly sequential: encoder -> decoder -> ADCU -> fusion)
        # CRM overlaps with patch_embed
        crm_overlap = min(crm_est["total_cycles"], patch_embed["effective_cycles"])
        total = (
            patch_embed["effective_cycles"]
            + (crm_est["total_cycles"] - crm_overlap)
            + encoder_total
            + decoder_total
            + dpc_est["total_dpc_overhead_cycles"]
            + adcu_est["total_cycles"]
            + fusion_est["total_cycles"]
        )

        latency_ms = total / (self.config.clock_freq_mhz * 1e3)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        return {
            "patch_embed_cycles": patch_embed["effective_cycles"],
            "crm_cycles": crm_est["total_cycles"],
            "encoder_per_block_cycles": encoder_per_block,
            "encoder_total_cycles": encoder_total,
            "decoder_total_cycles": decoder_total,
            "dpc_overhead_cycles": dpc_est["total_dpc_overhead_cycles"],
            "adcu_cycles": adcu_est["total_cycles"],
            "fusion_cycles": fusion_est["total_cycles"],
            "total_cycles": total,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 2),
            "keep_ratio": keep_ratio,
            "stage_policy": stage_policy,
        }


# ---------------------------------------------------------------------------
# Convenience printer
# ---------------------------------------------------------------------------

def print_accelerator_summary(config: AcceleratorConfig | None = None):
    """Print a formatted accelerator summary."""
    accel = EdgeStereoDAv2Accelerator(config)

    print("=" * 60)
    print("EdgeStereoDAv2 Accelerator Summary (SA + 5 SCUs)")
    print("=" * 60)

    cfg = accel.config
    print(f"\nProcess: {cfg.process_node_nm}nm, Clock: {cfg.clock_freq_mhz} MHz")
    print(f"Peak throughput: {cfg.peak_tops:.3f} TOPS (INT8)")
    print(f"SA: {cfg.sa.rows}x{cfg.sa.cols} = {cfg.sa.num_macs} MACs")

    # Area
    print(f"\n--- Area Breakdown ({cfg.process_node_nm}nm) ---")
    area = accel.area_breakdown()
    for k, v in area.items():
        if isinstance(v, float):
            print(f"  {k:<25s}: {v:.4f} mm2")

    # Power
    print(f"\n--- Power Estimate ---")
    pwr = accel.power_estimate()
    print(f"  Dynamic: {pwr['dynamic_mw']:.1f} mW")
    print(f"  Leakage: {pwr['leakage_mw']:.1f} mW")
    print(f"  IO: {pwr['io_mw']:.1f} mW")
    print(f"  Total: {pwr['total_mw']:.1f} mW")

    # L2 Budget
    print(f"\n--- L2 Budget (Spec C worst-case) ---")
    budget = accel.mem.spec_c_worst_case_budget()
    for name, size in budget["allocations"].items():
        print(f"  {name:<30s}: {size // 1024:>4d} KB")
    print(f"  {'Total':<30s}: {budget['total_bytes'] // 1024:>4d} KB / "
          f"{budget['capacity_bytes'] // 1024} KB "
          f"({'OK' if budget['ok'] else 'EXCEEDED!'})")

    # Performance
    print(f"\n--- Frame Latency ---")
    for kr in [1.0, 0.5]:
        est = accel.estimate_frame_cycles(keep_ratio=kr)
        label = "dense" if kr == 1.0 else f"merge kr={kr}"
        print(f"  {label:<20s}: {est['total_cycles']:>12,} cycles = "
              f"{est['latency_ms']:>7.2f} ms = {est['fps']:>5.2f} FPS")

    # Memory
    print(f"\n--- Memory ---")
    print(f"  SA L1: {accel.mem.levels['L1_SA'].capacity_bytes // 1024} KB")
    print(f"  SCU L1 total: {accel.mem.total_l1_scu_bytes() // 1024} KB")
    print(f"  L2: {accel.mem.levels['L2_Global'].capacity_bytes // 1024} KB")


if __name__ == "__main__":
    print_accelerator_summary()

    # 7nm variant
    print("\n")
    cfg_7nm = AcceleratorConfig(process_node_nm=7, clock_freq_mhz=1000)
    print_accelerator_summary(cfg_7nm)
