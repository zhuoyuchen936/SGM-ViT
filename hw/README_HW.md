# Hardware Accelerator — ICCAD Submission

This directory contains all FPGA hardware description and HLS source code
related to the **SGM-ViT** hardware accelerator design, targeting Xilinx
Zynq UltraScale+ (or similar) edge devices.

---

## Planned Structure

```
hw/
├── hls/                  # Vivado HLS / Vitis HLS C++ kernels
│   ├── sgm_core/         # Pipelined SGM disparity + confidence engine
│   ├── token_router/     # Confidence-threshold token gating logic (HLS)
│   └── attention_sparse/ # Sparse multi-head attention accelerator
├── rtl/                  # Hand-written RTL (Verilog/VHDL) if needed
├── constraints/          # Timing & pin constraint files (.xdc)
├── scripts/              # Tcl scripts for Vivado project generation
└── sim/                  # Testbenches and simulation data
```

---

## Design Goals (ICCAD)

| Metric | Target |
|---|---|
| Platform | Xilinx ZCU102 (Zynq UltraScale+) |
| SGM throughput | ≥ 30 fps @ 640×480 |
| Token pruning rate | ≥ 50 % tokens skipped in high-confidence regions |
| Attention FLOPs reduction | ≥ 40 % vs. dense DepthAnythingV2 |
| End-to-end latency | < 100 ms per frame |

---

## Notes

- All HLS code should be synthesized with **Vivado HLS 2022.2** or **Vitis HLS 2023.1**.
- Use `#pragma HLS PIPELINE II=1` for latency-critical loops.
- Interface with PS (ARM) via AXI4-Lite (control) and AXI4-Stream (data).
- Model weights are quantized to **INT8** / **FP16** for on-chip BRAM storage.
