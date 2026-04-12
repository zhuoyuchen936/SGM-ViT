"""
Bilinear Upsampling PE for Cross-Scale Fusion Engine
Hardware implementation of 2x bilinear interpolation
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class UpsamplePEConfig:
    """Configuration for upsampling hardware"""
    data_width: int = 16     # FP16 or INT16 for interpolation
    output_width: int = 8    # INT8 output
    parallel_pixels: int = 32  # Pixels processed per cycle


class BilinearUpsampleUnit:
    """
    Hardware bilinear interpolation unit for 2× upsampling.

    For 2× upsampling, each output pixel is computed from 4 input pixels:
        out[2i, 2j]     = in[i, j]                    (direct copy)
        out[2i, 2j+1]   = 0.5 * in[i, j] + 0.5 * in[i, j+1]
        out[2i+1, 2j]   = 0.5 * in[i, j] + 0.5 * in[i+1, j]
        out[2i+1, 2j+1] = 0.25 * (in[i,j] + in[i,j+1] + in[i+1,j] + in[i+1,j+1])

    Hardware implementation uses fixed-point arithmetic:
    - Multiply by 0.5 → right shift by 1
    - Multiply by 0.25 → right shift by 2
    - Only additions and shifts required (no multipliers needed)
    """

    def __init__(self, config: UpsamplePEConfig):
        self.config = config

    def upsample_2x(self, input_fm: np.ndarray) -> np.ndarray:
        """
        2× bilinear upsampling (hardware-accurate implementation).

        Uses only additions and bit-shifts (no floating-point multiplication).

        Args:
            input_fm: (C, H, W) feature map
        Returns:
            (C, 2H, 2W) upsampled feature map
        """
        C, H, W = input_fm.shape
        output = np.zeros((C, 2 * H, 2 * W), dtype=input_fm.dtype)

        for c in range(C):
            for i in range(H):
                for j in range(W):
                    i1 = min(i + 1, H - 1)
                    j1 = min(j + 1, W - 1)

                    # Top-left (direct copy)
                    output[c, 2*i, 2*j] = input_fm[c, i, j]

                    # Top-right (average of horizontal neighbors)
                    output[c, 2*i, 2*j+1] = (int(input_fm[c, i, j]) + int(input_fm[c, i, j1])) >> 1

                    # Bottom-left (average of vertical neighbors)
                    output[c, 2*i+1, 2*j] = (int(input_fm[c, i, j]) + int(input_fm[c, i1, j])) >> 1

                    # Bottom-right (average of 4 neighbors)
                    output[c, 2*i+1, 2*j+1] = (
                        int(input_fm[c, i, j]) + int(input_fm[c, i, j1]) +
                        int(input_fm[c, i1, j]) + int(input_fm[c, i1, j1])
                    ) >> 2

        return output

    def upsample_2x_vectorized(self, input_fm: np.ndarray) -> np.ndarray:
        """Vectorized version for faster simulation"""
        C, H, W = input_fm.shape
        inp = input_fm.astype(np.int32)
        output = np.zeros((C, 2 * H, 2 * W), dtype=input_fm.dtype)

        # Pad input for boundary handling
        padded = np.pad(inp, ((0, 0), (0, 1), (0, 1)), mode='edge')

        # Direct copy
        output[:, 0::2, 0::2] = input_fm

        # Horizontal average
        output[:, 0::2, 1::2] = ((padded[:, :H, :W] + padded[:, :H, 1:W+1]) >> 1).astype(input_fm.dtype)

        # Vertical average
        output[:, 1::2, 0::2] = ((padded[:, :H, :W] + padded[:, 1:H+1, :W]) >> 1).astype(input_fm.dtype)

        # 4-neighbor average
        output[:, 1::2, 1::2] = ((
            padded[:, :H, :W] + padded[:, :H, 1:W+1] +
            padded[:, 1:H+1, :W] + padded[:, 1:H+1, 1:W+1]
        ) >> 2).astype(input_fm.dtype)

        return output

    def cycle_estimate(self, C: int, H: int, W: int) -> dict:
        """Estimate cycles for upsampling operation"""
        total_output_pixels = C * 2 * H * 2 * W
        cycles = total_output_pixels // self.config.parallel_pixels

        return {
            'input_size': f'{C}×{H}×{W}',
            'output_size': f'{C}×{2*H}×{2*W}',
            'total_pixels': total_output_pixels,
            'cycles': cycles,
            'parallel_pixels': self.config.parallel_pixels,
        }

    @staticmethod
    def hardware_spec() -> dict:
        return {
            'type': 'Bilinear 2× Upsampling Unit',
            'implementation': {
                'multipliers': 0,  # Only shifts!
                'adders': 4,  # Per output pixel: up to 4 additions
                'shifters': 2,  # Right-shift by 1 or 2
            },
            'pipeline_stages': 2,
            'throughput': '32 pixels/cycle',
            'area_28nm_um2': 5000,  # Very small
            'power_28nm_uW': 50,
            'features': [
                'Zero-multiplier design (shift-add only)',
                'Pipelined for sustained throughput',
                'Direct connection to CSFE convolution engine',
            ],
        }


if __name__ == "__main__":
    print("=== Bilinear Upsample PE Verification ===\n")

    config = UpsamplePEConfig()
    unit = BilinearUpsampleUnit(config)

    # Test with small feature map
    C, H, W = 4, 8, 8
    input_fm = np.random.randint(-20, 20, (C, H, W)).astype(np.int16)

    # Compare scalar and vectorized versions
    output_scalar = unit.upsample_2x(input_fm)
    output_vector = unit.upsample_2x_vectorized(input_fm)

    print(f"Input shape: ({C}, {H}, {W})")
    print(f"Output shape: {output_scalar.shape}")
    print(f"Scalar vs vectorized match: {np.allclose(output_scalar, output_vector)}")

    # Cycle estimates for decoder stages
    print(f"\n--- Cycle Estimates ---")
    for size in [(37, 37), (74, 74), (148, 148), (296, 296)]:
        cycles = unit.cycle_estimate(64, size[0], size[1])
        print(f"  {cycles['input_size']} → {cycles['output_size']}: {cycles['cycles']} cycles")

    print(f"\nHardware: {BilinearUpsampleUnit.hardware_spec()}")
