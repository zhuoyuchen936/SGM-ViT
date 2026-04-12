"""
Convolution PE Design for Cross-Scale Fusion Engine (CSFE)
Supports 1×1 and 3×3 convolutions in Output-Stationary mode
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ConvPEConfig:
    """Configuration for convolution PE array"""
    pe_rows: int = 16       # Subset of main array
    pe_cols: int = 16
    data_width: int = 8
    acc_width: int = 32
    max_kernel_size: int = 3


class ConvolutionEngine:
    """
    Reconfigurable convolution engine supporting 1×1 and 3×3 kernels.

    In Output-Stationary mode:
    - Output pixels remain in PE accumulators
    - Weight kernels are streamed through
    - Input feature map tiles are streamed through

    For 1×1 conv: equivalent to matrix multiply (same as VFE systolic array)
    For 3×3 conv: 9 MAC cycles per output pixel per input channel
    """

    def __init__(self, config: ConvPEConfig):
        self.config = config
        self.pe_rows = config.pe_rows
        self.pe_cols = config.pe_cols

    def conv2d_1x1(self, input_fm: np.ndarray, weight: np.ndarray,
                     bias: np.ndarray = None) -> np.ndarray:
        """
        1×1 convolution (pointwise).
        Equivalent to per-pixel matrix multiply.

        Args:
            input_fm: (C_in, H, W) INT8
            weight: (C_out, C_in) INT8
            bias: (C_out,) INT32 optional
        Returns:
            (C_out, H, W) INT32
        """
        C_in, H, W = input_fm.shape
        C_out = weight.shape[0]

        # Reshape to matrix multiply: (C_out, C_in) × (C_in, H*W) → (C_out, H*W)
        input_flat = input_fm.reshape(C_in, -1).astype(np.int32)
        output_flat = weight.astype(np.int32) @ input_flat

        if bias is not None:
            output_flat += bias.reshape(-1, 1)

        return output_flat.reshape(C_out, H, W)

    def conv2d_3x3(self, input_fm: np.ndarray, weight: np.ndarray,
                     bias: np.ndarray = None, padding: int = 1) -> np.ndarray:
        """
        3×3 convolution in Output-Stationary mode.

        Args:
            input_fm: (C_in, H, W) INT8
            weight: (C_out, C_in, 3, 3) INT8
            bias: (C_out,) INT32 optional
        Returns:
            (C_out, H_out, W_out) INT32
        """
        C_in, H, W = input_fm.shape
        C_out = weight.shape[0]

        if padding > 0:
            input_padded = np.pad(input_fm,
                                  ((0, 0), (padding, padding), (padding, padding)),
                                  mode='constant')
        else:
            input_padded = input_fm

        H_out = H
        W_out = W
        output = np.zeros((C_out, H_out, W_out), dtype=np.int32)

        # Output-stationary: iterate over output positions
        # Each PE holds one output pixel and accumulates over the 3×3 kernel and C_in channels
        for co in range(C_out):
            for ci in range(C_in):
                for ki in range(3):
                    for kj in range(3):
                        w = int(weight[co, ci, ki, kj])
                        output[co, :, :] += input_padded[ci, ki:ki+H_out, kj:kj+W_out].astype(np.int32) * w

        if bias is not None:
            output += bias.reshape(-1, 1, 1)

        return output

    def conv2d_3x3_tiled(self, input_fm: np.ndarray, weight: np.ndarray,
                           tile_h: int = 8, tile_w: int = 8,
                           tile_co: int = 16) -> np.ndarray:
        """
        Tiled 3×3 convolution for hardware simulation.
        Processes output tiles that fit in PE array.
        """
        C_in, H, W = input_fm.shape
        C_out = weight.shape[0]

        input_padded = np.pad(input_fm, ((0, 0), (1, 1), (1, 1)), mode='constant')
        output = np.zeros((C_out, H, W), dtype=np.int32)

        tile_count = 0
        for co_start in range(0, C_out, tile_co):
            co_end = min(co_start + tile_co, C_out)
            for h_start in range(0, H, tile_h):
                h_end = min(h_start + tile_h, H)
                for w_start in range(0, W, tile_w):
                    w_end = min(w_start + tile_w, W)
                    tile_count += 1

                    # Process this output tile
                    for co in range(co_start, co_end):
                        for ci in range(C_in):
                            for ki in range(3):
                                for kj in range(3):
                                    w = int(weight[co, ci, ki, kj])
                                    output[co, h_start:h_end, w_start:w_end] += \
                                        input_padded[ci, h_start+ki:h_end+ki, w_start+kj:w_end+kj].astype(np.int32) * w

        return output

    def cycle_estimate(self, C_in: int, C_out: int, H: int, W: int,
                        kernel_size: int = 3) -> dict:
        """Estimate cycle count for a convolution operation"""
        import math

        if kernel_size == 1:
            # 1×1 conv is just matrix multiply
            total_macs = C_out * C_in * H * W
        else:
            total_macs = C_out * C_in * kernel_size * kernel_size * H * W

        num_pes = self.pe_rows * self.pe_cols
        compute_cycles = math.ceil(total_macs / num_pes)

        # Assume ~80% utilization for conv
        effective_cycles = int(compute_cycles / 0.80)

        return {
            'total_macs': total_macs,
            'compute_cycles': compute_cycles,
            'effective_cycles': effective_cycles,
            'utilization': 0.80,
        }

    @staticmethod
    def hardware_spec() -> dict:
        return {
            'type': 'Reconfigurable Convolution Engine',
            'modes': {
                '1x1': 'Matrix multiply mode (same as VFE systolic)',
                '3x3': 'Output-stationary spatial conv mode',
            },
            'pe_array': '16×16 = 256 MACs (subset of main 32×32)',
            'reconfiguration_time': '1 cycle',
            'features': [
                'Zero-skipping for ReLU sparsity',
                'Kernel unrolling for 3×3',
                'Double-buffered input/output tiles',
            ],
        }


if __name__ == "__main__":
    print("=== Convolution PE Verification ===\n")

    config = ConvPEConfig()
    engine = ConvolutionEngine(config)

    # Test 1×1 convolution
    C_in, H, W = 64, 8, 8
    C_out = 64
    input_fm = np.random.randint(-5, 5, (C_in, H, W)).astype(np.int8)
    weight_1x1 = np.random.randint(-5, 5, (C_out, C_in)).astype(np.int8)

    output_1x1 = engine.conv2d_1x1(input_fm, weight_1x1)
    print(f"1×1 Conv: ({C_in},{H},{W}) → ({C_out},{H},{W})")
    print(f"  Output shape: {output_1x1.shape}")

    # Test 3×3 convolution
    weight_3x3 = np.random.randint(-5, 5, (C_out, C_in, 3, 3)).astype(np.int8)

    output_3x3 = engine.conv2d_3x3(input_fm, weight_3x3)
    output_tiled = engine.conv2d_3x3_tiled(input_fm, weight_3x3, tile_h=4, tile_w=4, tile_co=16)

    diff = np.abs(output_3x3 - output_tiled)
    print(f"\n3×3 Conv: ({C_in},{H},{W}) → ({C_out},{H},{W})")
    print(f"  Output shape: {output_3x3.shape}")
    print(f"  Tiled vs direct match: {(diff == 0).all()}")

    # Cycle estimates for decoder convolutions
    print(f"\n--- Decoder Convolution Cycle Estimates ---")
    for size in [37, 74, 148, 296]:
        cycles = engine.cycle_estimate(64, 64, size, size, 3)
        print(f"  {size}×{size}×64→64 (3×3): {cycles['effective_cycles']:,} cycles, "
              f"{cycles['total_macs']:,} MACs")
