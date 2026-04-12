"""
Multi-Head Self-Attention Processing Element (PE) Design
Implements the datapath for attention computation in the VFE
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MHSAPEConfig:
    """Configuration for MHSA PE"""
    data_width: int = 8       # INT8 operands
    acc_width: int = 32       # INT32 accumulator
    num_heads: int = 6
    head_dim: int = 64
    softmax_segments: int = 16  # Piecewise-linear segments
    pe_rows: int = 32
    pe_cols: int = 32


class INT8MAC:
    """Single INT8 multiply-accumulate unit"""

    def __init__(self):
        self.accumulator = 0  # INT32

    def compute(self, a: int, b: int) -> int:
        """Perform a * b + accumulator"""
        product = np.int32(np.int8(a)) * np.int32(np.int8(b))
        self.accumulator += int(product)
        return self.accumulator

    def reset(self):
        self.accumulator = 0

    @staticmethod
    def hardware_spec() -> dict:
        return {
            'operation': 'INT8 × INT8 + INT32 → INT32',
            'latency': '1 cycle',
            'area_28nm_um2': 400,
            'power_28nm_uW': 15,
        }


class PiecewiseLinearSoftmax:
    """
    Hardware-friendly softmax approximation using piecewise-linear function.

    Implements the Flash Attention online softmax:
    1. Compute max(x) - running max
    2. Approximate exp(x - max) using PWL
    3. Maintain running sum for normalization
    4. Final normalization pass
    """

    def __init__(self, num_segments: int = 16, input_range: float = 8.0):
        self.num_segments = num_segments
        self.input_range = input_range

        # Build PWL approximation table for exp(x) where x ∈ [-range, 0]
        self.breakpoints = np.linspace(-input_range, 0, num_segments + 1)
        self.slopes = np.zeros(num_segments)
        self.intercepts = np.zeros(num_segments)

        for i in range(num_segments):
            x0 = self.breakpoints[i]
            x1 = self.breakpoints[i + 1]
            y0 = np.exp(x0)
            y1 = np.exp(x1)
            self.slopes[i] = (y1 - y0) / (x1 - x0)
            self.intercepts[i] = y0 - self.slopes[i] * x0

    def exp_approx(self, x: np.ndarray) -> np.ndarray:
        """Approximate exp(x) using piecewise-linear function"""
        result = np.zeros_like(x)
        x_clipped = np.clip(x, -self.input_range, 0)

        for i in range(self.num_segments):
            mask = (x_clipped >= self.breakpoints[i]) & (x_clipped < self.breakpoints[i + 1])
            result[mask] = self.slopes[i] * x_clipped[mask] + self.intercepts[i]

        # Handle x = 0 exactly
        result[x_clipped >= 0] = 1.0

        return result

    def forward(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply softmax to attention scores.

        Args:
            scores: (..., N) attention scores (last dim is softmax dim)
        Returns:
            (..., N) softmax probabilities
        """
        # Online softmax (Flash Attention style)
        max_val = scores.max(axis=-1, keepdims=True)
        shifted = scores - max_val
        exp_vals = self.exp_approx(shifted)
        sum_exp = exp_vals.sum(axis=-1, keepdims=True)
        return exp_vals / (sum_exp + 1e-10)

    def error_analysis(self, n_samples: int = 1000, dim: int = 64) -> dict:
        """Analyze approximation error vs exact softmax"""
        x = np.random.randn(n_samples, dim).astype(np.float32)

        # Exact softmax
        exact_max = x.max(axis=-1, keepdims=True)
        exact_exp = np.exp(x - exact_max)
        exact = exact_exp / exact_exp.sum(axis=-1, keepdims=True)

        # Approximate
        approx = self.forward(x)

        abs_err = np.abs(exact - approx)
        rel_err = abs_err / (exact + 1e-10)

        return {
            'max_abs_error': float(abs_err.max()),
            'mean_abs_error': float(abs_err.mean()),
            'max_rel_error': float(np.percentile(rel_err, 99)),  # 99th percentile
            'mean_rel_error': float(rel_err.mean()),
        }

    @staticmethod
    def hardware_spec() -> dict:
        return {
            'pipeline_stages': 4,
            'stage_1': 'Subtract running max (1 subtractor)',
            'stage_2': 'PWL exp lookup (comparator + multiplier + adder)',
            'stage_3': 'Accumulate sum (1 adder)',
            'stage_4': 'Normalize (1 divider or reciprocal LUT)',
            'lut_size_bytes': 16 * 4 * 2,  # 16 segments × (slope, intercept) × 4 bytes
        }


class GELUApprox:
    """Hardware-friendly GELU approximation"""

    def __init__(self, num_segments: int = 32):
        self.num_segments = num_segments
        # GELU(x) ≈ 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x³)))
        # PWL approximation over [-4, 4]
        self.range = 4.0
        x_pts = np.linspace(-self.range, self.range, num_segments + 1)

        # Build PWL table
        self.breakpoints = x_pts
        self.slopes = np.zeros(num_segments)
        self.intercepts = np.zeros(num_segments)

        for i in range(num_segments):
            x0, x1 = x_pts[i], x_pts[i+1]
            y0 = self._exact_gelu(x0)
            y1 = self._exact_gelu(x1)
            self.slopes[i] = (y1 - y0) / (x1 - x0)
            self.intercepts[i] = y0 - self.slopes[i] * x0

    def _exact_gelu(self, x: float) -> float:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        x_clipped = np.clip(x, -self.range, self.range)

        for i in range(self.num_segments):
            mask = (x_clipped >= self.breakpoints[i]) & (x_clipped < self.breakpoints[i+1])
            result[mask] = self.slopes[i] * x_clipped[mask] + self.intercepts[i]

        # Boundary: x >= range → x (GELU ≈ x for large positive)
        result[x >= self.range] = x[x >= self.range]
        # x <= -range → 0 (GELU ≈ 0 for large negative)

        return result


class SystolicPEArray:
    """
    32×32 Systolic PE Array for Weight-Stationary dataflow.

    Data flow:
    - Weights preloaded into PE registers (stationary)
    - Input activations flow left-to-right
    - Partial sums flow top-to-bottom
    """

    def __init__(self, config: MHSAPEConfig):
        self.config = config
        self.rows = config.pe_rows
        self.cols = config.pe_cols

        # PE state
        self.weights = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.accumulators = np.zeros((self.rows, self.cols), dtype=np.int32)

    def load_weights(self, weight_tile: np.ndarray):
        """Load weight tile into PE array registers"""
        r = min(weight_tile.shape[0], self.rows)
        c = min(weight_tile.shape[1], self.cols)
        self.weights[:r, :c] = weight_tile[:r, :c]

    def compute_tile(self, input_tile: np.ndarray) -> np.ndarray:
        """
        Compute output tile using systolic dataflow.

        Args:
            input_tile: (M, K) INT8 input activation tile
        Returns:
            (M, N) INT32 output tile (partial sums)
        """
        M, K = input_tile.shape
        N = self.cols

        self.accumulators.fill(0)
        output = np.zeros((M, N), dtype=np.int32)

        # Simulate systolic execution
        for m in range(M):
            for k in range(K):
                # Each input element flows across a row of PEs
                for n in range(N):
                    # MAC operation
                    product = np.int32(input_tile[m, k]) * np.int32(self.weights[k % self.rows, n])
                    output[m, n] += product

        return output

    def compute_matmul_tiled(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Full tiled matrix multiplication: C = A × B

        Args:
            A: (M, K) INT8
            B: (K, N) INT8
        Returns:
            C: (M, N) INT32
        """
        M, K = A.shape
        _, N = B.shape

        C = np.zeros((M, N), dtype=np.int32)

        # Tile dimensions
        tm = self.rows
        tn = self.cols
        tk = self.rows  # Reduction tile

        for n_start in range(0, N, tn):
            n_end = min(n_start + tn, N)
            for k_start in range(0, K, tk):
                k_end = min(k_start + tk, K)

                # Load weight tile
                w_tile = B[k_start:k_end, n_start:n_end]
                padded_w = np.zeros((self.rows, self.cols), dtype=np.int8)
                padded_w[:w_tile.shape[0], :w_tile.shape[1]] = w_tile
                self.load_weights(padded_w)

                for m_start in range(0, M, tm):
                    m_end = min(m_start + tm, M)

                    # Input tile
                    a_tile = A[m_start:m_end, k_start:k_end]

                    # Compute
                    result = self.compute_tile(a_tile)
                    C[m_start:m_end, n_start:n_end] += result[:m_end-m_start, :n_end-n_start]

        return C

    @staticmethod
    def hardware_spec(pe_rows: int = 32, pe_cols: int = 32) -> dict:
        num_pes = pe_rows * pe_cols
        return {
            'array_size': f'{pe_rows}×{pe_cols} = {num_pes} PEs',
            'dataflow': 'Weight-Stationary (configurable to OS)',
            'per_pe': {
                'mac_unit': 'INT8×INT8 + INT32 → INT32',
                'weight_reg': '8 bits',
                'accumulator': '32 bits',
                'input_reg': '8 bits',
            },
            'array_features': {
                'systolic_connections': 'Left-to-right (input), Top-to-bottom (psum)',
                'pipeline_depth': pe_rows + pe_cols - 1,
                'sustained_throughput': f'{num_pes} MACs/cycle',
            },
            'area_28nm_mm2': num_pes * 400e-6 + 0.01,  # MACs + overhead
            'power_28nm_mW': num_pes * 0.015 * 500,  # at 500MHz
        }


if __name__ == "__main__":
    print("=== MHSA PE Design Verification ===\n")

    config = MHSAPEConfig()

    # Test softmax approximation
    print("--- Softmax Approximation ---")
    softmax = PiecewiseLinearSoftmax(num_segments=16)
    err = softmax.error_analysis(n_samples=1000, dim=64)
    print(f"  Mean abs error: {err['mean_abs_error']:.6f}")
    print(f"  Max abs error: {err['max_abs_error']:.6f}")
    print(f"  Mean rel error: {err['mean_rel_error']:.6f}")

    # Test GELU approximation
    print("\n--- GELU Approximation ---")
    gelu = GELUApprox(num_segments=32)
    x_test = np.linspace(-3, 3, 1000)
    exact_gelu = 0.5 * x_test * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x_test + 0.044715 * x_test**3)))
    approx_gelu = gelu.forward(x_test)
    gelu_err = np.abs(exact_gelu - approx_gelu)
    print(f"  Mean abs error: {gelu_err.mean():.6f}")
    print(f"  Max abs error: {gelu_err.max():.6f}")

    # Test systolic array
    print("\n--- Systolic PE Array ---")
    pe_array = SystolicPEArray(config)

    M, K, N = 64, 64, 64
    A = np.random.randint(-10, 10, (M, K)).astype(np.int8)
    B = np.random.randint(-10, 10, (K, N)).astype(np.int8)

    # Reference
    C_ref = A.astype(np.int32) @ B.astype(np.int32)

    # Systolic
    C_hw = pe_array.compute_matmul_tiled(A, B)

    diff = np.abs(C_ref - C_hw)
    print(f"  Test size: ({M}, {K}) × ({K}, {N})")
    print(f"  Max error: {diff.max()}")
    print(f"  Correct: {(diff == 0).all()}")

    # Hardware spec
    spec = SystolicPEArray.hardware_spec()
    print(f"\n--- Hardware Specification ---")
    for k, v in spec.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
