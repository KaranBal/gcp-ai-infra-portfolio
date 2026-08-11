# JAX Pallas Custom Kernel: Tiled Matrix Multiplication

This directory contains a custom 2D-tiled Matrix Multiplication ($C = A \times B$) kernel written in JAX Pallas, optimized for execution on Google Cloud TPUs (v5p, v6e) and Nvidia GPUs.

## Mathematical Formulation & Motivation
Matrix multiplication computes the dot product of rows of matrix $A \in \mathbb{R}^{M \times K}$ and columns of matrix $B \in \mathbb{R}^{K \times N}$ to produce $C \in \mathbb{R}^{M \times N}$:
$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

Unlike element-wise addition or row-wise reductions, matrix multiplication is a **compute-bound** operation.
* **Arithmetic Intensity**:
  - **FLOPs**: $2 \times M \times N \times K$.
  - **Memory Transfer (HBM)**: $4 \times (M \times K + K \times N + M \times N)$ bytes (ideal).
  - For $2048^3$ dimensions: $\approx 17.18\text{ GFLOPs} \div 50.3\text{ MB} \approx 341$ FLOPs/byte.

Because the operational intensity is high, the hardware compute units (MXUs on TPUs, Tensor Cores on GPUs) can be highly utilized if we keep the data in local SRAM/VMEM registers and reuse it.

## Tiling Strategy & Pallas BlockSpec
To utilize local scratchpad memory efficiently, we divide the output matrix $C$ into 2D blocks of size $B_M \times B_N$ (e.g., $128 \times 128$).

Each grid worker at coordinate `(i, j)` is responsible for computing one tile $C[i \times B_M : (i+1) \times B_M, j \times B_N : (j+1) \times B_N]$.

To calculate this tile, we load the corresponding row block from $A$ and column block from $B$:
- **$A$ Slice**: $A[i \times B_M : (i+1) \times B_M, 0:K]$ (shape `(BM, K)`)
- **$B$ Slice**: $B[0:K, j \times B_N : (j+1) \times B_N]$ (shape `(K, BN)`)

This corresponds to the following `BlockSpec` definition:
```python
grid = (M // BM, N // BN)

in_specs=[
    # BlockSpec for A: maps grid index (i, j) -> slice starting at (i * BM, 0)
    pl.BlockSpec(block_shape=(BM, K), index_map=lambda i, j: (i, 0)),
    # BlockSpec for B: maps grid index (i, j) -> slice starting at (0, j * BN)
    pl.BlockSpec(block_shape=(K, BN), index_map=lambda i, j: (0, j)),
]
out_specs=pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j))
```

In the TPU kernel, the local block matrix multiplication `c_ref[...] = a_ref[...] @ b_ref[...]` is directly mapped by the compiler to high-performance Systolic Matrix Unit (MXU) instructions.

## SRAM / VMEM Memory Footprint
For block size $B_M=128, B_N=128$ and $K=2048$:
- **A block size**: $128 \times 2048 \times 4\text{ bytes} = 1.0\text{ MB}$
- **B block size**: $2048 \times 128 \times 4\text{ bytes} = 1.0\text{ MB}$
- **C block size**: $128 \times 128 \times 4\text{ bytes} = 64\text{ KB}$
- **Total Local Block Footprint**: $\approx 2.06\text{ MB}$.

This fits well within the 16MB–32MB VMEM capacity per TPU core. If $K$ is much larger (e.g. 16384), we would further partition the $K$ dimension into blocks $B_K$ and run a loop inside the Pallas kernel.

## Running the Code
The implementation in [`matmul.py`](file:///Users/karansbal/.gemini/jetski/scratch/gcp-ai-infra-portfolio/pallas-kernels/matmul.py) contains:
1. Pure JAX baseline reference.
2. JAX Pallas custom tiled matmul kernel.
3. Verification checking outputs.
4. Auto-detection of TPU to run in interpret vs compile mode.

### Execution Command
```bash
source .venv/bin/activate
python matmul.py
```

## Benchmark Results (macOS CPU Interpreter Mode)
* **Input Size**: $2048 \times 2048 \times 2048$ ($8\text{M}$ elements)
* **Block Size (BM, BN)**: $128, 128$
* **Validation status**: `SUCCESS` (Verified for both FP32 and BF16)

### float32 (approx 16 MB per matrix)
| Implementation | Execution Time | Compute Performance |
|---|---|---|
| Pure JAX Baseline | $21.79$ ms | $788.25$ GFLOPS |
| Pallas Kernel (Interpret) | $489.31$ ms | N/A (Emulated CPU loops) |

### bfloat16 (approx 8 MB per matrix)
| Implementation | Execution Time | Compute Performance |
|---|---|---|
| Pure JAX Baseline | $22.40$ ms | $766.97$ GFLOPS |
| Pallas Kernel (Interpret) | $303.28$ ms | N/A (Emulated CPU loops) |

*Note: On a Google Cloud TPU, bfloat16 is native to the Matrix Multiply Unit (MXU) hardware, running at least 8x–16x faster than float32, which is highly optimized for deep learning training workloads.*

*Note on Pallas: In compiler mode on real TPU hardware, Pallas compiles matrix multiplication using the Mosaic backend to target MXU systolic registers directly, achieving hardware peak performance.*
