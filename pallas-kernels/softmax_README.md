# JAX Pallas Custom Kernel: Row-wise Softmax

This directory contains a custom row-wise Softmax kernel written in JAX Pallas, optimized for execution on Google Cloud TPUs (v5p, v6e) and Nvidia GPUs.

## Mathematical Formulation & Motivation
Row-wise Softmax maps an input matrix $X \in \mathbb{R}^{M \times N}$ to an output matrix $S \in \mathbb{R}^{M \times N}$ where for each row $i$:
$$S_{i,j} = \frac{e^{X_{i,j} - \max_k(X_{i,k})}}{\sum_{k} e^{X_{i,k} - \max_k(X_{i,k})}}$$

Subtracting the maximum value along each row ($\max_k(X_{i,k})$) is mathematically equivalent to the standard Softmax but is crucial for **numerical stability** (preventing floating-point overflow when computing exponentiations).

### Memory Bandwidth Bottleneck
A standard JAX implementation of row-wise softmax compiles into multiple distinct HBM operations:
1. Find max of each row (reads $M \times N$, writes $M$).
2. Subtract max and exponentiate (reads $M \times N + M$, writes $M \times N$ to intermediate memory).
3. Compute sum of exponentials (reads $M \times N$, writes $M$).
4. Divide exponential matrix by sum (reads $M \times N + M$, writes $M \times N$).

This results in a total memory transfer of $\approx 5 \times M \times N \times 4$ bytes (for FP32).

By fusing these operations into a single Pallas kernel, we process each block completely in local VMEM/SRAM:
- **Bytes Read**: $M \times N \times 4$ bytes.
- **Bytes Written**: $M \times N \times 4$ bytes.
- **Total HBM Transfer**: $2 \times M \times N \times 4$ bytes (an ideal **2.5x reduction** in HBM traffic!).

## Tiling Strategy & Tiled Row BlockSpec
To implement row-wise reduction without inter-block synchronization, each thread block must process the entire columns of the rows assigned to it.

Therefore, we partition only along the row dimension:
* **Grid**: 1D grid of size `M // BM` where `BM` is the row block size.
* **Block Shape**: `(BM, N)` — where `N` is the full column dimension.
* **BlockSpec**:
  ```python
  grid = (M // BM,)
  pl.BlockSpec(
      block_shape=(BM, N),
      index_map=lambda i: (i, 0)
  )
  ```

In `Blocked` mode, Pallas multiplies the grid coordinate `i` by the block size `BM` to start the slice at `(i * BM, 0)`, mapping the 1D grid index to a 2D block offset.

## VMEM SRAM Limits on TPU
Since the entire column dimension `N` is loaded into SRAM/VMEM, we must ensure the block fits inside hardware limits:
* For $N = 8192$ (FP32), a single row is $32\text{ KB}$.
* With $B_M = 128$, the tile size is $128 \times 32\text{ KB} = 4\text{ MB}$.
* This comfortably fits inside the VMEM limits of Google Cloud TPU v5p/v6e (typically 16MB–32MB per core).

## Running the Code
The implementation in [`softmax.py`](file:///Users/karansbal/.gemini/jetski/scratch/gcp-ai-infra-portfolio/pallas-kernels/softmax.py) contains:
1. Pure JAX baseline reference.
2. JAX Pallas row-wise custom kernel.
3. Verification checking outputs.
4. Auto-detection of TPU to run in interpret vs compile mode.

### Execution Command
```bash
source .venv/bin/activate
python softmax.py
```

## Benchmark Results (macOS CPU Interpreter Mode)
* **Input Size**: $8192 \times 8192$ ($67.1\text{M}$ elements)
* **Block Size (BM)**: $128$
* **Validation status**: `SUCCESS` (Verified for both FP32 and BF16)

### float32 (256 MB input / 256 MB output)
| Implementation | Execution Time | Ideal Memory Bandwidth |
|---|---|---|
| Pure JAX Baseline | $37.58$ ms | $14.29$ GB/s |
| Pallas Kernel (Interpret) | $857.86$ ms | N/A (Emulated CPU loops) |

### bfloat16 (128 MB input / 128 MB output)
| Implementation | Execution Time | Ideal Memory Bandwidth |
|---|---|---|
| Pure JAX Baseline | $75.53$ ms | $3.55$ GB/s |
| Pallas Kernel (Interpret) | $589.43$ ms | N/A (Emulated CPU loops) |

*Note on CPU bfloat16 slowdown:* The Pure JAX baseline runs slower in bfloat16 on CPU compared to float32. This is a known behavior on x86 CPUs that lack native vector hardware support for bfloat16 arithmetic, requiring emulation and casting overhead. On a Google Cloud TPU, which has native hardware-level bfloat16 systolic matrix/vector units, bfloat16 runs at least 2x faster than float32.

*Note on Pallas:* Pallas CPU interpreter mode is designed for correctness verification, not performance benchmarking. When run on actual TPU hardware, Pallas compiles directly to native Mosaic assembly, bypassing JAX Python overheads.
