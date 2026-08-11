# JAX Pallas Custom Accelerator Kernels

This directory contains high-performance custom TPU/GPU kernel implementations using **JAX Pallas** ($pl.pallas\_call$). 

These recipes demonstrate how to write custom tiled operations, manage local memory hierarchies (SRAM/VMEM vs. HBM), define indexing strategies with `BlockSpec`, and benchmark performance against JAX/XLA baselines.

## Completed Kernels

1. **[Vector Addition (`vector_add`)](file:///Users/karansbal/.gemini/jetski/scratch/gcp-ai-infra-portfolio/pallas-kernels/vector_add_README.md)**:
   * Foundational element-wise operation.
   * Teaches: 2D Grid partitioning, `BlockSpec` mapping, and CPU Interpreter validation.

2. **[Row-wise Softmax (`softmax`)](file:///Users/karansbal/.gemini/jetski/scratch/gcp-ai-infra-portfolio/pallas-kernels/softmax_README.md)**:
   * Row-wise memory-reduction kernel.
   * Teaches: Row reduction logic, local SRAM accumulation, and online Softmax calculation.

3. **[Tiled Matrix Multiplication (`matmul`)](file:///Users/karansbal/.gemini/jetski/scratch/gcp-ai-infra-portfolio/pallas-kernels/matmul_README.md)**:
   * Compute-bound tiled dense matrix multiplication.
   * Teaches: 2D grids, contract dimension indexing, accumulators, and hardware double buffering.

## Setup Instructions

Set up the isolated virtual environment:
```bash
cd pallas-kernels
python3 -m venv .venv
source .venv/bin/activate
pip install jax jaxlib --index-url https://pypi.org/simple
```
