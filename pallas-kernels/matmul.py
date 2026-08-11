import jax
import jax.numpy as jnp
import time
from typing import Tuple
from jax.experimental import pallas as pl

def matmul_jax(a: jax.Array, b: jax.Array) -> jax.Array:
    """Pure JAX reference implementation of matrix multiplication."""
    return a @ b

def matmul_kernel(a_ref, b_ref, c_ref):
    """Pallas tiled matmul kernel. Performs a local matrix multiplication blockwise."""
    c_ref[...] = a_ref[...] @ b_ref[...]

def matmul_pallas(a: jax.Array, b: jax.Array, BM: int = 128, BN: int = 128, interpret: bool = True) -> jax.Array:
    """Invokes the Pallas tiled matmul kernel.
    
    Args:
        a: Input matrix A of shape (M, K).
        b: Input matrix B of shape (K, N).
        BM: Block size along the row dimension of A/C.
        BN: Block size along the column dimension of B/C.
        interpret: If True, executes the kernel on CPU using the interpreter.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: K dimensions must match ({K} != {K2})"
    
    # 2D Grid partitioning the output C (M x N)
    grid = (M // BM, N // BN)
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        in_specs=[
            # BlockSpec for A: maps grid index (i, j) -> slice A[i*BM : (i+1)*BM, 0:K]
            pl.BlockSpec(block_shape=(BM, K), index_map=lambda i, j: (i, 0)),
            # BlockSpec for B: maps grid index (i, j) -> slice B[0:K, j*BN : (j+1)*BN]
            pl.BlockSpec(block_shape=(K, BN), index_map=lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j)),
        grid=grid,
        interpret=interpret
    )(a, b)

def benchmark_op(op_func, *args, num_warmup: int = 10, num_iters: int = 50) -> float:
    """Benchmarks a function and returns average execution time in milliseconds."""
    # Warmup
    for _ in range(num_warmup):
        res = op_func(*args)
        res.block_until_ready()
    
    # Timing loop
    start_time = time.perf_counter()
    for _ in range(num_iters):
        res = op_func(*args)
        res.block_until_ready()
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / num_iters) * 1000.0

def main():
    # Detect hardware backend
    devices = jax.devices()
    tpu_available = any(d.platform == 'tpu' for d in devices)
    interpret_mode = not tpu_available
    
    print("=== JAX Pallas Tiled Matmul Custom Kernel ===")
    print(f"Detected Platform: {[d.platform for d in devices][0]}")
    print(f"TPU Available: {tpu_available} (Running in {'Compile' if not interpret_mode else 'Interpret'} mode)")
    
    # Matrix Size: 2048 x 2048 (8M elements)
    M, K, N = 2048, 2048, 2048
    BM, BN = 128, 128
    
    dtypes = [jnp.float32, jnp.bfloat16]
    
    for dtype in dtypes:
        dtype_name = "float32" if dtype == jnp.float32 else "bfloat16"
        print(f"\n--- Running Benchmark for: {dtype_name} ---")
        
        print(f"1. Generating input matrices: A ({M}x{K}) and B ({K}x{N})...")
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (M, K), dtype=dtype)
        b = jax.random.normal(k2, (K, N), dtype=dtype)
        
        # --- Correctness Check ---
        print("2. Verifying numerical correctness...")
        c_ref = matmul_jax(a, b)
        
        # Run once to compile / compile trace
        c_pal = matmul_pallas(a, b, BM, BN, interpret=interpret_mode)
        
        tol = 1e-2 if dtype == jnp.bfloat16 else 1e-3
        correct = jnp.allclose(c_ref, c_pal, atol=tol, rtol=tol)
        print(f"   Validation (Pallas vs Reference JAX): {'SUCCESS' if correct else 'FAILED'}")
        if not correct:
            # Let's print maximum difference
            max_diff = jnp.max(jnp.abs(c_ref - c_pal))
            print(f"   Max Diff: {max_diff}")
            raise ValueError(f"Validation failed for {dtype_name}!")
            
        # --- Benchmarking ---
        print("3. Benchmarking Pure JAX vs. Pallas Kernel...")
        
        # Benchmark JAX
        jax_time = benchmark_op(matmul_jax, a, b, num_warmup=10, num_iters=50)
        print(f"   Pure JAX Time: {jax_time:.4f} ms")
        
        # Benchmark Pallas
        pallas_iters = 3 if interpret_mode else 50
        pallas_time = benchmark_op(
            lambda x, y: matmul_pallas(x, y, BM, BN, interpret=interpret_mode), 
            a, b, num_warmup=1, num_iters=pallas_iters
        )
        print(f"   Pallas Kernel Time: {pallas_time:.4f} ms")
        
        # Performance metrics
        flops = 2 * M * N * K
        gflops = flops / 1e9
        
        jax_tflops = (gflops / 1000.0) / (jax_time / 1000.0)
        print(f"   Pure JAX Performance: {jax_tflops * 1000:.2f} GFLOPS (TFLOPS: {jax_tflops:.4f})")
        if not interpret_mode:
            pal_tflops = (gflops / 1000.0) / (pallas_time / 1000.0)
            print(f"   Pallas Performance  : {pal_tflops * 1000:.2f} GFLOPS (TFLOPS: {pal_tflops:.4f})")
        else:
            print("   Pallas Performance omitted (interpret mode is not representative of hardware speed).")

if __name__ == "__main__":
    main()
