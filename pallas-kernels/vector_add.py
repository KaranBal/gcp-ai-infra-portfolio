import jax
import jax.numpy as jnp
import time
from typing import Tuple
from jax.experimental import pallas as pl

def vector_add_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pure JAX reference implementation of element-wise vector addition."""
    return x + y

def vector_add_kernel(x_ref, y_ref, o_ref):
    """Pallas kernel function. Operates on local memory references (SRAM/VMEM)."""
    o_ref[...] = x_ref[...] + y_ref[...]

def vector_add_pallas(x: jax.Array, y: jax.Array, BM: int = 1024, BN: int = 1024, interpret: bool = True) -> jax.Array:
    """Invokes the Pallas vector add kernel using pl.pallas_call.
    
    Args:
        x: Input matrix of shape (M, N).
        y: Input matrix of shape (M, N).
        BM: Block size along the row dimension.
        BN: Block size along the column dimension.
        interpret: If True, executes the kernel on CPU using the interpreter.
    """
    M, N = x.shape
    # Grid represents the 2D block partitioning of the iteration space
    grid = (M // BM, N // BN)
    
    return pl.pallas_call(
        vector_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j)),
            pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(BM, BN), index_map=lambda i, j: (i, j)),
        grid=grid,
        interpret=interpret
    )(x, y)

def benchmark_op(op_func, *args, num_warmup: int = 10, num_iters: int = 100) -> float:
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
    
    print("=== JAX Pallas Vector Addition Custom Kernel ===")
    print(f"Detected Platform: {[d.platform for d in devices][0]}")
    print(f"TPU Available: {tpu_available} (Running in {'Compile' if not interpret_mode else 'Interpret'} mode)")
    
    # Matrix Size: 8192 x 8192 (67.1M elements)
    M, N = 8192, 8192
    BM, BN = 1024, 1024
    
    dtypes = [jnp.float32, jnp.bfloat16]
    
    for dtype in dtypes:
        dtype_name = "float32" if dtype == jnp.float32 else "bfloat16"
        print(f"\n--- Running Benchmark for: {dtype_name} ---")
        
        print(f"1. Generating input matrices of shape ({M}, {N})...")
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (M, N), dtype=dtype)
        y = jax.random.normal(k2, (M, N), dtype=dtype)
        
        # --- Correctness Check ---
        print("2. Verifying numerical correctness...")
        z_ref = vector_add_jax(x, y)
        
        # Run once to compile / compile trace
        z_pal = vector_add_pallas(x, y, BM, BN, interpret=interpret_mode)
        
        # Adjust tolerance based on dtype precision
        tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
        correct = jnp.allclose(z_ref, z_pal, atol=tol, rtol=tol)
        print(f"   Validation (Pallas vs Reference JAX): {'SUCCESS' if correct else 'FAILED'}")
        if not correct:
            raise ValueError(f"Validation failed for {dtype_name}!")
            
        # --- Benchmarking ---
        print("3. Benchmarking Pure JAX vs. Pallas Kernel...")
        
        # Benchmark JAX
        jax_time = benchmark_op(vector_add_jax, x, y, num_warmup=10, num_iters=100)
        print(f"   Pure JAX Time: {jax_time:.4f} ms")
        
        # Benchmark Pallas
        pallas_iters = 10 if interpret_mode else 100
        pallas_time = benchmark_op(
            lambda a, b: vector_add_pallas(a, b, BM, BN, interpret=interpret_mode), 
            x, y, num_warmup=2, num_iters=pallas_iters
        )
        print(f"   Pallas Kernel Time: {pallas_time:.4f} ms")
        
        # Performance metrics
        size_in_bytes = x.nbytes + y.nbytes + z_ref.nbytes
        gbytes = size_in_bytes / 1e9
        
        jax_throughput = gbytes / (jax_time / 1000.0)
        print(f"   Pure JAX Memory Throughput: {jax_throughput:.2f} GB/s")
        if not interpret_mode:
            pal_throughput = gbytes / (pallas_time / 1000.0)
            print(f"   Pallas Memory Throughput  : {pal_throughput:.2f} GB/s")
        else:
            print("   Pallas Throughput omitted (interpret mode is not representative of hardware speed).")

if __name__ == "__main__":
    main()
