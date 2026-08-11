import jax
import jax.numpy as jnp
import time
from typing import Tuple
from jax.experimental import pallas as pl

def softmax_jax(x: jax.Array) -> jax.Array:
    """Pure JAX reference implementation of row-wise softmax."""
    x_max = jnp.max(x, axis=-1, keepdims=True)
    unnormalized = jnp.exp(x - x_max)
    x_sum = jnp.sum(unnormalized, axis=-1, keepdims=True)
    return unnormalized / x_sum

def softmax_kernel(x_ref, o_ref):
    """Pallas row-wise softmax kernel. Operates on local SRAM/VMEM blocks."""
    x = x_ref[...]
    row_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - row_max)
    row_sum = jnp.sum(exp_x, axis=-1, keepdims=True)
    o_ref[...] = exp_x / row_sum

def softmax_pallas(x: jax.Array, BM: int = 128, interpret: bool = True) -> jax.Array:
    """Invokes the Pallas row-wise softmax kernel.
    
    Args:
        x: Input matrix of shape (M, N).
        BM: Block size along the row dimension.
        interpret: If True, executes the kernel on CPU using the interpreter.
    """
    M, N = x.shape
    # 1D grid along the row dimension
    grid = (M // BM,)
    
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(block_shape=(BM, N), index_map=lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec(block_shape=(BM, N), index_map=lambda i: (i, 0)),
        grid=grid,
        interpret=interpret
    )(x)

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
    
    print("=== JAX Pallas Row-wise Softmax Custom Kernel ===")
    print(f"Detected Platform: {[d.platform for d in devices][0]}")
    print(f"TPU Available: {tpu_available} (Running in {'Compile' if not interpret_mode else 'Interpret'} mode)")
    
    # Matrix Size: 8192 x 8192 (67.1M elements)
    M, N = 8192, 8192
    BM = 128
    
    dtypes = [jnp.float32, jnp.bfloat16]
    
    for dtype in dtypes:
        dtype_name = "float32" if dtype == jnp.float32 else "bfloat16"
        print(f"\n--- Running Benchmark for: {dtype_name} ---")
        
        print(f"1. Generating input matrix of shape ({M}, {N})...")
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (M, N), dtype=dtype)
        
        # --- Correctness Check ---
        print("2. Verifying numerical correctness...")
        z_ref = softmax_jax(x)
        
        # Run once to compile / compile trace
        z_pal = softmax_pallas(x, BM, interpret=interpret_mode)
        
        tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
        correct = jnp.allclose(z_ref, z_pal, atol=tol, rtol=tol)
        print(f"   Validation (Pallas vs Reference JAX): {'SUCCESS' if correct else 'FAILED'}")
        if not correct:
            raise ValueError(f"Validation failed for {dtype_name}!")
            
        # --- Benchmarking ---
        print("3. Benchmarking Pure JAX vs. Pallas Kernel...")
        
        # Benchmark JAX
        jax_time = benchmark_op(softmax_jax, x, num_warmup=10, num_iters=100)
        print(f"   Pure JAX Time: {jax_time:.4f} ms")
        
        # Benchmark Pallas
        pallas_iters = 5 if interpret_mode else 100
        pallas_time = benchmark_op(
            lambda a: softmax_pallas(a, BM, interpret=interpret_mode), 
            x, num_warmup=2, num_iters=pallas_iters
        )
        print(f"   Pallas Kernel Time: {pallas_time:.4f} ms")
        
        # Performance metrics
        size_in_bytes = x.nbytes + z_ref.nbytes
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
