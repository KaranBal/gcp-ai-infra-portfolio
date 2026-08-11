import time
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx
import optax
import numpy as np

# ==============================================================================
# JAX TPU MESH DEFINITION & SHARDING CONFIGURATION
# ==============================================================================
# Google Cloud TPUs are arranged in a multi-dimensional torus. For large-scale
# training, we partition parameters and activations across a physical Mesh.
# We define a 2D mesh layout:
#   - 'fsdp' (Fully Sharded Data Parallel): sharded across data/batch dimension.
#   - 'tp' (Tensor Parallel): sharded across model/hidden dimensions.
# ==============================================================================

def setup_tpu_mesh():
    # Detect available devices (TPU chips/cores)
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Detected Platform: {jax.config.read('jax_platform_name')}")
    print(f"Total Available Devices: {num_devices} ({[d.device_kind for d in devices[:1]]})")
    
    # Define a 2D mesh topology. If running on single-host CPU/GPU, we shape the
    # mesh based on available virtual/physical devices.
    if num_devices >= 8:
        # e.g., for 8 TPU cores (v5litepod-8 or v6e-8)
        mesh_shape = (4, num_devices // 4)
    elif num_devices >= 2:
        mesh_shape = (2, num_devices // 2)
    else:
        mesh_shape = (1, 1)
        
    mesh_devices = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(mesh_devices, axis_names=('fsdp', 'tp'))
    print(f"Initialized 2D TPU Mesh with shape: {mesh_shape} (fsdp, tp)")
    return mesh

# ==============================================================================
# SHARDED LORA DENSE LAYER (FLAX NNX)
# ==============================================================================
# Low-Rank Adaptation (LoRA) freezes pre-trained weights and adds low-rank 
# update matrices to reduce HBM traffic and memory footprint.
# We use bfloat16 for all calculations to leverage TPU matrix multiplication units.
# ==============================================================================

class LoRADense(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, *, rngs: nnx.Rngs):
        # Frozen base weight (simulating pre-trained weights)
        self.base_kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features), dtype=jnp.bfloat16)
        )
        # Trainable LoRA adapter weights
        self.lora_a = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, rank), dtype=jnp.bfloat16) * (1.0 / rank)
        )
        self.lora_b = nnx.Param(
            jnp.zeros((rank, out_features), dtype=jnp.bfloat16)
        )
        self.scaling = alpha / rank

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Cast input to bfloat16 for TPU Tensor Core acceleration
        x = x.astype(jnp.bfloat16)
        
        # Stop gradients on base weights to freeze them during backward pass
        base_w = jax.lax.stop_gradient(self.base_kernel.value)
        base_out = jnp.matmul(x, base_w)
        
        # Low-rank path: x -> lora_a -> lora_b
        lora_out = jnp.matmul(jnp.matmul(x, self.lora_a.value), self.lora_b.value) * self.scaling
        return base_out + lora_out

# ==============================================================================
# LORA FINE-TUNING MLP MODEL
# ==============================================================================

class LoRAMLP(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, rank: int = 8, *, rngs: nnx.Rngs):
        self.layer1 = LoRADense(in_features, hidden_features, rank=rank, rngs=rngs)
        self.layer2 = LoRADense(hidden_features, out_features, rank=rank, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layer1(x)
        x = jax.nn.relu(x)
        x = self.layer2(x)
        return x

# ==============================================================================
# SHARDED LOSS & TRAINING STEP
# ==============================================================================

def compute_loss(model: LoRAMLP, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = model(x)
    # Mean Squared Error Loss
    return jnp.mean(jnp.square(preds - y))

@nnx.jit
def train_step(model: LoRAMLP, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
    # Compute gradients only for parameters that do not have stop_gradient applied.
    # In NNX, self.base_kernel is wrapped with stop_gradient in forward pass,
    # so its calculated gradient will be 0.
    loss, grads = nnx.value_and_grad(compute_loss)(model, x, y)
    
    # Clip gradients to ensure stable training on TPUs
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # Update optimizer state and model parameters
    optimizer.update(grads)
    return loss

# ==============================================================================
# MAIN EXECUTION & BENCHMARKING
# ==============================================================================

def main():
    # Setup the TPU virtual 2D mesh
    mesh = setup_tpu_mesh()
    
    # Model dimensions
    batch_size = 1024
    in_features = 4096
    hidden_features = 8192
    out_features = 4096
    lora_rank = 16
    
    # Initialize RNGs and Model
    rngs = nnx.Rngs(params=0)
    model = LoRAMLP(in_features, hidden_features, out_features, rank=lora_rank, rngs=rngs)
    
    # Initialize Optax Optimizer
    # We only update trainable parameters (LoRA adapters)
    tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
    optimizer = nnx.Optimizer(model, tx)
    
    # Setup Named Sharding for inputs
    # Inputs are partitioned across 'fsdp' (data parallelism) dimension, 
    # while model features remain unsharded (None) or partitioned via Tensor Parallelism.
    input_sharding = NamedSharding(mesh, P('fsdp', None))
    output_sharding = NamedSharding(mesh, P('fsdp', None))
    
    # Generate dummy input and target data
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    raw_x = jax.random.normal(k1, (batch_size, in_features), dtype=jnp.bfloat16)
    raw_y = jax.random.normal(k2, (batch_size, out_features), dtype=jnp.bfloat16)
    
    # Shard inputs onto the device mesh
    x = jax.device_put(raw_x, input_sharding)
    y = jax.device_put(raw_y, output_sharding)
    
    print("\nStarting Training Step Profiling...")
    # Warmup step to trigger XLA compilation
    start_compile = time.perf_counter()
    loss = train_step(model, optimizer, x, y)
    loss.block_until_ready()
    compile_time = time.perf_counter() - start_compile
    print(f"XLA Compilation Time: {compile_time:.4f} seconds")
    
    # Benchmark step execution
    num_steps = 20
    start_bench = time.perf_counter()
    for step in range(num_steps):
        loss = train_step(model, optimizer, x, y)
    loss.block_until_ready()
    bench_time = time.perf_counter() - start_bench
    
    avg_step_ms = (bench_time / num_steps) * 1000
    print(f"Average Step Time (over {num_steps} steps): {avg_step_ms:.2f} ms")
    
    # Calculate GFLOPS
    # MLP with 2 layers of size (In -> Hidden -> Out)
    # Layer 1 parameters: 4096 * 8192 = 33.5M
    # Layer 2 parameters: 8192 * 4096 = 33.5M
    # FLOPs per forward pass: 2 * (Batch * In * Hidden + Batch * Hidden * Out) = 2 * (1024 * 4096 * 8192 + 1024 * 8192 * 4096) = 137.4 GFLOPs
    # Backward pass is approx 2x forward pass FLOPs. Total step ~412 GFLOPs.
    total_flops = 3 * 2 * batch_size * in_features * hidden_features
    tflops_metric = (total_flops / (avg_step_ms / 1000)) / 1e12
    print(f"Approximate Compute Performance: {tflops_metric:.4f} TFLOPS")

if __name__ == '__main__':
    main()
