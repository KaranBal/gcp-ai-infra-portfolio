import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import reduce

def _get_next_indices(grid, indices):
    next_indices = []
    carry = True
    for dim_size, index in reversed(list(zip(grid, indices))):
        i = jnp.where(carry, index + 1, index)
        carry = dim_size == i
        next_indices.append(jnp.where(carry, 0, i))
    return tuple(reversed(next_indices))

def test():
    # 1D test: size 16, block_size 4, grid 4
    grid = (4,)
    size = 16
    block_size = 4
    
    # Initialize array with NaNs
    x = jnp.arange(size, dtype=jnp.float32)
    y = jnp.arange(size, dtype=jnp.float32)
    out = jnp.full((size,), jnp.nan, dtype=jnp.float32)
    
    num_iterations = 4
    grid_start_indices = (jnp.int32(0),)
    
    carry = [x, y, out]
    
    def cond(state):
        i, *_ = state
        return i < num_iterations
        
    def body(state):
        i, loop_idx, val_x, val_y, val_out = state
        
        # Log index
        jax.debug.print("Iteration: {i}, loop_idx: {idx}", i=i, idx=loop_idx)
        
        # Compute start index
        # index_map = lambda idx: idx * block_size
        idx = loop_idx[0]
        start_idx = (idx * block_size,)
        
        # Read blocks
        bx = lax.dynamic_slice(val_x, start_idx, (block_size,))
        by = lax.dynamic_slice(val_y, start_idx, (block_size,))
        
        # Kernel computation
        bout = bx + by
        
        # Write block
        val_out_new = lax.dynamic_update_slice(val_out, bout, start_idx)
        
        next_idx = _get_next_indices(grid, loop_idx)
        return (i + 1, next_idx, val_x, val_y, val_out_new)
        
    res_state = lax.while_loop(
        cond,
        body,
        (jnp.int32(0), grid_start_indices, x, y, out)
    )
    
    _, _, _, _, final_out = res_state
    print("Final Output:")
    print(final_out)

if __name__ == "__main__":
    test()
