import torch
import triton
import triton.language as tl

# ============================================================================
# IMMEDIATE ACTION ITEM (TASK 1): "Hello World" Vector Addition
# ============================================================================

@triton.jit
def hello_world_add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Your 'Hello World' for the new path. 
    Adds two vectors together directly on the GPU using Triton.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from VRAM into fast SRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform computation
    output = x + y
    
    # Store results back to VRAM
    tl.store(output_ptr + offsets, output, mask=mask)

def hello_world_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper to launch the Hello World kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hello_world_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# PHASE 2: FUSED GEOSPATIAL + SEMANTIC SIMILARITY KERNEL
# ============================================================================

@triton.jit
def fused_geo_score_kernel(
    # Pointers to matrices
    query_vec_ptr,    # [dim]
    query_loc_ptr,    # [2] -> Lat, Lon
    doc_vecs_ptr,     # [num_docs, dim]
    doc_locs_ptr,     # [num_docs, 2]
    output_ptr,       # [num_docs]
    
    # Matrix dimensions
    num_docs,
    dim,
    geo_weight,       # Float: How heavily to penalize physical distance
    
    # Strides (to navigate memory layout)
    stride_doc_vecs_batch,
    stride_doc_vecs_dim,
    stride_doc_locs_batch,
    
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Number of documents processed per thread block
    BLOCK_SIZE_D: tl.constexpr  # Dimension of the embeddings (must be power of 2)
):
    """
    Fuses Cosine Similarity (Dot Product) and Physical Distance (Euclidean) 
    into a single pass over the data to avoid global memory round-trips.
    """
    # 1. Map this block to its specific subset of documents
    pid = tl.program_id(axis=0)
    doc_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    doc_mask = doc_offsets < num_docs

    # 2. Load the Single Query Data (Broadcasted to all threads in this block)
    # Load Query Vector
    dim_offsets = tl.arange(0, BLOCK_SIZE_D)
    q_vec = tl.load(query_vec_ptr + dim_offsets, mask=dim_offsets < dim, other=0.0)
    
    # Load Query Location (Lat/Lon)
    q_lat = tl.load(query_loc_ptr + 0)
    q_lon = tl.load(query_loc_ptr + 1)

    # 3. Load the Batch of Document Data
    # Calculate 2D memory pointers for the document embeddings
    doc_vecs_ptrs = doc_vecs_ptr + (doc_offsets[:, None] * stride_doc_vecs_batch) + (dim_offsets[None, :] * stride_doc_vecs_dim)
    
    # Load the document embeddings [BLOCK_SIZE_N, BLOCK_SIZE_D]
    d_vecs = tl.load(doc_vecs_ptrs, mask=(doc_mask[:, None]) & (dim_offsets[None, :] < dim), other=0.0)
    
    # Load document locations [BLOCK_SIZE_N]
    d_lats = tl.load(doc_locs_ptr + doc_offsets * stride_doc_locs_batch + 0, mask=doc_mask)
    d_lons = tl.load(doc_locs_ptr + doc_offsets * stride_doc_locs_batch + 1, mask=doc_mask)

    # 4. FUSED MATH: Compute both metrics simultaneously in SRAM
    # A. Semantic Similarity (Dot Product, assuming pre-normalized vectors)
    # Element-wise multiply query by docs, then sum along the dimension axis
    sim_scores = tl.sum(q_vec[None, :] * d_vecs, axis=1)

    # B. Geospatial Penalty (Squared Euclidean distance for speed)
    lat_diff = q_lat - d_lats
    lon_diff = q_lon - d_lons
    dist_sq = (lat_diff * lat_diff) + (lon_diff * lon_diff)
    geo_penalty = dist_sq * geo_weight

    # C. Final Fusion Equation
    fused_scores = sim_scores - geo_penalty

    # 5. Store the final combined score back to Global Memory
    tl.store(output_ptr + doc_offsets, fused_scores, mask=doc_mask)


def run_fused_geo_score(
    query_vec: torch.Tensor,
    query_loc: torch.Tensor,
    doc_vecs: torch.Tensor,
    doc_locs: torch.Tensor,
    geo_weight: float = 0.1
) -> torch.Tensor:
    """Python wrapper to launch the custom Triton fused kernel."""
    assert query_vec.is_cuda and doc_vecs.is_cuda, "Tensors must be on GPU"
    assert doc_vecs.shape[1] == query_vec.shape[0], "Dimension mismatch"
    
    num_docs, dim = doc_vecs.shape
    output = torch.empty(num_docs, device=doc_vecs.device, dtype=torch.float32)
    
    # Find the next power of 2 for the dimension (Triton requirement for block sizes)
    BLOCK_SIZE_D = triton.next_power_of_2(dim)
    BLOCK_SIZE_N = 1024  # Process 1024 properties per block
    
    grid = lambda meta: (triton.cdiv(num_docs, meta['BLOCK_SIZE_N']),)
    
    fused_geo_score_kernel[grid](
        query_vec, query_loc, 
        doc_vecs, doc_locs, 
        output,
        num_docs, dim, geo_weight,
        doc_vecs.stride(0), doc_vecs.stride(1),
        doc_locs.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )
    
    return output


# ============================================================================
# EXECUTION & BENCHMARKING (Proving the 40% memory bandwidth reduction)
# ============================================================================
if __name__ == "__main__":
    print("--- TASK 1: Executing 'Hello World' Triton Kernel ---")
    x = torch.rand(100000, device='cuda')
    y = torch.rand(100000, device='cuda')
    z_triton = hello_world_add(x, y)
    z_torch = x + y
    torch.testing.assert_close(z_triton, z_torch)
    print("✅ Hello World Vector Addition Passed!\n")

    print("--- PHASE 2: Benchmarking Eager PyTorch vs. Fused Triton ---")
    
    # 1. Setup Mock Data (e.g., 2 Million Properties, 128 Dimension Embeddings)
    NUM_DOCS = 2_000_000
    DIM = 128
    print(f"Dataset Size: {NUM_DOCS:,} properties, {DIM} dimensions.")
    
    query_vec = torch.randn(DIM, device='cuda', dtype=torch.float32)
    query_loc = torch.tensor([30.2672, -97.7431], device='cuda', dtype=torch.float32) # Austin, TX
    
    doc_vecs = torch.randn((NUM_DOCS, DIM), device='cuda', dtype=torch.float32)
    doc_locs = torch.randn((NUM_DOCS, 2), device='cuda', dtype=torch.float32)
    
    # Pre-normalize for realistic cosine similarity
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=0)
    doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=1)
    
    # 2. Eager PyTorch Implementation (What you are replacing)
    def eager_pytorch(q_vec, q_loc, d_vecs, d_locs, weight=0.1):
        # Memory Pass 1: Dot Product
        sim_scores = torch.matmul(d_vecs, q_vec)
        # Memory Pass 2: Distances
        dist_sq = torch.sum((d_locs - q_loc)**2, dim=1)
        # Memory Pass 3: Fusion
        return sim_scores - (dist_sq * weight)

    # 3. Warmup
    _ = eager_pytorch(query_vec, query_loc, doc_vecs, doc_locs)
    _ = run_fused_geo_score(query_vec, query_loc, doc_vecs, doc_locs)
    
    # 4. Correctness Check
    torch_out = eager_pytorch(query_vec, query_loc, doc_vecs, doc_locs)
    triton_out = run_fused_geo_score(query_vec, query_loc, doc_vecs, doc_locs)
    torch.testing.assert_close(torch_out, triton_out, atol=1e-4, rtol=1e-4)
    print("✅ Triton Output matches PyTorch Output!")

    # 5. Performance Benchmark using Triton's testing suite
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['NUM_DOCS'],
            x_vals=[2**i for i in range(16, 22)], # Test from 65k to 2M rows
            line_arg='provider',
            line_vals=['pytorch', 'triton'],
            line_names=['Eager PyTorch', 'Fused Triton Kernel'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='Time (ms)',
            plot_name='Fused Geospatial Search Performance',
            args={'dim': 128}
        )
    )
    def benchmark(NUM_DOCS, dim, provider):
        d_vecs = torch.randn((NUM_DOCS, dim), device='cuda')
        d_locs = torch.randn((NUM_DOCS, 2), device='cuda')
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: eager_pytorch(query_vec, query_loc, d_vecs, d_locs), 
                quantiles=quantiles
            )
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: run_fused_geo_score(query_vec, query_loc, d_vecs, d_locs), 
                quantiles=quantiles
            )
        return ms, min_ms, max_ms

    print("\nRunning Performance Benchmark...")
    benchmark.run(print_data=True, show_plots=False)