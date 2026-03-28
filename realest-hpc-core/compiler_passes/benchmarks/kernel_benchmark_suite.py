import os
import sys
import torch
import triton
import triton.testing

# Add the acceleration directory to the Python path so we can import our custom kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from acceleration.triton_kernels.fused_geo_score import run_fused_geo_score
except ImportError:
    print("Error: Could not import run_fused_geo_score. Ensure the path is correct.")
    sys.exit(1)

# ============================================================================
# BASELINE IMPLEMENTATIONS
# ============================================================================

def eager_pytorch(q_vec, q_loc, d_vecs, d_locs, weight=0.1):
    """
    Standard PyTorch implementation.
    This is memory-bandwidth bound because it materializes intermediate 
    tensors (sim_scores, dist_sq) into global VRAM before combining them.
    """
    # 1. Dot Product (Read N*D, Read D, Write N)
    sim_scores = torch.matmul(d_vecs, q_vec)
    
    # 2. Distance Squared (Read N*2, Read 2, Write N)
    dist_sq = torch.sum((d_locs - q_loc)**2, dim=1)
    
    # 3. Final Fusion (Read N, Read N, Write N)
    fused = sim_scores - (dist_sq * weight)
    return fused

# Compile the eager implementation using PyTorch 2.0 Inductor
# This will be used to see if PyTorch's automatic compiler can beat our hand-written Triton kernel.
compiled_pytorch = torch.compile(eager_pytorch)

# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def measure_peak_memory(func, *args):
    """Measures the peak VRAM allocated during the function's execution."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Run once to warm up the allocator
    _ = func(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Run the actual measurement
    _ = func(*args)
    torch.cuda.synchronize()
    
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 * 1024) # Return in MB

def run_benchmark():
    print("===================================================================")
    print("🚀 RealEst-HPC: Geospatial Vector Search Benchmark Suite")
    print("===================================================================")
    
    # Define test sizes (Number of properties in the database shard)
    test_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
    dim = 128
    
    print(f"{'Num Docs':<12} | {'Method':<15} | {'Latency (ms)':<12} | {'Peak VRAM (MB)':<15} | {'Mem Saved (%)':<12}")
    print("-" * 75)
    
    for num_docs in test_sizes:
        # Generate mock data on the GPU
        q_vec = torch.randn(dim, device='cuda', dtype=torch.float32)
        q_loc = torch.tensor([30.2672, -97.7431], device='cuda', dtype=torch.float32)
        
        d_vecs = torch.randn((num_docs, dim), device='cuda', dtype=torch.float32)
        d_locs = torch.randn((num_docs, 2), device='cuda', dtype=torch.float32)
        
        # Normalize
        q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=0)
        d_vecs = torch.nn.functional.normalize(d_vecs, p=2, dim=1)

        # --------------------------------------------------------------------
        # 1. EAGER PYTORCH (Baseline)
        # --------------------------------------------------------------------
        eager_ms, min_eager, max_eager = triton.testing.do_bench(
            lambda: eager_pytorch(q_vec, q_loc, d_vecs, d_locs), 
            quantiles=[0.5, 0.2, 0.8]
        )
        eager_mem = measure_peak_memory(eager_pytorch, q_vec, q_loc, d_vecs, d_locs)
        
        print(f"{num_docs:<12} | {'Eager PyTorch':<15} | {eager_ms:<12.3f} | {eager_mem:<15.2f} | {'Baseline':<12}")

        # --------------------------------------------------------------------
        # 2. TORCH.COMPILE (Inductor)
        # --------------------------------------------------------------------
        # Warmup the compiler
        _ = compiled_pytorch(q_vec, q_loc, d_vecs, d_locs)
        
        compile_ms, _, _ = triton.testing.do_bench(
            lambda: compiled_pytorch(q_vec, q_loc, d_vecs, d_locs), 
            quantiles=[0.5, 0.2, 0.8]
        )
        compile_mem = measure_peak_memory(compiled_pytorch, q_vec, q_loc, d_vecs, d_locs)
        
        mem_saved_compile = ((eager_mem - compile_mem) / eager_mem) * 100 if eager_mem > 0 else 0
        print(f"{num_docs:<12} | {'torch.compile':<15} | {compile_ms:<12.3f} | {compile_mem:<15.2f} | {mem_saved_compile:>5.1f}%")

        # --------------------------------------------------------------------
        # 3. CUSTOM FUSED TRITON KERNEL
        # --------------------------------------------------------------------
        triton_ms, _, _ = triton.testing.do_bench(
            lambda: run_fused_geo_score(q_vec, q_loc, d_vecs, d_locs), 
            quantiles=[0.5, 0.2, 0.8]
        )
        triton_mem = measure_peak_memory(run_fused_geo_score, q_vec, q_loc, d_vecs, d_locs)
        
        mem_saved_triton = ((eager_mem - triton_mem) / eager_mem) * 100 if eager_mem > 0 else 0
        print(f"{num_docs:<12} | {'Fused Triton':<15} | {triton_ms:<12.3f} | {triton_mem:<15.2f} | {mem_saved_triton:>5.1f}%")
        print("-" * 75)

    print("\n✅ Benchmark Complete.")
    print("Phase 4 Validation:")
    print("1. Compare 'Fused Triton' Latency to ensure sub-millisecond scaling.")
    print("2. Check 'Mem Saved (%)' to validate the 40% memory bandwidth reduction goal.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA is not available. This benchmark requires an NVIDIA GPU.")
        sys.exit(1)
        
    run_benchmark()