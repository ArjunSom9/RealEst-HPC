#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

/**
 * ============================================================================
 * PHASE 2: FUSED GEOSPATIAL + SEMANTIC SIMILARITY KERNEL (RAW CUDA)
 * ============================================================================
 * * Hardware Architecture Notes:
 * 1. __restrict__ : Promises the compiler that pointers don't alias, enabling 
 * aggressive read-only caching via the Texture/L1 cache.
 * 2. __shared__ memory : The query vector is identical for every document. Instead 
 * of having thousands of threads hit Global Memory (VRAM) to read the same 
 * query floats, the thread block collaboratively loads the query into Shared 
 * Memory (SRAM) once, dropping memory latency from ~400 cycles to ~30 cycles.
 */
__global__ void FusedGeoScoreKernel(
    const float* __restrict__ query_vec,
    float query_lat,
    float query_lon,
    const float* __restrict__ doc_vecs,
    const float* __restrict__ doc_lats,
    const float* __restrict__ doc_lons,
    float* __restrict__ output_scores,
    int num_docs,
    int dim,
    float geo_weight
) {
    // Dynamically allocated Shared Memory (SRAM) for the query vector
    extern __shared__ float shared_query[];

    // 1. COLLABORATIVE LOAD: Load the query vector from VRAM into SRAM
    // Each thread in the block loads a fraction of the vector.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        shared_query[i] = query_vec[i];
    }

    // Barrier synchronization: Ensure the entire query vector is in SRAM 
    // before any thread starts computing document scores.
    __syncthreads();

    // 2. MAP THREAD TO DOCUMENT
    // Global thread index maps 1:1 to a specific property in the database
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (doc_idx < num_docs) {
        
        // 3. SEMANTIC SIMILARITY (Dot Product)
        float dot_product = 0.0f;
        
        // Offset pointer to this specific document's embedding
        // Note: For max HPC bandwidth, embeddings should ideally be Column-Major 
        // to ensure coalesced memory access across the warp. We assume Row-Major 
        // here to match standard PyTorch layouts.
        const float* my_doc_vec = &doc_vecs[doc_idx * dim];
        
        // Unroll the loop slightly if dimension is known (e.g., #pragma unroll)
        for (int i = 0; i < dim; ++i) {
            dot_product += shared_query[i] * my_doc_vec[i];
        }

        // 4. GEOSPATIAL PENALTY
        float d_lat = doc_lats[doc_idx];
        float d_lon = doc_lons[doc_idx];
        
        float lat_diff = query_lat - d_lat;
        float lon_diff = query_lon - d_lon;
        
        // Squared distance avoids the expensive sqrtf() hardware instruction
        float dist_sq = (lat_diff * lat_diff) + (lon_diff * lon_diff);

        // 5. FUSED WRITE
        // Calculate the final score and write to Global VRAM exactly once.
        output_scores[doc_idx] = dot_product - (geo_weight * dist_sq);
    }
}

// ============================================================================
// C++ HOST LAUNCHER
// ============================================================================

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void LaunchFusedGeoScore(
    const float* d_query_vec, float query_lat, float query_lon,
    const float* d_doc_vecs, const float* d_doc_lats, const float* d_doc_lons,
    float* d_output_scores, int num_docs, int dim, float geo_weight
) {
    // 256 threads per block is a standard tuning starting point for Ampere/Hopper GPUs
    int threads_per_block = 256; 
    
    // Calculate required blocks to cover all documents
    int blocks_per_grid = (num_docs + threads_per_block - 1) / threads_per_block;

    // Calculate dynamic shared memory size required for the query vector
    size_t shared_mem_bytes = dim * sizeof(float);

    // Launch the kernel
    FusedGeoScoreKernel<<<blocks_per_grid, threads_per_block, shared_mem_bytes>>>(
        d_query_vec, query_lat, query_lon,
        d_doc_vecs, d_doc_lats, d_doc_lons,
        d_output_scores, num_docs, dim, geo_weight
    );

    // Catch synchronous launch errors
    CHECK_CUDA(cudaGetLastError());
}

// ============================================================================
// BENCHMARK AND TEST
// ============================================================================

int main() {
    std::cout << "--- RealEst-HPC: Raw CUDA Fused Kernel Benchmark ---\n";

    int num_docs = 2000000; // 2 Million properties
    int dim = 128;          // 128-dimensional embeddings
    float geo_weight = 0.1f;
    float q_lat = 30.2672f;
    float q_lon = -97.7431f;

    size_t vec_size = num_docs * dim * sizeof(float);
    size_t loc_size = num_docs * sizeof(float);
    size_t out_size = num_docs * sizeof(float);

    // 1. Allocate Host (CPU) Memory
    std::vector<float> h_query_vec(dim, 1.0f / sqrt(dim)); // Dummy normalized query
    std::vector<float> h_doc_vecs(num_docs * dim, 1.0f / sqrt(dim));
    std::vector<float> h_doc_lats(num_docs, 30.2700f);
    std::vector<float> h_doc_lons(num_docs, -97.7400f);
    std::vector<float> h_output_scores(num_docs, 0.0f);

    // 2. Allocate Device (GPU) Memory
    float *d_query_vec, *d_doc_vecs, *d_doc_lats, *d_doc_lons, *d_output_scores;
    CHECK_CUDA(cudaMalloc(&d_query_vec, dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_doc_vecs, vec_size));
    CHECK_CUDA(cudaMalloc(&d_doc_lats, loc_size));
    CHECK_CUDA(cudaMalloc(&d_doc_lons, loc_size));
    CHECK_CUDA(cudaMalloc(&d_output_scores, out_size));

    // 3. Copy Data to GPU
    CHECK_CUDA(cudaMemcpy(d_query_vec, h_query_vec.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_doc_vecs, h_doc_vecs.data(), vec_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_doc_lats, h_doc_lats.data(), loc_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_doc_lons, h_doc_lons.data(), loc_size, cudaMemcpyHostToDevice));

    // 4. Warmup run
    LaunchFusedGeoScore(d_query_vec, q_lat, q_lon, d_doc_vecs, d_doc_lats, d_doc_lons, 
                        d_output_scores, num_docs, dim, geo_weight);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Benchmark timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run the kernel 100 times to get a stable average
    int iterations = 100;
    for(int i = 0; i < iterations; i++) {
        LaunchFusedGeoScore(d_query_vec, q_lat, q_lon, d_doc_vecs, d_doc_lats, d_doc_lons, 
                            d_output_scores, num_docs, dim, geo_weight);
    }
    
    // Wait for the GPU to finish all queued work
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 6. Output metrics
    std::chrono::duration<double, std::milli> duration = (end - start) / iterations;
    
    // Copy a single result back to verify it executed
    CHECK_CUDA(cudaMemcpy(h_output_scores.data(), d_output_scores, out_size, cudaMemcpyDeviceToHost));

    std::cout << "Target Documents : " << num_docs << "\n";
    std::cout << "Vector Dimension : " << dim << "\n";
    std::cout << "Avg Compute Time : " << duration.count() << " ms\n";
    std::cout << "Sample Score[0]  : " << h_output_scores[0] << "\n";
    std::cout << "Result           : " << (duration.count() < 1.0 ? "✅ SUB-MILLISECOND LATENCY ACHIEVED" : "❌ TOO SLOW") << "\n";

    // 7. Cleanup
    cudaFree(d_query_vec); cudaFree(d_doc_vecs); 
    cudaFree(d_doc_lats); cudaFree(d_doc_lons); cudaFree(d_output_scores);

    return 0;
}