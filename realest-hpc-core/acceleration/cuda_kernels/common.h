#ifndef REALEST_HPC_CUDA_COMMON_H
#define REALEST_HPC_CUDA_COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace realest {
namespace hpc {
namespace cuda {

/**
 * ============================================================================
 * CUDA ERROR CHECKING MACRO
 * ============================================================================
 * Wraps CUDA API calls and throws a C++ runtime exception if they fail.
 * * Systems Engineering Note: In the raw `geo_attention.cu` file, a failure 
 * just called `exit(EXIT_FAILURE)`. However, when integrated with your Phase 1 
 * gRPC server, a GPU out-of-memory error shouldn't crash the whole server! 
 * Throwing a `std::runtime_error` allows your `WorkerPool` to catch the error 
 * and return a gRPC Status::INTERNAL error to the client cleanly.
 */
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::string error_msg = std::string("CUDA Error at ") + __FILE__ + ":" + \
                                std::to_string(__LINE__) + " - " + \
                                cudaGetErrorString(err); \
        std::cerr << "[GPU Error] " << error_msg << std::endl; \
        throw std::runtime_error(error_msg); \
    } \
} while (0)


/**
 * ============================================================================
 * KERNEL LAUNCH INTERFACES
 * ============================================================================
 * These function prototypes allow standard C++ files (like `worker_pool.cc`) 
 * to invoke CUDA kernels. The C++ code includes this header, and the build 
 * system (like CMake) links the compiled `.o` files together.
 */

/**
 * @brief Launches the fused geospatial and semantic similarity kernel on the GPU.
 * * @param d_query_vec Pointer to the query embedding vector in VRAM.
 * @param query_lat The latitude of the search center.
 * @param query_lon The longitude of the search center.
 * @param d_doc_vecs Pointer to the flattened matrix of document embeddings in VRAM.
 * @param d_doc_lats Pointer to the array of document latitudes in VRAM.
 * @param d_doc_lons Pointer to the array of document longitudes in VRAM.
 * @param d_output_scores Pointer to the pre-allocated output array in VRAM.
 * @param num_docs Total number of documents mapped to this shard/worker node.
 * @param dim The dimensionality of the vector embeddings (e.g., 128).
 * @param geo_weight How heavily to penalize physical distance vs. semantic score.
 */
void LaunchFusedGeoScore(
    const float* d_query_vec, 
    float query_lat, 
    float query_lon,
    const float* d_doc_vecs, 
    const float* d_doc_lats, 
    const float* d_doc_lons,
    float* d_output_scores, 
    int num_docs, 
    int dim, 
    float geo_weight
);

} // namespace cuda
} // namespace hpc
} // namespace realest

#endif // REALEST_HPC_CUDA_COMMON_H