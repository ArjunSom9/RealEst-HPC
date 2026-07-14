#include <iostream>
#include <vector>
#include <cstdint>
#include <immintrin.h> 
#include <chrono>
#include <random>

namespace realest {
namespace hpc {
namespace search {

// Section 5: Flat-array contiguous memory layout.
// Nodes and edges are completely devoid of standard C++ heap pointers 
// to prevent catastrophic CPU cache misses during traversal.

struct FlatNode {
    // Offset to a giant contiguous 1D float array instead of float*
    uint32_t vector_offset; 
    
    // Offset to a contiguous 1D uint32_t array for graph neighbors
    uint32_t edge_list_offset; 
    
    uint16_t num_edges;
    
    // The GeoHash prefix this node belongs to (for spatial sharding verification)
    char geohash[5]; 
};

class FlatHNSWIndex {
private:
    // Massive flat arrays locked into the server's RAM
    std::vector<FlatNode> nodes_;
    std::vector<uint32_t> edges_;
    std::vector<float> vectors_;
    
    int dim_;
    int max_edges_;

public:
    FlatHNSWIndex(int max_elements, int dim, int m_parameter) 
        : dim_(dim), max_edges_(m_parameter) {
        
        // Pre-allocate to prevent dynamic resizing pauses (C++ equivalent of GC)
        nodes_.reserve(max_elements);
        vectors_.reserve(max_elements * dim);
        edges_.reserve(max_elements * m_parameter * 5); 
    }

    void AddDocument(const std::vector<float>& vec, const char* geo) {
        uint32_t v_offset = vectors_.size();
        for(float val : vec) vectors_.push_back(val);
        
        FlatNode node;
        node.vector_offset = v_offset;
        node.edge_list_offset = edges_.size();
        node.num_edges = 0;
        for(int i=0; i<4; i++) node.geohash[i] = geo[i];
        node.geohash[4] = '\0';
        
        nodes_.push_back(node);
    }

    // Executes distance calculations via 256-bit registers.
    // This processes 8 floating-point differences simultaneously in a single CPU cycle.
    inline float ComputeSIMDDistance(const float* vecA, const float* vecB) const {
        __m256 sum256 = _mm256_setzero_ps();
        int i = 0;
        
        // Loop unrolling for SIMD vectorization
        for (; i <= dim_ - 8; i += 8) {
            __m256 a = _mm256_loadu_ps(vecA + i);
            __m256 b = _mm256_loadu_ps(vecB + i);
            __m256 diff = _mm256_sub_ps(a, b);
            __m256 sq = _mm256_mul_ps(diff, diff);
            sum256 = _mm256_add_ps(sum256, sq);
        }
        
        // Horizontal reduction of the AVX register
        alignas(32) float buffer[8];
        _mm256_store_ps(buffer, sum256);
        float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] + 
                         buffer[4] + buffer[5] + buffer[6] + buffer[7];
                         
        // Tail cleanup for dimensions not cleanly divisible by 8
        for (; i < dim_; ++i) {
            float diff = vecA[i] - vecB[i];
            distance += diff * diff;
        }
        return distance;
    }

    void BenchmarkRetrieval(const std::vector<float>& query) {
        float min_dist = 999999.0f;
        uint32_t best_node = 0;
        
        const float* q_ptr = query.data();
        const float* memory_block = vectors_.data();
        
        // Linear scan demonstrating the raw speed of contiguous AVX operations.
        // In the full HNSW implementation, this same ComputeSIMDDistance function 
        // is called dynamically as you navigate the edge_list_offsets.
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const float* doc_ptr = memory_block + nodes_[i].vector_offset;
            float dist = ComputeSIMDDistance(q_ptr, doc_ptr);
            if (dist < min_dist) {
                min_dist = dist;
                best_node = i;
            }
        }
    }
};

} // search
} // hpc
} // realest

int main() {
    std::cout << "=== Phase 1: C++ Flat-Memory Engine Benchmark ===\n";
    
    int num_docs = 100000;
    int dim = 128;
    
    realest::hpc::search::FlatHNSWIndex index(num_docs, dim, 16);
    
    std::cout << "Allocating and filling flat memory graph with " << num_docs << " nodes...\n";
    std::vector<float> dummy_vec(dim, 0.5f);
    for(int i=0; i<num_docs; i++) {
        index.AddDocument(dummy_vec, "9q5c");
    }
    
    std::vector<float> query(dim, 0.55f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute 1000 queries to test sub-millisecond throughput
    int iterations = 1000;
    for(int i=0; i<iterations; i++) {
        index.BenchmarkRetrieval(query);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start) / iterations;
    
    std::cout << "Avg Query Latency: " << duration.count() << " ms\n";
    if (duration.count() < 1.0) {
        std::cout << "SUB-MILLISECOND LATENCY ACHIEVED (CPU SIMD)\n";
    } else {
        std::cout << "Tuning required for sub-millisecond bounds.\n";
    }
    
    return 0;
}