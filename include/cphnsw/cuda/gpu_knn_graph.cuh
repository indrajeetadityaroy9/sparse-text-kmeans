#pragma once

#include "../core/types.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cphnsw {
namespace cuda {

/**
 * GPU k-NN Graph Builder for CAGRA-style construction.
 *
 * Builds a k-nearest neighbor graph for N vectors using tiled GPU computation.
 * The algorithm:
 * 1. Process vectors in query batches (Q vectors at a time)
 * 2. For each query batch, tile through all base vectors
 * 3. Use cuBLAS SGEMM for similarity computation
 * 4. Fused top-k merge after each tile (avoids O(N²) memory)
 * 5. Output: N × k edge list
 *
 * Memory strategy for 100M vectors:
 * - Query batch: 10K vectors
 * - Base tile: 1M vectors
 * - Temp buffer: 10K × 1M × 4B = 40GB (reused per tile)
 * - Running top-k: 10K × k × 8B = minimal
 */
class GPUKNNGraphBuilder {
public:
    /**
     * Constructor
     * @param dim Vector dimension
     * @param k Number of neighbors per node
     */
    GPUKNNGraphBuilder(size_t dim, size_t k);
    ~GPUKNNGraphBuilder();

    // Disable copy
    GPUKNNGraphBuilder(const GPUKNNGraphBuilder&) = delete;
    GPUKNNGraphBuilder& operator=(const GPUKNNGraphBuilder&) = delete;

    /**
     * Build k-NN graph using tiled GPU computation.
     *
     * @param h_vectors Host pointer to vectors [N x dim], row-major
     * @param N Number of vectors
     * @param h_neighbors Output: neighbor indices [N x k]
     * @param h_distances Output: neighbor distances [N x k] (optional, can be nullptr)
     * @param query_batch_size Number of queries per batch (default: auto)
     * @param base_tile_size Number of base vectors per tile (default: auto)
     */
    void build(const Float* h_vectors, size_t N,
               uint32_t* h_neighbors, Float* h_distances = nullptr,
               size_t query_batch_size = 0, size_t base_tile_size = 0);

    /**
     * Build k-NN graph with vectors already on GPU.
     * Useful when vectors are generated/transformed on GPU.
     */
    void build_device(const Float* d_vectors, size_t N,
                      uint32_t* h_neighbors, Float* h_distances = nullptr,
                      size_t query_batch_size = 0, size_t base_tile_size = 0);

    /**
     * Get recommended batch sizes for given N and available GPU memory.
     */
    static void get_recommended_sizes(size_t N, size_t dim, size_t k,
                                      size_t available_memory_gb,
                                      size_t& query_batch, size_t& base_tile);

private:
    size_t dim_;
    size_t k_;

    // GPU resources
    void* cublas_handle_;
    cudaStream_t stream_;

    // Persistent buffers (allocated once, reused)
    Float* d_similarities_;      // Query batch × base tile similarity matrix
    Float* d_running_topk_dist_; // Query batch × k running top-k distances
    uint32_t* d_running_topk_idx_; // Query batch × k running top-k indices

    size_t max_query_batch_;
    size_t max_base_tile_;

    void allocate_buffers(size_t query_batch, size_t base_tile);
    void free_buffers();

    /**
     * Process one tile: compute similarities and merge into running top-k.
     */
    void process_tile(const Float* d_queries, size_t num_queries,
                      const Float* d_base_tile, size_t tile_size,
                      size_t tile_base_idx);

    /**
     * Initialize running top-k with worst values.
     */
    void init_running_topk(size_t num_queries);
};

/**
 * Edge list representation of k-NN graph.
 * More efficient for graph construction than N×k matrix.
 */
struct KNNEdgeList {
    std::vector<uint32_t> sources;    // Source node IDs
    std::vector<uint32_t> targets;    // Target node IDs
    std::vector<float> distances;     // Edge distances (optional)

    size_t num_nodes;
    size_t k;

    /**
     * Convert from N×k matrix format to edge list.
     */
    static KNNEdgeList from_matrix(const uint32_t* neighbors, const float* distances,
                                   size_t N, size_t k);

    /**
     * Get neighbors of a specific node.
     */
    std::pair<const uint32_t*, size_t> get_neighbors(uint32_t node_id) const;
};

// CUDA error checking macro - throws exception for RAII cleanup
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string msg = std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err); \
            throw std::runtime_error(msg); \
        } \
    } while(0)
#endif

}  // namespace cuda
}  // namespace cphnsw
