#pragma once

#include "../core/types.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace cphnsw {
namespace cuda {

/**
 * GPU-accelerated CP-LSH Encoder
 *
 * Implements batch encoding using CUDA:
 * 1. Parallel Fast Hadamard Transform (FHT)
 * 2. Parallel diagonal multiplication (random signs)
 * 3. Parallel argmax for code extraction
 *
 * Optimized for H100 with:
 * - Coalesced memory access
 * - Shared memory for FHT butterfly operations
 * - Warp-level reductions for argmax
 */
template<typename ComponentT, size_t K>
class GPUEncoder {
public:
    /**
     * Constructor - allocates GPU memory and generates random signs
     *
     * @param dim Original vector dimension
     * @param seed Random seed for reproducibility
     */
    GPUEncoder(size_t dim, uint64_t seed = 42);

    ~GPUEncoder();

    // Disable copy
    GPUEncoder(const GPUEncoder&) = delete;
    GPUEncoder& operator=(const GPUEncoder&) = delete;

    // Enable move
    GPUEncoder(GPUEncoder&&) noexcept;
    GPUEncoder& operator=(GPUEncoder&&) noexcept;

    /**
     * Encode a batch of vectors on GPU
     *
     * @param h_vectors Host pointer to vectors [num_vecs x dim]
     * @param num_vecs Number of vectors to encode
     * @param h_codes Output host pointer for codes [num_vecs x K]
     */
    void encode_batch(const Float* h_vectors, size_t num_vecs, ComponentT* h_codes);

    /**
     * Encode vectors already on GPU
     *
     * @param d_vectors Device pointer to vectors [num_vecs x padded_dim]
     * @param num_vecs Number of vectors
     * @param d_codes Output device pointer [num_vecs x K]
     */
    void encode_batch_device(const Float* d_vectors, size_t num_vecs, ComponentT* d_codes);

    /**
     * Encode query vectors with magnitude storage (for asymmetric distance)
     *
     * @param h_vectors Host vectors [num_queries x dim]
     * @param num_queries Number of queries
     * @param h_codes Output codes [num_queries x K]
     * @param h_magnitudes Output magnitudes [num_queries x K]
     */
    void encode_queries_batch(const Float* h_vectors, size_t num_queries,
                              ComponentT* h_codes, Float* h_magnitudes);

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }

private:
    size_t dim_;
    size_t padded_dim_;

    // Device memory for random signs: K rotations x 3 layers x padded_dim
    int8_t* d_signs_;

    // Temporary buffers
    Float* d_vectors_padded_;  // Padded input vectors
    Float* d_rotated_;         // After rotation chain
    size_t max_batch_size_;

    cudaStream_t stream_;

    void allocate_buffers(size_t batch_size);
    void generate_signs(uint64_t seed);
};

/**
 * GPU Brute Force Search using cuBLAS
 *
 * Computes: similarities = queries @ base^T
 * Then extracts top-k using custom CUDA kernel
 */
class GPUBruteForce {
public:
    /**
     * @param dim Vector dimension
     * @param num_base Number of base vectors
     */
    GPUBruteForce(size_t dim, size_t num_base);
    ~GPUBruteForce();

    /**
     * Upload base vectors to GPU (call once during index build)
     */
    void set_base_vectors(const Float* h_base, size_t num_base);

    /**
     * Search for top-k nearest neighbors
     *
     * @param h_queries Query vectors [num_queries x dim]
     * @param num_queries Number of queries
     * @param k Number of neighbors to return
     * @param h_indices Output indices [num_queries x k]
     * @param h_distances Output distances [num_queries x k]
     */
    void search(const Float* h_queries, size_t num_queries, size_t k,
                uint32_t* h_indices, Float* h_distances);

    /**
     * Re-rank candidates using true cosine similarity
     *
     * @param h_queries Query vectors [num_queries x dim]
     * @param h_candidate_ids Candidate IDs [num_queries x num_candidates]
     * @param num_queries Number of queries
     * @param num_candidates Candidates per query
     * @param k Final top-k to return
     * @param h_indices Output indices [num_queries x k]
     * @param h_distances Output distances [num_queries x k]
     */
    void rerank(const Float* h_queries, const uint32_t* h_candidate_ids,
                size_t num_queries, size_t num_candidates, size_t k,
                uint32_t* h_indices, Float* h_distances);

private:
    size_t dim_;
    size_t num_base_;

    Float* d_base_;           // Base vectors on GPU [num_base x dim]
    Float* d_queries_;        // Query buffer
    Float* d_similarities_;   // Similarity matrix
    uint32_t* d_indices_;     // Top-k indices
    Float* d_distances_;      // Top-k distances

    void* cublas_handle_;
    cudaStream_t stream_;

    size_t max_queries_;
};

// CUDA error checking macro - throws exception for RAII cleanup
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string msg = std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err); \
            throw std::runtime_error(msg); \
        } \
    } while(0)

}  // namespace cuda
}  // namespace cphnsw
