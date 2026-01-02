#pragma once

#include "../core/types.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace cphnsw {
namespace cuda {

/**
 * GPU-Accelerated Batch Search for 2x H100 Configuration.
 *
 * For high-throughput search, this uses cuBLAS SGEMM to compute
 * all query-base similarities in parallel, then extracts top-k.
 *
 * Architecture:
 * - GPU 0: First half of base vectors
 * - GPU 1: Second half of base vectors
 * - Both GPUs process all queries in parallel
 * - Results merged on host
 *
 * Performance on H100:
 * - cuBLAS SGEMM: ~1.5 TFLOPS sustained for matmul
 * - 10K queries × 1M vectors × 128 dims = 1.28 TFLOPS
 * - Expected throughput: ~100K+ QPS with batching
 */
class GPUBatchSearch {
public:
    /**
     * Constructor - initializes GPU resources on both GPUs.
     *
     * @param dim Vector dimension
     * @param num_gpus Number of GPUs to use (default: 2)
     */
    GPUBatchSearch(size_t dim, int num_gpus = 2);
    ~GPUBatchSearch();

    // Disable copy
    GPUBatchSearch(const GPUBatchSearch&) = delete;
    GPUBatchSearch& operator=(const GPUBatchSearch&) = delete;

    /**
     * Upload base vectors to GPUs (call once during index build).
     * Vectors are partitioned across GPUs.
     *
     * @param h_vectors Host pointer to normalized vectors [N x dim]
     * @param N Number of base vectors
     */
    void set_base_vectors(const Float* h_vectors, size_t N);

    /**
     * Batch search: find top-k nearest neighbors for multiple queries.
     *
     * Uses cuBLAS SGEMM for similarity computation, then GPU top-k extraction.
     * This is brute-force search but extremely fast on H100.
     *
     * @param h_queries Query vectors [num_queries x dim]
     * @param num_queries Number of queries
     * @param k Number of neighbors to return
     * @param h_indices Output: neighbor indices [num_queries x k]
     * @param h_distances Output: neighbor distances [num_queries x k]
     */
    void search_batch(const Float* h_queries, size_t num_queries, size_t k,
                      uint32_t* h_indices, Float* h_distances);

    /**
     * Compute ground truth (brute-force k-NN) for evaluation.
     *
     * @param h_queries Query vectors [num_queries x dim]
     * @param num_queries Number of queries
     * @param k Number of neighbors
     * @param h_indices Output: ground truth indices [num_queries x k]
     */
    void compute_ground_truth(const Float* h_queries, size_t num_queries, size_t k,
                              uint32_t* h_indices);

    /**
     * Rerank candidates using true cosine similarity.
     *
     * Given candidate IDs from graph search, compute exact similarities
     * and return top-k.
     *
     * @param h_queries Query vectors [num_queries x dim]
     * @param h_candidate_ids Candidate IDs [num_queries x num_candidates]
     * @param num_queries Number of queries
     * @param num_candidates Candidates per query
     * @param k Final top-k to return
     * @param h_indices Output: reranked indices [num_queries x k]
     * @param h_distances Output: reranked distances [num_queries x k]
     */
    void rerank_batch(const Float* h_queries, const uint32_t* h_candidate_ids,
                      size_t num_queries, size_t num_candidates, size_t k,
                      uint32_t* h_indices, Float* h_distances);

    size_t dim() const { return dim_; }
    size_t num_base() const { return num_base_; }
    int num_gpus() const { return num_gpus_; }

private:
    size_t dim_;
    size_t num_base_;
    int num_gpus_;

    // Per-GPU resources
    struct GPUContext {
        int device_id;
        cublasHandle_t cublas_handle;
        cudaStream_t stream;

        Float* d_base;           // Base vectors partition [partition_size x dim]
        size_t partition_start;  // Starting index in global base
        size_t partition_size;   // Number of vectors in this partition

        Float* d_queries;        // Query buffer [max_queries x dim]
        Float* d_similarities;   // Similarity matrix [max_queries x partition_size]
        Float* d_topk_distances; // Top-k distances [max_queries x k]
        uint32_t* d_topk_indices;// Top-k indices [max_queries x k]

        size_t max_queries;
    };

    std::vector<GPUContext> gpu_contexts_;

    void allocate_query_buffers(size_t num_queries, size_t k);
    void free_resources();

    // Top-k extraction kernel wrapper
    void extract_topk(int gpu_idx, size_t num_queries, size_t k);

    // Merge top-k results from multiple GPUs
    void merge_topk_results(size_t num_queries, size_t k,
                           uint32_t* h_indices, Float* h_distances);
};

/**
 * GPU Ground Truth Computer - optimized for evaluation.
 *
 * Computes exact k-NN using cosine similarity for benchmark evaluation.
 * Much faster than CPU brute force for large datasets.
 */
class GPUGroundTruth {
public:
    GPUGroundTruth(size_t dim, int num_gpus = 2);
    ~GPUGroundTruth();

    /**
     * Compute ground truth for normalized vectors.
     *
     * @param h_base Base vectors [N x dim]
     * @param N Number of base vectors
     * @param h_queries Query vectors [num_queries x dim]
     * @param num_queries Number of queries
     * @param k Number of neighbors
     * @param h_gt_indices Output: ground truth indices [num_queries x k]
     */
    void compute(const Float* h_base, size_t N,
                 const Float* h_queries, size_t num_queries, size_t k,
                 uint32_t* h_gt_indices);

private:
    size_t dim_;
    int num_gpus_;

    struct GPUContext {
        int device_id;
        cublasHandle_t cublas_handle;
        cudaStream_t stream;
        Float* d_base;
        Float* d_queries;
        Float* d_similarities;
        uint32_t* d_topk_indices;
        Float* d_topk_distances;
    };

    std::vector<GPUContext> contexts_;
};

// Utility: Check GPU memory
inline size_t get_gpu_free_memory(int device = 0) {
    size_t free_mem, total_mem;
    cudaSetDevice(device);
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

// Utility: Get number of available GPUs
inline int get_num_gpus() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

}  // namespace cuda
}  // namespace cphnsw
