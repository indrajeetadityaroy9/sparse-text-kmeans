/**
 * GPU-Accelerated Batch Search Implementation for 2x H100.
 *
 * Uses cuBLAS SGEMM for similarity computation and custom CUDA kernels
 * for top-k extraction. Designed for maximum throughput on H100.
 */

#include "../../include/cphnsw/cuda/gpu_search.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>

namespace cphnsw {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

// =============================================================================
// Top-K Extraction Kernel (Bitonic Sort based)
// =============================================================================

/**
 * Extract top-k from similarity row using partial bitonic sort.
 *
 * Each block processes one query's similarities.
 * Uses shared memory for fast sorting.
 */
__global__ void search_topk_kernel(
    const float* __restrict__ similarities,  // [num_queries x N]
    uint32_t* __restrict__ topk_indices,     // [num_queries x k]
    float* __restrict__ topk_distances,      // [num_queries x k]
    size_t N,
    size_t k,
    size_t partition_offset)  // Add offset to indices for multi-GPU
{
    extern __shared__ char shared_mem[];

    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory layout: [block_size] pairs of (similarity, index)
    float* s_sims = reinterpret_cast<float*>(shared_mem);
    uint32_t* s_indices = reinterpret_cast<uint32_t*>(s_sims + block_size);

    const float* query_sims = similarities + query_idx * N;

    // Initialize with worst values
    float best_sim = -1e30f;
    uint32_t best_idx = 0;

    // Find local maximum for this thread
    for (size_t i = tid; i < N; i += block_size) {
        float sim = query_sims[i];
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = static_cast<uint32_t>(i);
        }
    }

    s_sims[tid] = best_sim;
    s_indices[tid] = best_idx + partition_offset;
    __syncthreads();

    // Simple extraction of top-k (k is usually small, like 10-200)
    // This is a simple O(k*blockSize) approach suitable for small k
    if (tid < k) {
        float* out_dist = topk_distances + query_idx * k;
        uint32_t* out_idx = topk_indices + query_idx * k;

        // Each of first k threads finds the (tid+1)-th best
        // Simple but works for small k
        float threshold = 1e30f;
        for (int rank = 0; rank <= tid; ++rank) {
            float best = -1e30f;
            uint32_t best_i = 0;
            for (int j = 0; j < block_size; ++j) {
                float v = s_sims[j];
                if (v > best && (rank == 0 || v < threshold)) {
                    best = v;
                    best_i = s_indices[j];
                }
            }
            if (rank == tid) {
                out_dist[tid] = -best;  // Convert to distance (negative similarity)
                out_idx[tid] = best_i;
            }
            threshold = best;
        }
    }
}

/**
 * More efficient top-k for larger k using heap.
 * Each warp maintains a min-heap of size k.
 */
__global__ void search_topk_heap_kernel(
    const float* __restrict__ similarities,
    uint32_t* __restrict__ topk_indices,
    float* __restrict__ topk_distances,
    size_t N,
    size_t k,
    size_t partition_offset)
{
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const float* query_sims = similarities + query_idx * N;

    // Each thread maintains local top-k (simplified: just track best seen)
    // For production, use proper heap-based approach

    // Simple approach: each thread scans a portion and keeps best k
    extern __shared__ char shared_mem[];
    float* s_sims = reinterpret_cast<float*>(shared_mem);
    uint32_t* s_indices = reinterpret_cast<uint32_t*>(s_sims + blockDim.x * k);

    // Initialize local heap
    for (int i = 0; i < k && tid * k + i < blockDim.x * k; ++i) {
        s_sims[tid * k + i] = -1e30f;
        s_indices[tid * k + i] = 0;
    }
    __syncthreads();

    // Scan similarities
    for (size_t i = tid; i < N; i += blockDim.x) {
        float sim = query_sims[i];

        // Insert into thread's local top-k (simple insertion sort)
        float* local_sims = s_sims + tid * k;
        uint32_t* local_indices = s_indices + tid * k;

        if (sim > local_sims[k-1]) {
            // Find insertion point
            int pos = k - 1;
            while (pos > 0 && sim > local_sims[pos-1]) {
                local_sims[pos] = local_sims[pos-1];
                local_indices[pos] = local_indices[pos-1];
                pos--;
            }
            local_sims[pos] = sim;
            local_indices[pos] = static_cast<uint32_t>(i) + partition_offset;
        }
    }
    __syncthreads();

    // Merge all threads' top-k (done by first warp)
    if (warp_id == 0 && lane_id == 0) {
        float* out_dist = topk_distances + query_idx * k;
        uint32_t* out_idx = topk_indices + query_idx * k;

        // Simple merge: collect all candidates and sort
        // For production, use proper parallel merge
        struct Candidate {
            float sim;
            uint32_t idx;
        };

        // Collect best from each thread
        Candidate candidates[1024];  // Assuming max 1024 threads
        int num_candidates = 0;

        for (int t = 0; t < blockDim.x; ++t) {
            for (int j = 0; j < k; ++j) {
                candidates[num_candidates].sim = s_sims[t * k + j];
                candidates[num_candidates].idx = s_indices[t * k + j];
                num_candidates++;
                if (num_candidates >= 1024) break;
            }
            if (num_candidates >= 1024) break;
        }

        // Sort candidates (simple insertion sort for now)
        for (int i = 1; i < num_candidates; ++i) {
            Candidate key = candidates[i];
            int j = i - 1;
            while (j >= 0 && candidates[j].sim < key.sim) {
                candidates[j + 1] = candidates[j];
                j--;
            }
            candidates[j + 1] = key;
        }

        // Output top-k
        for (int i = 0; i < k; ++i) {
            out_dist[i] = -candidates[i].sim;  // Convert to distance
            out_idx[i] = candidates[i].idx;
        }
    }
}

/**
 * Simple but fast top-k for small k (< 64).
 * Uses register-based selection.
 */
template<int K_MAX>
__global__ void search_topk_simple_kernel(
    const float* __restrict__ similarities,
    uint32_t* __restrict__ topk_indices,
    float* __restrict__ topk_distances,
    size_t N,
    size_t k,
    size_t partition_offset)
{
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const float* query_sims = similarities + query_idx * N;

    // Each thread maintains its own top-k in registers
    float local_topk[K_MAX];
    uint32_t local_idx[K_MAX];

    for (int i = 0; i < K_MAX; ++i) {
        local_topk[i] = -1e30f;
        local_idx[i] = 0;
    }

    // Scan assigned portion
    for (size_t i = tid; i < N; i += blockDim.x) {
        float sim = query_sims[i];

        // Insert if better than worst in local top-k
        if (sim > local_topk[k-1]) {
            // Shift and insert
            int pos = k - 1;
            while (pos > 0 && sim > local_topk[pos-1]) {
                local_topk[pos] = local_topk[pos-1];
                local_idx[pos] = local_idx[pos-1];
                pos--;
            }
            local_topk[pos] = sim;
            local_idx[pos] = static_cast<uint32_t>(i) + partition_offset;
        }
    }

    // Reduce across threads using shared memory
    extern __shared__ char shared_mem[];
    float* s_sims = reinterpret_cast<float*>(shared_mem);
    uint32_t* s_indices = reinterpret_cast<uint32_t*>(s_sims + blockDim.x * k);

    // Store local results
    for (int i = 0; i < k; ++i) {
        s_sims[tid * k + i] = local_topk[i];
        s_indices[tid * k + i] = local_idx[i];
    }
    __syncthreads();

    // Parallel reduction for top-k merge
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            // Merge my top-k with partner's top-k
            float* my_sims = s_sims + tid * k;
            uint32_t* my_idx = s_indices + tid * k;
            float* other_sims = s_sims + (tid + stride) * k;
            uint32_t* other_idx = s_indices + (tid + stride) * k;

            // Simple k-way merge
            float merged_sims[K_MAX];
            uint32_t merged_idx[K_MAX];

            int i = 0, j = 0, m = 0;
            while (m < k && (i < k || j < k)) {
                if (i < k && (j >= k || my_sims[i] >= other_sims[j])) {
                    merged_sims[m] = my_sims[i];
                    merged_idx[m] = my_idx[i];
                    i++;
                } else {
                    merged_sims[m] = other_sims[j];
                    merged_idx[m] = other_idx[j];
                    j++;
                }
                m++;
            }

            for (int x = 0; x < k; ++x) {
                my_sims[x] = merged_sims[x];
                my_idx[x] = merged_idx[x];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        float* out_dist = topk_distances + query_idx * k;
        uint32_t* out_idx = topk_indices + query_idx * k;

        for (int i = 0; i < k; ++i) {
            out_dist[i] = -s_sims[i];  // Convert similarity to distance
            out_idx[i] = s_indices[i];
        }
    }
}

// =============================================================================
// GPUBatchSearch Implementation
// =============================================================================

GPUBatchSearch::GPUBatchSearch(size_t dim, int num_gpus)
    : dim_(dim), num_base_(0), num_gpus_(std::min(num_gpus, get_num_gpus())) {

    if (num_gpus_ <= 0) {
        throw std::runtime_error("No CUDA GPUs available");
    }

    gpu_contexts_.resize(num_gpus_);

    for (int i = 0; i < num_gpus_; ++i) {
        auto& ctx = gpu_contexts_[i];
        ctx.device_id = i;
        ctx.partition_start = 0;
        ctx.partition_size = 0;
        ctx.max_queries = 0;

        CUDA_CHECK(cudaSetDevice(i));
        CUBLAS_CHECK(cublasCreate(&ctx.cublas_handle));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle, ctx.stream));

        ctx.d_base = nullptr;
        ctx.d_queries = nullptr;
        ctx.d_similarities = nullptr;
        ctx.d_topk_distances = nullptr;
        ctx.d_topk_indices = nullptr;
    }

    std::cout << "GPUBatchSearch initialized with " << num_gpus_ << " GPUs" << std::endl;
}

GPUBatchSearch::~GPUBatchSearch() {
    free_resources();
    for (auto& ctx : gpu_contexts_) {
        cudaSetDevice(ctx.device_id);
        if (ctx.cublas_handle) cublasDestroy(ctx.cublas_handle);
        if (ctx.stream) cudaStreamDestroy(ctx.stream);
    }
}

void GPUBatchSearch::free_resources() {
    for (auto& ctx : gpu_contexts_) {
        cudaSetDevice(ctx.device_id);
        if (ctx.d_base) { cudaFree(ctx.d_base); ctx.d_base = nullptr; }
        if (ctx.d_queries) { cudaFree(ctx.d_queries); ctx.d_queries = nullptr; }
        if (ctx.d_similarities) { cudaFree(ctx.d_similarities); ctx.d_similarities = nullptr; }
        if (ctx.d_topk_distances) { cudaFree(ctx.d_topk_distances); ctx.d_topk_distances = nullptr; }
        if (ctx.d_topk_indices) { cudaFree(ctx.d_topk_indices); ctx.d_topk_indices = nullptr; }
    }
}

void GPUBatchSearch::set_base_vectors(const Float* h_vectors, size_t N) {
    num_base_ = N;

    // Partition base vectors across GPUs
    size_t partition_size = (N + num_gpus_ - 1) / num_gpus_;

    for (int i = 0; i < num_gpus_; ++i) {
        auto& ctx = gpu_contexts_[i];
        ctx.partition_start = i * partition_size;
        ctx.partition_size = std::min(partition_size, N - ctx.partition_start);

        if (ctx.partition_size == 0) continue;

        CUDA_CHECK(cudaSetDevice(i));

        // Free old base vectors if any
        if (ctx.d_base) {
            cudaFree(ctx.d_base);
        }

        // Allocate and copy partition
        size_t bytes = ctx.partition_size * dim_ * sizeof(Float);
        CUDA_CHECK(cudaMalloc(&ctx.d_base, bytes));
        CUDA_CHECK(cudaMemcpy(ctx.d_base,
                              h_vectors + ctx.partition_start * dim_,
                              bytes,
                              cudaMemcpyHostToDevice));

        std::cout << "  GPU " << i << ": " << ctx.partition_size << " vectors "
                  << "[" << ctx.partition_start << ", "
                  << ctx.partition_start + ctx.partition_size << ")" << std::endl;
    }
}

void GPUBatchSearch::allocate_query_buffers(size_t num_queries, size_t k) {
    for (int i = 0; i < num_gpus_; ++i) {
        auto& ctx = gpu_contexts_[i];
        if (ctx.partition_size == 0) continue;

        CUDA_CHECK(cudaSetDevice(i));

        // Reallocate if needed
        if (num_queries > ctx.max_queries) {
            if (ctx.d_queries) cudaFree(ctx.d_queries);
            if (ctx.d_similarities) cudaFree(ctx.d_similarities);
            if (ctx.d_topk_distances) cudaFree(ctx.d_topk_distances);
            if (ctx.d_topk_indices) cudaFree(ctx.d_topk_indices);

            ctx.max_queries = num_queries * 2;  // Over-allocate for future reuse

            CUDA_CHECK(cudaMalloc(&ctx.d_queries,
                                  ctx.max_queries * dim_ * sizeof(Float)));
            CUDA_CHECK(cudaMalloc(&ctx.d_similarities,
                                  ctx.max_queries * ctx.partition_size * sizeof(Float)));
            CUDA_CHECK(cudaMalloc(&ctx.d_topk_distances,
                                  ctx.max_queries * k * sizeof(Float)));
            CUDA_CHECK(cudaMalloc(&ctx.d_topk_indices,
                                  ctx.max_queries * k * sizeof(uint32_t)));
        }
    }
}

void GPUBatchSearch::search_batch(const Float* h_queries, size_t num_queries, size_t k,
                                   uint32_t* h_indices, Float* h_distances) {
    if (num_base_ == 0) {
        throw std::runtime_error("Base vectors not set. Call set_base_vectors first.");
    }

    // K_MAX=64 is hardcoded in search_topk_simple_kernel template instantiation
    constexpr size_t K_MAX = 64;
    if (k > K_MAX) {
        throw std::runtime_error("GPU search: k=" + std::to_string(k) +
            " exceeds maximum supported value of " + std::to_string(K_MAX));
    }

    allocate_query_buffers(num_queries, k);

    // Per-GPU results
    std::vector<std::vector<uint32_t>> gpu_indices(num_gpus_);
    std::vector<std::vector<float>> gpu_distances(num_gpus_);

    for (int i = 0; i < num_gpus_; ++i) {
        auto& ctx = gpu_contexts_[i];
        if (ctx.partition_size == 0) continue;

        CUDA_CHECK(cudaSetDevice(i));

        // Copy queries to GPU
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_queries,
                                   h_queries,
                                   num_queries * dim_ * sizeof(Float),
                                   cudaMemcpyHostToDevice,
                                   ctx.stream));

        // Compute similarities using cuBLAS SGEMM
        // C = alpha * A * B^T + beta * C
        // similarities[num_queries x partition_size] =
        //     queries[num_queries x dim] @ base[partition_size x dim]^T
        float alpha = 1.0f;
        float beta = 0.0f;

        CUBLAS_CHECK(cublasSgemm(
            ctx.cublas_handle,
            CUBLAS_OP_T,      // B^T
            CUBLAS_OP_N,      // A
            ctx.partition_size,  // N (columns of result)
            num_queries,         // M (rows of result)
            dim_,                // K (inner dimension)
            &alpha,
            ctx.d_base,          // B: [partition_size x dim]
            dim_,                // ldb
            ctx.d_queries,       // A: [num_queries x dim]
            dim_,                // lda
            &beta,
            ctx.d_similarities,  // C: [num_queries x partition_size]
            ctx.partition_size   // ldc
        ));

        // Extract top-k
        int block_size = 256;
        size_t shared_mem = block_size * k * (sizeof(float) + sizeof(uint32_t));

        search_topk_simple_kernel<64><<<num_queries, block_size, shared_mem, ctx.stream>>>(
            ctx.d_similarities,
            ctx.d_topk_indices,
            ctx.d_topk_distances,
            ctx.partition_size,
            k,
            ctx.partition_start  // Offset for global indices
        );

        // Copy results back
        gpu_indices[i].resize(num_queries * k);
        gpu_distances[i].resize(num_queries * k);

        CUDA_CHECK(cudaMemcpyAsync(gpu_indices[i].data(),
                                   ctx.d_topk_indices,
                                   num_queries * k * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_distances[i].data(),
                                   ctx.d_topk_distances,
                                   num_queries * k * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   ctx.stream));
    }

    // Sync all GPUs
    for (int i = 0; i < num_gpus_; ++i) {
        CUDA_CHECK(cudaSetDevice(gpu_contexts_[i].device_id));
        CUDA_CHECK(cudaStreamSynchronize(gpu_contexts_[i].stream));
    }

    // Merge results from all GPUs
    merge_topk_results(num_queries, k, h_indices, h_distances);

    // Actually merge using the per-GPU results
    #pragma omp parallel for
    for (size_t q = 0; q < num_queries; ++q) {
        // Collect all candidates from all GPUs
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(num_gpus_ * k);

        for (int g = 0; g < num_gpus_; ++g) {
            if (gpu_contexts_[g].partition_size == 0) continue;

            for (size_t j = 0; j < k; ++j) {
                candidates.emplace_back(
                    gpu_distances[g][q * k + j],
                    gpu_indices[g][q * k + j]
                );
            }
        }

        // Sort by distance (ascending)
        std::partial_sort(candidates.begin(),
                         candidates.begin() + k,
                         candidates.end());

        // Output top-k
        for (size_t j = 0; j < k; ++j) {
            h_indices[q * k + j] = candidates[j].second;
            h_distances[q * k + j] = candidates[j].first;
        }
    }
}

void GPUBatchSearch::merge_topk_results(size_t num_queries, size_t k,
                                         uint32_t* h_indices, Float* h_distances) {
    // This is now handled inline in search_batch
    // Keep method for API compatibility
}

void GPUBatchSearch::compute_ground_truth(const Float* h_queries, size_t num_queries, size_t k,
                                           uint32_t* h_indices) {
    std::vector<float> distances(num_queries * k);
    search_batch(h_queries, num_queries, k, h_indices, distances.data());
}

void GPUBatchSearch::rerank_batch(const Float* h_queries, const uint32_t* h_candidate_ids,
                                   size_t num_queries, size_t num_candidates, size_t k,
                                   uint32_t* h_indices, Float* h_distances) {
    // For reranking, we compute similarities only for the candidate set
    // This is more efficient than full brute force

    // For now, use CPU reranking (fast enough for small candidate sets)
    #pragma omp parallel for
    for (size_t q = 0; q < num_queries; ++q) {
        const Float* query = h_queries + q * dim_;
        const uint32_t* candidates = h_candidate_ids + q * num_candidates;

        std::vector<std::pair<float, uint32_t>> scored_candidates(num_candidates);

        for (size_t c = 0; c < num_candidates; ++c) {
            uint32_t id = candidates[c];
            // Note: We need base vectors on host for this
            // For full GPU rerank, we'd gather vectors on GPU first
            scored_candidates[c] = {0.0f, id};  // Placeholder
        }

        // Sort and output
        std::partial_sort(scored_candidates.begin(),
                         scored_candidates.begin() + k,
                         scored_candidates.end());

        for (size_t j = 0; j < k; ++j) {
            h_indices[q * k + j] = scored_candidates[j].second;
            h_distances[q * k + j] = scored_candidates[j].first;
        }
    }
}

// =============================================================================
// GPUGroundTruth Implementation
// =============================================================================

GPUGroundTruth::GPUGroundTruth(size_t dim, int num_gpus)
    : dim_(dim), num_gpus_(std::min(num_gpus, get_num_gpus())) {
    // Minimal initialization - resources allocated on demand
}

GPUGroundTruth::~GPUGroundTruth() {
    for (auto& ctx : contexts_) {
        cudaSetDevice(ctx.device_id);
        if (ctx.d_base) cudaFree(ctx.d_base);
        if (ctx.d_queries) cudaFree(ctx.d_queries);
        if (ctx.d_similarities) cudaFree(ctx.d_similarities);
        if (ctx.d_topk_indices) cudaFree(ctx.d_topk_indices);
        if (ctx.d_topk_distances) cudaFree(ctx.d_topk_distances);
        if (ctx.cublas_handle) cublasDestroy(ctx.cublas_handle);
        if (ctx.stream) cudaStreamDestroy(ctx.stream);
    }
}

void GPUGroundTruth::compute(const Float* h_base, size_t N,
                              const Float* h_queries, size_t num_queries, size_t k,
                              uint32_t* h_gt_indices) {
    // Use GPUBatchSearch for ground truth computation
    GPUBatchSearch searcher(dim_, num_gpus_);
    searcher.set_base_vectors(h_base, N);

    std::vector<float> distances(num_queries * k);
    searcher.search_batch(h_queries, num_queries, k, h_gt_indices, distances.data());
}

}  // namespace cuda
}  // namespace cphnsw
