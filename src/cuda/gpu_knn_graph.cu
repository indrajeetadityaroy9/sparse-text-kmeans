#include "../../include/cphnsw/cuda/gpu_knn_graph.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cphnsw {
namespace cuda {

// =============================================================================
// CUDA Kernels for k-NN Graph Building
// =============================================================================

/**
 * Initialize running top-k with worst values (lowest similarity = highest distance).
 * Sets distances to -infinity (worst similarity) and indices to UINT32_MAX.
 */
__global__ void init_topk_kernel(Float* running_dist, uint32_t* running_idx,
                                  size_t num_queries, size_t k) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_queries * k;

    if (idx < total) {
        running_dist[idx] = -1e30f;  // Worst similarity (most negative)
        running_idx[idx] = UINT32_MAX;
    }
}

/**
 * Fused top-k merge kernel: extract top-k from tile similarities and merge with running top-k.
 *
 * For each query:
 * 1. Find top-k from current tile (tile_size candidates)
 * 2. Merge with existing running top-k (k candidates)
 * 3. Keep best k
 *
 * This runs immediately after SGEMM, before moving to next tile.
 * Tile similarities are discarded after this kernel.
 *
 * Uses warp-level primitives for efficiency.
 */
__global__ void fused_topk_merge_kernel(
    const Float* tile_similarities,   // [num_queries x tile_size]
    const size_t tile_size,
    const uint32_t tile_base_idx,     // Global index offset for this tile
    Float* running_topk_dist,         // [num_queries x k] (in/out)
    uint32_t* running_topk_idx,       // [num_queries x k] (in/out)
    size_t num_queries,
    size_t k) {

    // One block per query
    size_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    const Float* sims = tile_similarities + query_idx * tile_size;
    Float* my_dist = running_topk_dist + query_idx * k;
    uint32_t* my_idx = running_topk_idx + query_idx * k;

    // Shared memory for thread-local top-k
    extern __shared__ char shared_mem[];
    Float* s_dist = reinterpret_cast<Float*>(shared_mem);
    uint32_t* s_idx = reinterpret_cast<uint32_t*>(s_dist + k * blockDim.x);

    Float* thread_dist = s_dist + threadIdx.x * k;
    uint32_t* thread_idx = s_idx + threadIdx.x * k;

    // Initialize thread's top-k with worst values
    for (size_t i = 0; i < k; ++i) {
        thread_dist[i] = -1e30f;
        thread_idx[i] = UINT32_MAX;
    }

    // Each thread scans a portion of the tile and maintains local top-k
    for (size_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        Float sim = sims[i];
        uint32_t global_idx = tile_base_idx + static_cast<uint32_t>(i);

        // Insert into thread's top-k if better than worst
        if (sim > thread_dist[k - 1]) {
            // Find insertion point (sorted descending by similarity)
            size_t pos = k - 1;
            while (pos > 0 && sim > thread_dist[pos - 1]) {
                thread_dist[pos] = thread_dist[pos - 1];
                thread_idx[pos] = thread_idx[pos - 1];
                --pos;
            }
            thread_dist[pos] = sim;
            thread_idx[pos] = global_idx;
        }
    }
    __syncthreads();

    // Merge all threads' results (thread 0 does final merge)
    if (threadIdx.x == 0) {
        // Start with running top-k
        Float merged_dist[128];  // Assume k <= 128
        uint32_t merged_idx[128];

        for (size_t i = 0; i < k; ++i) {
            merged_dist[i] = my_dist[i];
            merged_idx[i] = my_idx[i];
        }

        // Merge each thread's results
        for (int t = 0; t < blockDim.x; ++t) {
            Float* t_dist = s_dist + t * k;
            uint32_t* t_idx = s_idx + t * k;

            for (size_t i = 0; i < k; ++i) {
                Float sim = t_dist[i];
                uint32_t idx = t_idx[i];

                if (sim > merged_dist[k - 1] && idx != UINT32_MAX) {
                    size_t pos = k - 1;
                    while (pos > 0 && sim > merged_dist[pos - 1]) {
                        merged_dist[pos] = merged_dist[pos - 1];
                        merged_idx[pos] = merged_idx[pos - 1];
                        --pos;
                    }
                    merged_dist[pos] = sim;
                    merged_idx[pos] = idx;
                }
            }
        }

        // Write back merged top-k
        for (size_t i = 0; i < k; ++i) {
            my_dist[i] = merged_dist[i];
            my_idx[i] = merged_idx[i];
        }
    }
}

/**
 * Convert similarities to distances and copy to output.
 * Distance = 1 - similarity (positive metric for pruning algorithm)
 * For normalized vectors: this is related to squared Euclidean distance
 */
__global__ void convert_to_distances_kernel(
    const Float* similarities,
    Float* distances,
    size_t total) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        distances[idx] = 1.0f - similarities[idx];  // Positive for pruning
    }
}

/**
 * Remove self-loops from k-NN results.
 * If neighbors[i][j] == i, shift remaining neighbors and set last to UINT32_MAX.
 */
__global__ void remove_self_loops_kernel(
    uint32_t* neighbors,
    Float* distances,  // Can be nullptr
    size_t N,
    size_t k,
    size_t query_offset) {  // Global query index offset

    size_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= N) return;

    uint32_t global_query_id = static_cast<uint32_t>(query_offset + query_idx);
    uint32_t* my_neighbors = neighbors + query_idx * k;
    Float* my_distances = distances ? (distances + query_idx * k) : nullptr;

    // Find and remove self-loop
    for (size_t i = 0; i < k; ++i) {
        if (my_neighbors[i] == global_query_id) {
            // Shift remaining elements
            for (size_t j = i; j < k - 1; ++j) {
                my_neighbors[j] = my_neighbors[j + 1];
                if (my_distances) my_distances[j] = my_distances[j + 1];
            }
            my_neighbors[k - 1] = UINT32_MAX;
            if (my_distances) my_distances[k - 1] = 1e30f;
            break;  // At most one self-loop
        }
    }
}

// =============================================================================
// GPUKNNGraphBuilder Implementation
// =============================================================================

GPUKNNGraphBuilder::GPUKNNGraphBuilder(size_t dim, size_t k)
    : dim_(dim), k_(k),
      d_similarities_(nullptr), d_running_topk_dist_(nullptr),
      d_running_topk_idx_(nullptr), max_query_batch_(0), max_base_tile_(0) {

    CUDA_CHECK(cudaStreamCreate(&stream_));

    cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    cublasSetStream(static_cast<cublasHandle_t>(cublas_handle_), stream_);
}

GPUKNNGraphBuilder::~GPUKNNGraphBuilder() {
    free_buffers();
    cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
    cudaStreamDestroy(stream_);
}

void GPUKNNGraphBuilder::free_buffers() {
    if (d_similarities_) { cudaFree(d_similarities_); d_similarities_ = nullptr; }
    if (d_running_topk_dist_) { cudaFree(d_running_topk_dist_); d_running_topk_dist_ = nullptr; }
    if (d_running_topk_idx_) { cudaFree(d_running_topk_idx_); d_running_topk_idx_ = nullptr; }
}

void GPUKNNGraphBuilder::allocate_buffers(size_t query_batch, size_t base_tile) {
    if (query_batch <= max_query_batch_ && base_tile <= max_base_tile_) {
        return;  // Already have enough space
    }

    free_buffers();

    // Similarity matrix: query_batch × base_tile
    CUDA_CHECK(cudaMalloc(&d_similarities_, query_batch * base_tile * sizeof(Float)));

    // Running top-k: query_batch × k
    CUDA_CHECK(cudaMalloc(&d_running_topk_dist_, query_batch * k_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_running_topk_idx_, query_batch * k_ * sizeof(uint32_t)));

    max_query_batch_ = query_batch;
    max_base_tile_ = base_tile;
}

void GPUKNNGraphBuilder::get_recommended_sizes(size_t N, size_t dim, size_t k,
                                                size_t available_memory_gb,
                                                size_t& query_batch, size_t& base_tile) {
    // Memory budget in bytes
    size_t memory_budget = available_memory_gb * 1024ULL * 1024 * 1024;

    // Reserve 20% for other allocations
    memory_budget = static_cast<size_t>(memory_budget * 0.8);

    // Base tile: we want at least 100K vectors per tile for efficiency
    // Similarity matrix: query_batch × base_tile × 4 bytes
    // Running top-k: query_batch × k × 8 bytes (negligible)

    // Start with reasonable defaults
    query_batch = std::min(N, static_cast<size_t>(10000));  // 10K queries per batch

    // Base tile size from remaining memory
    // sim_matrix_size = query_batch × base_tile × 4
    size_t max_sim_matrix = memory_budget;
    base_tile = max_sim_matrix / (query_batch * sizeof(Float));

    // Clamp to reasonable values
    base_tile = std::max(base_tile, static_cast<size_t>(10000));   // At least 10K
    base_tile = std::min(base_tile, N);  // At most N

    // If base_tile exceeds memory, reduce query_batch
    while (query_batch * base_tile * sizeof(Float) > memory_budget && query_batch > 1000) {
        query_batch /= 2;
    }
}

void GPUKNNGraphBuilder::init_running_topk(size_t num_queries) {
    int threads = 256;
    int blocks = (num_queries * k_ + threads - 1) / threads;

    init_topk_kernel<<<blocks, threads, 0, stream_>>>(
        d_running_topk_dist_, d_running_topk_idx_, num_queries, k_);
}

void GPUKNNGraphBuilder::process_tile(const Float* d_queries, size_t num_queries,
                                       const Float* d_base_tile, size_t tile_size,
                                       size_t tile_base_idx) {
    // SGEMM: similarities = queries @ base_tile^T
    // For row-major data in cuBLAS (column-major):
    // - Row-major A[m x k] is seen as column-major A^T[k x m]
    // - We want: C[i,j] = dot(queries[i], base[j])
    // - C_rowmajor = queries @ base^T
    // - In cuBLAS: C_colmajor = base @ queries^T (transposed result)
    //
    // So: C_cm = base_cm^T @ queries_cm = SGEMM(T, N, base, queries)

    float alpha = 1.0f, beta = 0.0f;

    // For row-major input:
    // queries: [num_queries x dim] row-major = [dim x num_queries] column-major
    // base:    [tile_size x dim] row-major = [dim x tile_size] column-major
    // output:  [num_queries x tile_size] row-major = [tile_size x num_queries] column-major
    //
    // C = alpha * op(A) * op(B) + beta * C
    // We want C_cm[tile_size x num_queries] = base_cm^T[tile_size x dim] @ queries_cm[dim x num_queries]
    // So: op(A) = A^T where A = base_cm (dim x tile_size), giving (tile_size x dim)
    //     op(B) = B where B = queries_cm (dim x num_queries)
    //     m = tile_size, n = num_queries, k = dim

    cublasStatus_t status = cublasSgemm(
        static_cast<cublasHandle_t>(cublas_handle_),
        CUBLAS_OP_T,        // A^T: base transposed
        CUBLAS_OP_N,        // B: queries as-is
        static_cast<int>(tile_size),      // m: rows of C and op(A)
        static_cast<int>(num_queries),    // n: cols of C and op(B)
        static_cast<int>(dim_),           // k: shared dimension
        &alpha,
        d_base_tile,        // A: base vectors [dim x tile_size] in column-major
        static_cast<int>(dim_),           // lda
        d_queries,          // B: query vectors [dim x num_queries] in column-major
        static_cast<int>(dim_),           // ldb
        &beta,
        d_similarities_,    // C: similarities [tile_size x num_queries] in column-major
        static_cast<int>(tile_size));     // ldc

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS SGEMM failed with status " << status << std::endl;
    }

    // Fused top-k merge
    // Shared memory per thread = k * (sizeof(Float) + sizeof(uint32_t)) = k * 8 bytes
    // Max shared memory (default) = 48KB = 49152 bytes
    // Max threads = 49152 / (k * 8)
    size_t bytes_per_thread = k_ * (sizeof(Float) + sizeof(uint32_t));
    int max_threads_for_shmem = static_cast<int>(48 * 1024 / bytes_per_thread);
    int threads = std::min({256, static_cast<int>(tile_size), max_threads_for_shmem});
    size_t shared_mem = k_ * threads * (sizeof(Float) + sizeof(uint32_t));

    fused_topk_merge_kernel<<<num_queries, threads, shared_mem, stream_>>>(
        d_similarities_, tile_size, static_cast<uint32_t>(tile_base_idx),
        d_running_topk_dist_, d_running_topk_idx_,
        num_queries, k_);

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
}

void GPUKNNGraphBuilder::build(const Float* h_vectors, size_t N,
                                uint32_t* h_neighbors, Float* h_distances,
                                size_t query_batch_size, size_t base_tile_size) {
    // Auto-select batch sizes if not specified
    if (query_batch_size == 0 || base_tile_size == 0) {
        size_t available_gb = 40;  // Assume 40GB available (H100 has 80GB)
        get_recommended_sizes(N, dim_, k_, available_gb, query_batch_size, base_tile_size);
    }

    std::cout << "  GPU k-NN: N=" << N << ", k=" << k_
              << ", query_batch=" << query_batch_size
              << ", base_tile=" << base_tile_size << std::endl;

    // Allocate buffers
    allocate_buffers(query_batch_size, base_tile_size);

    // Upload all vectors to GPU
    Float* d_vectors;
    CUDA_CHECK(cudaMalloc(&d_vectors, N * dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, N * dim_ * sizeof(Float),
                          cudaMemcpyHostToDevice));

    // Temporary device buffers for output
    uint32_t* d_batch_neighbors;
    Float* d_batch_distances;
    CUDA_CHECK(cudaMalloc(&d_batch_neighbors, query_batch_size * k_ * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_batch_distances, query_batch_size * k_ * sizeof(Float)));

    // Process in query batches
    for (size_t q_start = 0; q_start < N; q_start += query_batch_size) {
        size_t q_end = std::min(q_start + query_batch_size, N);
        size_t num_queries = q_end - q_start;

        const Float* d_queries = d_vectors + q_start * dim_;

        // Initialize running top-k for this batch
        init_running_topk(num_queries);

        // Tile through all base vectors
        for (size_t b_start = 0; b_start < N; b_start += base_tile_size) {
            size_t b_end = std::min(b_start + base_tile_size, N);
            size_t tile_size = b_end - b_start;

            const Float* d_base_tile = d_vectors + b_start * dim_;

            process_tile(d_queries, num_queries, d_base_tile, tile_size, b_start);
        }

        // Remove self-loops
        int threads = 256;
        int blocks = (num_queries + threads - 1) / threads;
        remove_self_loops_kernel<<<blocks, threads, 0, stream_>>>(
            d_running_topk_idx_, d_running_topk_dist_, num_queries, k_, q_start);

        // Copy results to host
        CUDA_CHECK(cudaMemcpyAsync(h_neighbors + q_start * k_,
                                    d_running_topk_idx_,
                                    num_queries * k_ * sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost, stream_));

        if (h_distances) {
            // Convert similarities to distances
            int conv_threads = 256;
            int conv_blocks = (num_queries * k_ + conv_threads - 1) / conv_threads;
            convert_to_distances_kernel<<<conv_blocks, conv_threads, 0, stream_>>>(
                d_running_topk_dist_, d_batch_distances, num_queries * k_);

            CUDA_CHECK(cudaMemcpyAsync(h_distances + q_start * k_,
                                        d_batch_distances,
                                        num_queries * k_ * sizeof(Float),
                                        cudaMemcpyDeviceToHost, stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Progress report
        if ((q_start + query_batch_size) % (N / 10 + 1) < query_batch_size) {
            std::cout << "  Progress: " << std::min(q_end, N) << "/" << N
                      << " (" << (100 * std::min(q_end, N) / N) << "%)" << std::endl;
        }
    }

    cudaFree(d_vectors);
    cudaFree(d_batch_neighbors);
    cudaFree(d_batch_distances);
}

void GPUKNNGraphBuilder::build_device(const Float* d_vectors, size_t N,
                                       uint32_t* h_neighbors, Float* h_distances,
                                       size_t query_batch_size, size_t base_tile_size) {
    // Similar to build() but vectors are already on GPU
    if (query_batch_size == 0 || base_tile_size == 0) {
        size_t available_gb = 40;
        get_recommended_sizes(N, dim_, k_, available_gb, query_batch_size, base_tile_size);
    }

    allocate_buffers(query_batch_size, base_tile_size);

    Float* d_batch_distances;
    CUDA_CHECK(cudaMalloc(&d_batch_distances, query_batch_size * k_ * sizeof(Float)));

    for (size_t q_start = 0; q_start < N; q_start += query_batch_size) {
        size_t q_end = std::min(q_start + query_batch_size, N);
        size_t num_queries = q_end - q_start;

        const Float* d_queries = d_vectors + q_start * dim_;

        init_running_topk(num_queries);

        for (size_t b_start = 0; b_start < N; b_start += base_tile_size) {
            size_t b_end = std::min(b_start + base_tile_size, N);
            size_t tile_size = b_end - b_start;

            const Float* d_base_tile = d_vectors + b_start * dim_;
            process_tile(d_queries, num_queries, d_base_tile, tile_size, b_start);
        }

        int threads = 256;
        int blocks = (num_queries + threads - 1) / threads;
        remove_self_loops_kernel<<<blocks, threads, 0, stream_>>>(
            d_running_topk_idx_, d_running_topk_dist_, num_queries, k_, q_start);

        CUDA_CHECK(cudaMemcpyAsync(h_neighbors + q_start * k_,
                                    d_running_topk_idx_,
                                    num_queries * k_ * sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost, stream_));

        if (h_distances) {
            int conv_threads = 256;
            int conv_blocks = (num_queries * k_ + conv_threads - 1) / conv_threads;
            convert_to_distances_kernel<<<conv_blocks, conv_threads, 0, stream_>>>(
                d_running_topk_dist_, d_batch_distances, num_queries * k_);

            CUDA_CHECK(cudaMemcpyAsync(h_distances + q_start * k_,
                                        d_batch_distances,
                                        num_queries * k_ * sizeof(Float),
                                        cudaMemcpyDeviceToHost, stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaFree(d_batch_distances);
}

// =============================================================================
// KNNEdgeList Implementation
// =============================================================================

KNNEdgeList KNNEdgeList::from_matrix(const uint32_t* neighbors, const float* distances,
                                      size_t N, size_t k) {
    KNNEdgeList result;
    result.num_nodes = N;
    result.k = k;

    // Reserve space (may have fewer edges due to UINT32_MAX entries)
    result.sources.reserve(N * k);
    result.targets.reserve(N * k);
    if (distances) result.distances.reserve(N * k);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < k; ++j) {
            uint32_t neighbor = neighbors[i * k + j];
            if (neighbor != UINT32_MAX && neighbor != static_cast<uint32_t>(i)) {
                result.sources.push_back(static_cast<uint32_t>(i));
                result.targets.push_back(neighbor);
                if (distances) {
                    result.distances.push_back(distances[i * k + j]);
                }
            }
        }
    }

    return result;
}

std::pair<const uint32_t*, size_t> KNNEdgeList::get_neighbors(uint32_t node_id) const {
    // Linear scan - inefficient but simple
    // For production, use CSR format
    size_t start = 0;
    size_t count = 0;

    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i] == node_id) {
            if (count == 0) start = i;
            ++count;
        } else if (count > 0) {
            break;  // Sources are sorted, so we're done
        }
    }

    return {targets.data() + start, count};
}

}  // namespace cuda
}  // namespace cphnsw
