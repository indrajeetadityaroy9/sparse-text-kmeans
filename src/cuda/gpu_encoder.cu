#include "../../include/cphnsw/cuda/gpu_encoder.cuh"
#include <cublas_v2.h>
#include <random>
#include <cmath>
#include <algorithm>

namespace cphnsw {
namespace cuda {

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * Fast Hadamard Transform kernel (in-place, power-of-2 size)
 *
 * Uses shared memory for efficient butterfly operations.
 * One block processes one vector.
 */
__global__ void fht_kernel(Float* data, size_t dim, size_t num_vecs) {
    extern __shared__ Float shared[];

    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    Float* vec = data + vec_idx * dim;

    // Load to shared memory (coalesced)
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        shared[i] = vec[i];
    }
    __syncthreads();

    // Butterfly iterations
    for (size_t len = 1; len < dim; len *= 2) {
        for (size_t i = threadIdx.x; i < dim / 2; i += blockDim.x) {
            size_t block = i / len;
            size_t offset = i % len;
            size_t j = block * 2 * len + offset;
            size_t k = j + len;

            Float a = shared[j];
            Float b = shared[k];
            shared[j] = a + b;
            shared[k] = a - b;
        }
        __syncthreads();
    }

    // Write back (coalesced)
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] = shared[i];
    }
}

/**
 * Apply diagonal matrix (element-wise multiply by ±1)
 */
__global__ void apply_diagonal_kernel(Float* data, const int8_t* signs,
                                       size_t dim, size_t num_vecs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dim * num_vecs;

    if (idx < total) {
        size_t d = idx % dim;
        data[idx] *= static_cast<Float>(signs[d]);
    }
}

/**
 * Combined rotation chain: D1 -> FHT -> D2 -> FHT -> D3 -> FHT
 *
 * Each rotation r uses signs at offset r * 3 * padded_dim
 */
__global__ void rotation_chain_kernel(Float* output, const Float* input,
                                       const int8_t* all_signs,
                                       size_t dim, size_t padded_dim,
                                       size_t num_vecs, size_t rotation_idx) {
    extern __shared__ Float shared[];

    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const Float* in_vec = input + vec_idx * dim;
    Float* out_vec = output + vec_idx * padded_dim;
    const int8_t* signs = all_signs + rotation_idx * 3 * padded_dim;

    // Initialize shared memory with zero-padded input
    for (size_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        shared[i] = (i < dim) ? in_vec[i] : 0.0f;
    }
    __syncthreads();

    // Three layers: D1, H, D2, H, D3, H
    for (int layer = 0; layer < 3; ++layer) {
        const int8_t* layer_signs = signs + layer * padded_dim;

        // Apply diagonal
        for (size_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
            shared[i] *= static_cast<Float>(layer_signs[i]);
        }
        __syncthreads();

        // FHT butterfly
        for (size_t len = 1; len < padded_dim; len *= 2) {
            for (size_t i = threadIdx.x; i < padded_dim / 2; i += blockDim.x) {
                size_t block = i / len;
                size_t offset = i % len;
                size_t j = block * 2 * len + offset;
                size_t k = j + len;

                Float a = shared[j];
                Float b = shared[k];
                shared[j] = a + b;
                shared[k] = a - b;
            }
            __syncthreads();
        }
    }

    // Write output
    for (size_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        out_vec[i] = shared[i];
    }
}

/**
 * Argmax kernel: find index and sign of maximum absolute value
 *
 * Uses warp-level reduction for efficiency.
 * Output: code = (index << 1) | sign_bit
 */
template<typename ComponentT>
__global__ void argmax_encode_kernel(const Float* rotated, ComponentT* codes,
                                      Float* magnitudes,  // Can be nullptr
                                      size_t padded_dim, size_t num_vecs,
                                      size_t K, size_t rotation_stride) {
    // One block per (vector, rotation) pair
    size_t vec_idx = blockIdx.x / K;
    size_t rot_idx = blockIdx.x % K;

    if (vec_idx >= num_vecs) return;

    const Float* vec = rotated + vec_idx * rotation_stride + rot_idx * padded_dim;

    // Thread-local max
    Float thread_max_abs = 0.0f;
    size_t thread_max_idx = 0;
    Float thread_max_val = 0.0f;

    for (size_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        Float val = vec[i];
        Float abs_val = fabsf(val);
        if (abs_val > thread_max_abs) {
            thread_max_abs = abs_val;
            thread_max_idx = i;
            thread_max_val = val;
        }
    }

    // Warp reduction
    __shared__ Float s_max_abs[32];
    __shared__ size_t s_max_idx[32];
    __shared__ Float s_max_val[32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_abs = __shfl_down_sync(0xffffffff, thread_max_abs, offset);
        size_t other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        Float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);

        if (other_abs > thread_max_abs) {
            thread_max_abs = other_abs;
            thread_max_idx = other_idx;
            thread_max_val = other_val;
        }
    }

    if (lane == 0) {
        s_max_abs[warp] = thread_max_abs;
        s_max_idx[warp] = thread_max_idx;
        s_max_val[warp] = thread_max_val;
    }
    __syncthreads();

    // Final reduction in first warp
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        if (threadIdx.x < num_warps) {
            thread_max_abs = s_max_abs[threadIdx.x];
            thread_max_idx = s_max_idx[threadIdx.x];
            thread_max_val = s_max_val[threadIdx.x];
        } else {
            thread_max_abs = 0.0f;
            thread_max_idx = 0;
            thread_max_val = 0.0f;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            Float other_abs = __shfl_down_sync(0xffffffff, thread_max_abs, offset);
            size_t other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            Float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);

            if (other_abs > thread_max_abs) {
                thread_max_abs = other_abs;
                thread_max_idx = other_idx;
                thread_max_val = other_val;
            }
        }

        if (threadIdx.x == 0) {
            // Encode: (index << 1) | sign_bit
            bool is_negative = (thread_max_val < 0);
            ComponentT code = static_cast<ComponentT>((thread_max_idx << 1) | (is_negative ? 1 : 0));
            codes[vec_idx * K + rot_idx] = code;

            if (magnitudes != nullptr) {
                magnitudes[vec_idx * K + rot_idx] = thread_max_abs;
            }
        }
    }
}

/**
 * Top-k selection kernel using bitonic sort
 *
 * For each query, finds top-k from similarity scores.
 */
__global__ void topk_kernel(const Float* similarities, uint32_t* indices,
                            Float* distances, size_t num_base, size_t num_queries,
                            size_t k) {
    size_t query_idx = blockIdx.x;
    if (query_idx >= num_queries) return;

    const Float* sims = similarities + query_idx * num_base;

    // Thread-local top-k using insertion sort (k is small)
    extern __shared__ char shared_mem[];
    Float* s_vals = reinterpret_cast<Float*>(shared_mem);
    uint32_t* s_idx = reinterpret_cast<uint32_t*>(s_vals + k * blockDim.x);

    Float* my_vals = s_vals + threadIdx.x * k;
    uint32_t* my_idx = s_idx + threadIdx.x * k;

    // Initialize with worst values
    for (size_t i = 0; i < k; ++i) {
        my_vals[i] = -1e30f;  // Negative similarity = bad
        my_idx[i] = 0;
    }

    // Each thread processes a subset of base vectors
    for (size_t i = threadIdx.x; i < num_base; i += blockDim.x) {
        Float sim = sims[i];

        // Insert if better than worst in our top-k
        if (sim > my_vals[k-1]) {
            // Find insertion point
            size_t pos = k - 1;
            while (pos > 0 && sim > my_vals[pos-1]) {
                my_vals[pos] = my_vals[pos-1];
                my_idx[pos] = my_idx[pos-1];
                --pos;
            }
            my_vals[pos] = sim;
            my_idx[pos] = static_cast<uint32_t>(i);
        }
    }
    __syncthreads();

    // Merge thread results (simplified: only thread 0 does final merge)
    if (threadIdx.x == 0) {
        Float final_vals[128];  // Assume k <= 128
        uint32_t final_idx[128];

        for (size_t i = 0; i < k; ++i) {
            final_vals[i] = my_vals[i];
            final_idx[i] = my_idx[i];
        }

        // Merge other threads' results
        for (int t = 1; t < blockDim.x; ++t) {
            Float* t_vals = s_vals + t * k;
            uint32_t* t_idx = s_idx + t * k;

            for (size_t i = 0; i < k; ++i) {
                Float sim = t_vals[i];
                if (sim > final_vals[k-1]) {
                    size_t pos = k - 1;
                    while (pos > 0 && sim > final_vals[pos-1]) {
                        final_vals[pos] = final_vals[pos-1];
                        final_idx[pos] = final_idx[pos-1];
                        --pos;
                    }
                    final_vals[pos] = sim;
                    final_idx[pos] = t_idx[i];
                }
            }
        }

        // Write output (convert similarity to distance: -sim)
        uint32_t* out_idx = indices + query_idx * k;
        Float* out_dist = distances + query_idx * k;
        for (size_t i = 0; i < k; ++i) {
            out_idx[i] = final_idx[i];
            out_dist[i] = -final_vals[i];
        }
    }
}

/**
 * Gather kernel: fetch specific vectors by index
 */
__global__ void gather_vectors_kernel(const Float* base, const uint32_t* indices,
                                       Float* output, size_t dim,
                                       size_t num_queries, size_t num_candidates) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_queries * num_candidates * dim;

    if (idx < total) {
        size_t d = idx % dim;
        size_t cand = (idx / dim) % num_candidates;
        size_t q = idx / (dim * num_candidates);

        uint32_t base_idx = indices[q * num_candidates + cand];
        output[idx] = base[base_idx * dim + d];
    }
}

/**
 * Batch dot product for re-ranking
 */
__global__ void batch_dot_product_kernel(const Float* queries, const Float* candidates,
                                          Float* similarities, size_t dim,
                                          size_t num_queries, size_t num_candidates) {
    size_t pair_idx = blockIdx.x;
    size_t q = pair_idx / num_candidates;
    size_t c = pair_idx % num_candidates;

    if (q >= num_queries) return;

    const Float* query = queries + q * dim;
    const Float* cand = candidates + pair_idx * dim;

    // Parallel reduction for dot product
    __shared__ Float shared[256];

    Float sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += query[i] * cand[i];
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        similarities[pair_idx] = shared[0];
    }
}

// =============================================================================
// GPUEncoder Implementation
// =============================================================================

template<typename ComponentT, size_t K>
GPUEncoder<ComponentT, K>::GPUEncoder(size_t dim, uint64_t seed)
    : dim_(dim), d_signs_(nullptr), d_vectors_padded_(nullptr),
      d_rotated_(nullptr), max_batch_size_(0) {

    // Compute padded dimension (next power of 2)
    padded_dim_ = 1;
    while (padded_dim_ < dim_) {
        padded_dim_ *= 2;
    }

    CUDA_CHECK(cudaStreamCreate(&stream_));
    generate_signs(seed);
}

template<typename ComponentT, size_t K>
GPUEncoder<ComponentT, K>::~GPUEncoder() {
    if (d_signs_) cudaFree(d_signs_);
    if (d_vectors_padded_) cudaFree(d_vectors_padded_);
    if (d_rotated_) cudaFree(d_rotated_);
    cudaStreamDestroy(stream_);
}

template<typename ComponentT, size_t K>
void GPUEncoder<ComponentT, K>::generate_signs(uint64_t seed) {
    // Generate random ±1 signs on host
    size_t total_signs = K * 3 * padded_dim_;
    std::vector<int8_t> h_signs(total_signs);

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution coin(0.5);

    for (size_t i = 0; i < total_signs; ++i) {
        h_signs[i] = coin(rng) ? 1 : -1;
    }

    // Upload to GPU
    CUDA_CHECK(cudaMalloc(&d_signs_, total_signs * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpy(d_signs_, h_signs.data(), total_signs * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
}

template<typename ComponentT, size_t K>
void GPUEncoder<ComponentT, K>::allocate_buffers(size_t batch_size) {
    if (batch_size <= max_batch_size_) return;

    if (d_vectors_padded_) cudaFree(d_vectors_padded_);
    if (d_rotated_) cudaFree(d_rotated_);

    CUDA_CHECK(cudaMalloc(&d_vectors_padded_, batch_size * padded_dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_rotated_, batch_size * K * padded_dim_ * sizeof(Float)));

    max_batch_size_ = batch_size;
}

template<typename ComponentT, size_t K>
void GPUEncoder<ComponentT, K>::encode_batch(const Float* h_vectors, size_t num_vecs,
                                              ComponentT* h_codes) {
    allocate_buffers(num_vecs);

    // Allocate device memory for input and output
    Float* d_input;
    ComponentT* d_codes;

    CUDA_CHECK(cudaMalloc(&d_input, num_vecs * dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_codes, num_vecs * K * sizeof(ComponentT)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_vectors, num_vecs * dim_ * sizeof(Float),
                                cudaMemcpyHostToDevice, stream_));

    // Process each rotation
    int threads = std::min(static_cast<int>(padded_dim_), 256);
    size_t shared_mem = padded_dim_ * sizeof(Float);

    for (size_t r = 0; r < K; ++r) {
        Float* d_rotated_r = d_rotated_ + r * num_vecs * padded_dim_;

        // Apply rotation chain
        rotation_chain_kernel<<<num_vecs, threads, shared_mem, stream_>>>(
            d_rotated_r, d_input, d_signs_, dim_, padded_dim_, num_vecs, r);
    }

    // Argmax encoding
    int argmax_threads = 256;
    argmax_encode_kernel<ComponentT><<<num_vecs * K, argmax_threads, 0, stream_>>>(
        d_rotated_, d_codes, nullptr, padded_dim_, num_vecs, K, K * padded_dim_);

    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(h_codes, d_codes, num_vecs * K * sizeof(ComponentT),
                                cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaFree(d_input);
    cudaFree(d_codes);
}

template<typename ComponentT, size_t K>
void GPUEncoder<ComponentT, K>::encode_queries_batch(const Float* h_vectors, size_t num_queries,
                                                      ComponentT* h_codes, Float* h_magnitudes) {
    allocate_buffers(num_queries);

    Float* d_input;
    ComponentT* d_codes;
    Float* d_magnitudes;

    CUDA_CHECK(cudaMalloc(&d_input, num_queries * dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_codes, num_queries * K * sizeof(ComponentT)));
    CUDA_CHECK(cudaMalloc(&d_magnitudes, num_queries * K * sizeof(Float)));

    CUDA_CHECK(cudaMemcpyAsync(d_input, h_vectors, num_queries * dim_ * sizeof(Float),
                                cudaMemcpyHostToDevice, stream_));

    int threads = std::min(static_cast<int>(padded_dim_), 256);
    size_t shared_mem = padded_dim_ * sizeof(Float);

    for (size_t r = 0; r < K; ++r) {
        Float* d_rotated_r = d_rotated_ + r * num_queries * padded_dim_;

        rotation_chain_kernel<<<num_queries, threads, shared_mem, stream_>>>(
            d_rotated_r, d_input, d_signs_, dim_, padded_dim_, num_queries, r);
    }

    int argmax_threads = 256;
    argmax_encode_kernel<ComponentT><<<num_queries * K, argmax_threads, 0, stream_>>>(
        d_rotated_, d_codes, d_magnitudes, padded_dim_, num_queries, K, K * padded_dim_);

    CUDA_CHECK(cudaMemcpyAsync(h_codes, d_codes, num_queries * K * sizeof(ComponentT),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_magnitudes, d_magnitudes, num_queries * K * sizeof(Float),
                                cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaFree(d_input);
    cudaFree(d_codes);
    cudaFree(d_magnitudes);
}

// Explicit instantiations
template class GPUEncoder<uint8_t, 16>;
template class GPUEncoder<uint8_t, 32>;
template class GPUEncoder<uint16_t, 16>;
template class GPUEncoder<uint16_t, 32>;

// =============================================================================
// GPUBruteForce Implementation
// =============================================================================

GPUBruteForce::GPUBruteForce(size_t dim, size_t num_base)
    : dim_(dim), num_base_(num_base), d_base_(nullptr), d_queries_(nullptr),
      d_similarities_(nullptr), d_indices_(nullptr), d_distances_(nullptr),
      max_queries_(0) {

    CUDA_CHECK(cudaMalloc(&d_base_, num_base * dim * sizeof(Float)));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    cublasCreate(reinterpret_cast<cublasHandle_t*>(&cublas_handle_));
    cublasSetStream(static_cast<cublasHandle_t>(cublas_handle_), stream_);
}

GPUBruteForce::~GPUBruteForce() {
    if (d_base_) cudaFree(d_base_);
    if (d_queries_) cudaFree(d_queries_);
    if (d_similarities_) cudaFree(d_similarities_);
    if (d_indices_) cudaFree(d_indices_);
    if (d_distances_) cudaFree(d_distances_);

    cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
    cudaStreamDestroy(stream_);
}

void GPUBruteForce::set_base_vectors(const Float* h_base, size_t num_base) {
    num_base_ = num_base;
    CUDA_CHECK(cudaMemcpy(d_base_, h_base, num_base * dim_ * sizeof(Float),
                          cudaMemcpyHostToDevice));
}

void GPUBruteForce::search(const Float* h_queries, size_t num_queries, size_t k,
                           uint32_t* h_indices, Float* h_distances) {
    // Allocate/reallocate buffers if needed
    if (num_queries > max_queries_) {
        if (d_queries_) cudaFree(d_queries_);
        if (d_similarities_) cudaFree(d_similarities_);
        if (d_indices_) cudaFree(d_indices_);
        if (d_distances_) cudaFree(d_distances_);

        CUDA_CHECK(cudaMalloc(&d_queries_, num_queries * dim_ * sizeof(Float)));
        CUDA_CHECK(cudaMalloc(&d_similarities_, num_queries * num_base_ * sizeof(Float)));
        CUDA_CHECK(cudaMalloc(&d_indices_, num_queries * k * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_distances_, num_queries * k * sizeof(Float)));

        max_queries_ = num_queries;
    }

    // Copy queries to device
    CUDA_CHECK(cudaMemcpyAsync(d_queries_, h_queries, num_queries * dim_ * sizeof(Float),
                                cudaMemcpyHostToDevice, stream_));

    // Matrix multiply: similarities = queries @ base^T
    // Using cuBLAS SGEMM: C = alpha * A * B + beta * C
    // A = queries (num_queries x dim), B = base^T (dim x num_base)
    // C = similarities (num_queries x num_base)
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(static_cast<cublasHandle_t>(cublas_handle_),
                CUBLAS_OP_T, CUBLAS_OP_N,  // B transposed, A not
                num_base_, num_queries, dim_,
                &alpha,
                d_base_, dim_,      // B (base vectors, column-major = row-major transposed)
                d_queries_, dim_,   // A (queries)
                &beta,
                d_similarities_, num_base_);  // C

    // Top-k selection
    size_t shared_mem = k * 256 * (sizeof(Float) + sizeof(uint32_t));
    topk_kernel<<<num_queries, 256, shared_mem, stream_>>>(
        d_similarities_, d_indices_, d_distances_,
        num_base_, num_queries, k);

    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(h_indices, d_indices_, num_queries * k * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_distances, d_distances_, num_queries * k * sizeof(Float),
                                cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void GPUBruteForce::rerank(const Float* h_queries, const uint32_t* h_candidate_ids,
                           size_t num_queries, size_t num_candidates, size_t k,
                           uint32_t* h_indices, Float* h_distances) {
    // Allocate temporary buffers
    Float* d_queries_tmp;
    uint32_t* d_candidate_ids;
    Float* d_candidate_vecs;
    Float* d_sims;

    CUDA_CHECK(cudaMalloc(&d_queries_tmp, num_queries * dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_candidate_ids, num_queries * num_candidates * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_candidate_vecs, num_queries * num_candidates * dim_ * sizeof(Float)));
    CUDA_CHECK(cudaMalloc(&d_sims, num_queries * num_candidates * sizeof(Float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpyAsync(d_queries_tmp, h_queries, num_queries * dim_ * sizeof(Float),
                                cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_candidate_ids, h_candidate_ids,
                                num_queries * num_candidates * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream_));

    // Gather candidate vectors
    size_t total_elements = num_queries * num_candidates * dim_;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gather_vectors_kernel<<<blocks, threads, 0, stream_>>>(
        d_base_, d_candidate_ids, d_candidate_vecs, dim_, num_queries, num_candidates);

    // Compute dot products
    batch_dot_product_kernel<<<num_queries * num_candidates, 128, 0, stream_>>>(
        d_queries_tmp, d_candidate_vecs, d_sims, dim_, num_queries, num_candidates);

    // Top-k from candidates (reuse similarities buffer logic)
    // For simplicity, copy back and do on CPU for now
    std::vector<Float> h_sims(num_queries * num_candidates);
    CUDA_CHECK(cudaMemcpyAsync(h_sims.data(), d_sims,
                                num_queries * num_candidates * sizeof(Float),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // CPU top-k selection from candidates
    #pragma omp parallel for
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<std::pair<Float, uint32_t>> scored(num_candidates);
        for (size_t c = 0; c < num_candidates; ++c) {
            scored[c] = {-h_sims[q * num_candidates + c], h_candidate_ids[q * num_candidates + c]};
        }
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end());

        for (size_t i = 0; i < k; ++i) {
            h_indices[q * k + i] = scored[i].second;
            h_distances[q * k + i] = scored[i].first;
        }
    }

    cudaFree(d_queries_tmp);
    cudaFree(d_candidate_ids);
    cudaFree(d_candidate_vecs);
    cudaFree(d_sims);
}

}  // namespace cuda
}  // namespace cphnsw
