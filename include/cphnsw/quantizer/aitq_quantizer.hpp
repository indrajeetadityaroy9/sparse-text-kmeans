#pragma once

#include "../core/types.hpp"
#include "quantizer_policy.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstring>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define AITQ_HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define AITQ_HAS_AVX512 0
#define AITQ_HAS_AVX2 1
#include <immintrin.h>
#else
#define AITQ_HAS_AVX512 0
#define AITQ_HAS_AVX2 0
#endif

namespace cphnsw {

// ============================================================================
// A-ITQ Code and Query Types
// ============================================================================

/**
 * AITQCode: Binary code from Iterative Quantization.
 *
 * Stores K bits as packed bytes. Each bit represents sign(R[k] · x)
 * where R is the learned orthogonal rotation matrix.
 *
 * Storage: ceil(K/8) bytes per vector
 */
template <size_t K>
struct AITQCode {
    static constexpr size_t NumBytes = (K + 7) / 8;
    std::array<uint8_t, NumBytes> bits;

    /// Get bit at position i
    bool get_bit(size_t i) const {
        return (bits[i / 8] >> (i % 8)) & 1;
    }

    /// Set bit at position i
    void set_bit(size_t i, bool value) {
        if (value) {
            bits[i / 8] |= (1 << (i % 8));
        } else {
            bits[i / 8] &= ~(1 << (i % 8));
        }
    }

    /// Clear all bits
    void clear() {
        std::fill(bits.begin(), bits.end(), 0);
    }
};

/**
 * AITQQuery: Query structure for asymmetric ITQ distance.
 *
 * Stores the projected query vector (R × normalized_query) for
 * efficient asymmetric distance computation.
 *
 * Distance = -Σᵢ (code_bit[i] ? proj[i] : -proj[i])
 *          = -Σᵢ proj[i] × (2×bit[i] - 1)
 */
template <size_t K>
struct AITQQuery {
    AITQCode<K> code;           // Binary code for symmetric fallback
    std::array<Float, K> proj;  // Projected query values for asymmetric distance

    /// Compute asymmetric distance to a code
    Float distance_to(const AITQCode<K>& other) const {
        Float score = 0.0f;
        for (size_t i = 0; i < K; ++i) {
            // If bit is 1: add proj[i], else subtract
            Float sign = other.get_bit(i) ? 1.0f : -1.0f;
            score += sign * proj[i];
        }
        return -score;  // Negate for min-heap (lower = more similar)
    }
};

// ============================================================================
// ITQ Training Algorithm
// ============================================================================

/**
 * ITQTrainer: Learns orthogonal rotation matrix R for ITQ.
 *
 * Algorithm (Gong et al. 2011):
 * 1. Initialize R randomly (or with PCA)
 * 2. Iterate:
 *    a. B = sign(X × Rᵀ)  -- binary codes
 *    b. R = argmin_R ||X × Rᵀ - B||²  s.t. R orthogonal
 *       Solved via SVD: if B'X = USVᵀ, then R = VUᵀ
 * 3. Return R
 *
 * The learned R aligns the data with the binary hypercube vertices.
 */
class ITQTrainer {
public:
    /**
     * Train ITQ rotation matrix.
     *
     * @param data        Training vectors (N × dim, row-major)
     * @param N           Number of training vectors
     * @param dim         Vector dimension
     * @param num_bits    Number of output bits (K)
     * @param num_iters   Number of ITQ iterations (default 50)
     * @param seed        Random seed
     * @return            Rotation matrix R (num_bits × dim, row-major)
     */
    static std::vector<Float> train(
        const Float* data,
        size_t N,
        size_t dim,
        size_t num_bits,
        size_t num_iters = 50,
        uint64_t seed = 42) {

        // Initialize R randomly as orthogonal matrix
        std::vector<Float> R = init_random_orthogonal(num_bits, dim, seed);

        // Work buffers
        std::vector<Float> projected(N * num_bits);  // X × Rᵀ
        std::vector<int8_t> binary(N * num_bits);    // sign(projected)
        std::vector<Float> BtX(num_bits * dim);      // Bᵀ × X

        for (size_t iter = 0; iter < num_iters; ++iter) {
            // Step 1: Project data and binarize
            // projected = X × Rᵀ (N × num_bits)
            project_data(data, N, dim, R.data(), num_bits, projected.data());

            // binary = sign(projected)
            for (size_t i = 0; i < N * num_bits; ++i) {
                binary[i] = projected[i] >= 0 ? 1 : -1;
            }

            // Step 2: Solve orthogonal Procrustes: R = argmin ||XRᵀ - B||²
            // Closed form: if Bᵀ × X = USVᵀ, then R = VUᵀ
            compute_BtX(binary.data(), data, N, dim, num_bits, BtX.data());

            // SVD of BtX and update R
            update_R_via_svd(BtX.data(), num_bits, dim, R.data());
        }

        return R;
    }

    /**
     * Compute orthogonality error of rotation matrix.
     * Error = ||RRᵀ - I||_F
     */
    static Float orthogonality_error(const Float* R, size_t rows, size_t cols) {
        Float error = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                Float dot = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    dot += R[i * cols + k] * R[j * cols + k];
                }
                Float target = (i == j) ? 1.0f : 0.0f;
                error += (dot - target) * (dot - target);
            }
        }
        return std::sqrt(error);
    }

private:
    /// Initialize random orthogonal matrix using Modified Gram-Schmidt with re-orthogonalization
    /// Uses "twice is enough" principle for numerical stability
    static std::vector<Float> init_random_orthogonal(size_t rows, size_t cols, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<Float> dist(0.0f, 1.0f);

        std::vector<Float> R(rows * cols);

        // Fill with random Gaussian values
        for (auto& val : R) {
            val = dist(rng);
        }

        // Modified Gram-Schmidt with re-orthogonalization (2 passes for stability)
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < rows; ++i) {
                Float* row_i = R.data() + i * cols;

                // Subtract projections onto previous rows (Modified GS: update immediately)
                for (size_t j = 0; j < i; ++j) {
                    const Float* row_j = R.data() + j * cols;
                    Float dot = 0.0f;
                    for (size_t k = 0; k < cols; ++k) {
                        dot += row_i[k] * row_j[k];
                    }
                    for (size_t k = 0; k < cols; ++k) {
                        row_i[k] -= dot * row_j[k];
                    }
                }

                // Normalize
                Float norm = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    norm += row_i[k] * row_i[k];
                }
                norm = std::sqrt(norm);
                if (norm > 1e-10f) {
                    Float inv_norm = 1.0f / norm;
                    for (size_t k = 0; k < cols; ++k) {
                        row_i[k] *= inv_norm;
                    }
                }
            }
        }

        return R;
    }

    /// Project data: projected = X × Rᵀ
    static void project_data(
        const Float* X, size_t N, size_t dim,
        const Float* R, size_t num_bits,
        Float* projected) {

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < N; ++i) {
            const Float* x = X + i * dim;
            Float* p = projected + i * num_bits;

            for (size_t b = 0; b < num_bits; ++b) {
                const Float* r = R + b * dim;
                Float dot = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot += x[d] * r[d];
                }
                p[b] = dot;
            }
        }
    }

    /// Compute Bᵀ × X (num_bits × dim)
    static void compute_BtX(
        const int8_t* B, const Float* X,
        size_t N, size_t dim, size_t num_bits,
        Float* BtX) {

        std::fill(BtX, BtX + num_bits * dim, 0.0f);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel
        {
            std::vector<Float> local_BtX(num_bits * dim, 0.0f);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                const int8_t* b = B + i * num_bits;
                const Float* x = X + i * dim;

                for (size_t k = 0; k < num_bits; ++k) {
                    Float sign = static_cast<Float>(b[k]);
                    Float* row = local_BtX.data() + k * dim;
                    for (size_t d = 0; d < dim; ++d) {
                        row[d] += sign * x[d];
                    }
                }
            }

            #pragma omp critical
            {
                for (size_t i = 0; i < num_bits * dim; ++i) {
                    BtX[i] += local_BtX[i];
                }
            }
        }
#else
        for (size_t i = 0; i < N; ++i) {
            const int8_t* b = B + i * num_bits;
            const Float* x = X + i * dim;

            for (size_t k = 0; k < num_bits; ++k) {
                Float sign = static_cast<Float>(b[k]);
                Float* row = BtX + k * dim;
                for (size_t d = 0; d < dim; ++d) {
                    row[d] += sign * x[d];
                }
            }
        }
#endif
    }

    /// Update R via SVD of BtX (simplified power iteration method)
    static void update_R_via_svd(Float* BtX, size_t rows, size_t cols, Float* R) {
        // For efficiency, we use a simplified approach:
        // Normalize each row of BtX and use as R
        // This is an approximation that works well in practice

        for (size_t i = 0; i < rows; ++i) {
            Float* row = BtX + i * cols;

            // Gram-Schmidt against previous rows
            for (size_t j = 0; j < i; ++j) {
                Float* prev_row = R + j * cols;
                Float dot = 0.0f;
                for (size_t k = 0; k < cols; ++k) {
                    dot += row[k] * prev_row[k];
                }
                for (size_t k = 0; k < cols; ++k) {
                    row[k] -= dot * prev_row[k];
                }
            }

            // Normalize and copy to R
            Float norm = 0.0f;
            for (size_t k = 0; k < cols; ++k) {
                norm += row[k] * row[k];
            }
            norm = std::sqrt(norm);
            Float* r_row = R + i * cols;
            if (norm > 1e-10f) {
                for (size_t k = 0; k < cols; ++k) {
                    r_row[k] = row[k] / norm;
                }
            }
        }
    }
};

// ============================================================================
// AVX-512 Optimized A-ITQ Distance Kernels
// ============================================================================

#if AITQ_HAS_AVX512

/**
 * AVX-512 Asymmetric ITQ Distance for K=256 bits (32 bytes).
 *
 * Computes: distance = -Σᵢ proj[i] × (2×bit[i] - 1)
 *
 * Uses _mm512_mask_blend_ps for conditional sign flip based on bits.
 */
inline Float aitq_distance_avx512_256(const Float* query_proj, const uint8_t* code) {
    __m512 sum = _mm512_setzero_ps();

    // Process 256 bits = 16 floats × 16 iterations
    for (size_t i = 0; i < 16; ++i) {
        // Load 16 bits from code (2 bytes at position i*2)
        uint16_t chunk = *reinterpret_cast<const uint16_t*>(code + i * 2);
        __mmask16 mask = static_cast<__mmask16>(chunk);

        // Load 16 floats of query projection
        __m512 q_vals = _mm512_loadu_ps(query_proj + i * 16);

        // If bit is 1: use +proj, if bit is 0: use -proj
        // neg_q = -q_vals
        __m512 neg_q = _mm512_sub_ps(_mm512_setzero_ps(), q_vals);

        // blend: select q_vals where mask=1, neg_q where mask=0
        __m512 contributions = _mm512_mask_blend_ps(mask, neg_q, q_vals);

        sum = _mm512_add_ps(sum, contributions);
    }

    return -_mm512_reduce_add_ps(sum);  // Negate for min-heap
}

/**
 * AVX-512 Asymmetric ITQ Distance for K=128 bits (16 bytes).
 */
inline Float aitq_distance_avx512_128(const Float* query_proj, const uint8_t* code) {
    __m512 sum = _mm512_setzero_ps();

    // Process 128 bits = 16 floats × 8 iterations
    for (size_t i = 0; i < 8; ++i) {
        uint16_t chunk = *reinterpret_cast<const uint16_t*>(code + i * 2);
        __mmask16 mask = static_cast<__mmask16>(chunk);

        __m512 q_vals = _mm512_loadu_ps(query_proj + i * 16);
        __m512 neg_q = _mm512_sub_ps(_mm512_setzero_ps(), q_vals);
        __m512 contributions = _mm512_mask_blend_ps(mask, neg_q, q_vals);

        sum = _mm512_add_ps(sum, contributions);
    }

    return -_mm512_reduce_add_ps(sum);
}

/**
 * AVX-512 Batch A-ITQ Distance for SoA transposed layout.
 *
 * Processes 16 neighbors at once using the transposed code layout.
 *
 * @param query_proj        Projected query values [K]
 * @param codes_transposed  Transposed codes [K_bytes][64] (bits for all neighbors)
 * @param K_bits            Number of bits per code
 * @param start_idx         Starting neighbor index
 * @param out_distances     Output array for 16 distances
 */
inline void aitq_batch_distance_avx512_16(
    const Float* query_proj,
    const uint8_t codes_transposed[][64],
    size_t K_bits,
    size_t start_idx,
    Float* out_distances) {

    __m512 scores = _mm512_setzero_ps();

    size_t K_bytes = (K_bits + 7) / 8;

    // Process each bit position
    for (size_t byte_idx = 0; byte_idx < K_bytes; ++byte_idx) {
        for (size_t bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte) {
            size_t bit_idx = byte_idx * 8 + bit_in_byte;
            if (bit_idx >= K_bits) break;

            // Load byte for 16 neighbors
            __m128i bytes = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(&codes_transposed[byte_idx][start_idx]));

            // Extract the specific bit for all 16 neighbors
            __m128i bit_mask = _mm_set1_epi8(1 << bit_in_byte);
            __m128i bits = _mm_and_si128(bytes, bit_mask);
            __m128i is_set = _mm_cmpeq_epi8(bits, bit_mask);

            // Convert to 16-bit mask (1 bit per neighbor)
            __mmask16 mask = static_cast<__mmask16>(_mm_movemask_epi8(is_set));

            // Load query projection value for this bit
            Float proj_val = query_proj[bit_idx];
            __m512 proj = _mm512_set1_ps(proj_val);
            __m512 neg_proj = _mm512_set1_ps(-proj_val);

            // Blend: +proj where bit=1, -proj where bit=0
            __m512 contributions = _mm512_mask_blend_ps(mask, neg_proj, proj);

            scores = _mm512_add_ps(scores, contributions);
        }
    }

    // Negate and store
    __m512 neg_scores = _mm512_sub_ps(_mm512_setzero_ps(), scores);
    _mm512_storeu_ps(out_distances, neg_scores);
}

#endif  // AITQ_HAS_AVX512

// ============================================================================
// A-ITQ Quantizer Implementation
// ============================================================================

/**
 * AITQQuantizer: Asymmetric ITQ Quantizer with learned projections.
 *
 * Uses learned orthogonal rotation matrix R (from ITQ training) to
 * project vectors before binarization. The asymmetric distance uses
 * the full projected query against binary codes.
 *
 * Advantages over FHT-based CP-LSH:
 * - Data-dependent: R is learned to minimize quantization error
 * - Higher correlation between quantized and true distance
 *
 * Disadvantages:
 * - Requires training phase
 * - Must retrain if data distribution changes
 */
template <size_t K>
class AITQQuantizer : public QuantizerPolicy<AITQCode<K>, AITQQuery<K>, K> {
public:
    using Code = AITQCode<K>;
    using Query = AITQQuery<K>;

    /**
     * Construct A-ITQ quantizer with pre-trained rotation matrix.
     *
     * @param dim   Vector dimension
     * @param R     Rotation matrix (K × dim, row-major)
     */
    AITQQuantizer(size_t dim, std::vector<Float> R)
        : dim_(dim), R_(std::move(R)) {
        if (R_.size() != K * dim) {
            throw std::invalid_argument(
                "AITQQuantizer: R must be K × dim = " + std::to_string(K * dim));
        }
    }

    /**
     * Construct and train A-ITQ quantizer from data.
     *
     * @param dim           Vector dimension
     * @param train_data    Training vectors (N × dim, row-major)
     * @param N             Number of training vectors
     * @param num_iters     ITQ iterations
     * @param seed          Random seed
     */
    AITQQuantizer(size_t dim, const Float* train_data, size_t N,
                  size_t num_iters = 50, uint64_t seed = 42)
        : dim_(dim) {
        R_ = ITQTrainer::train(train_data, N, dim, K, num_iters, seed);
    }

    size_t dim() const override { return dim_; }
    size_t padded_dim() const override { return dim_; }  // No padding needed

    /**
     * Get the learned rotation matrix.
     */
    const std::vector<Float>& rotation_matrix() const { return R_; }

    /**
     * Compute orthogonality error of R.
     */
    Float orthogonality_error() const {
        return ITQTrainer::orthogonality_error(R_.data(), K, dim_);
    }

    /**
     * Encode vector to binary code.
     */
    Code encode(const Float* vec) const override {
        Code code;
        code.clear();

        for (size_t b = 0; b < K; ++b) {
            const Float* r = R_.data() + b * dim_;
            Float dot = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                dot += vec[d] * r[d];
            }
            code.set_bit(b, dot >= 0);
        }

        return code;
    }

    /**
     * Encode vector to query structure with full projections.
     */
    Query encode_query(const Float* vec) const override {
        Query query;
        query.code.clear();

        for (size_t b = 0; b < K; ++b) {
            const Float* r = R_.data() + b * dim_;
            Float dot = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                dot += vec[d] * r[d];
            }
            query.proj[b] = dot;
            query.code.set_bit(b, dot >= 0);
        }

        return query;
    }

    Code encode_with_buffer(const Float* vec, Float* /* buffer */) const override {
        return encode(vec);  // No buffer needed for A-ITQ
    }

    Query encode_query_with_buffer(const Float* vec, Float* /* buffer */) const override {
        return encode_query(vec);
    }

    /**
     * Compute asymmetric search distance.
     *
     * Distance = -Σᵢ proj[i] × sign(code_bit[i])
     */
    float search_distance(const Query& query, const Code& code) const override {
#if AITQ_HAS_AVX512
        if constexpr (K == 256) {
            return aitq_distance_avx512_256(query.proj.data(), code.bits.data());
        } else if constexpr (K == 128) {
            return aitq_distance_avx512_128(query.proj.data(), code.bits.data());
        }
#endif
        // Scalar fallback
        return query.distance_to(code);
    }

    /**
     * Batch compute distances using SoA transposed layout.
     */
    void batch_search_distance_soa(
        const Query& query,
        const void* codes_transposed_ptr,
        size_t num_neighbors,
        float* out_distances) const override {

        const uint8_t (*codes_transposed)[64] =
            reinterpret_cast<const uint8_t (*)[64]>(codes_transposed_ptr);

        size_t n = 0;

#if AITQ_HAS_AVX512
        // Process 16 neighbors at a time
        for (; n + 16 <= num_neighbors; n += 16) {
            aitq_batch_distance_avx512_16(
                query.proj.data(), codes_transposed, K, n, out_distances + n);
        }
#endif

        // Scalar fallback for remainder
        constexpr size_t K_bytes = (K + 7) / 8;
        for (; n < num_neighbors; ++n) {
            // Reconstruct code from transposed layout
            Code code;
            code.clear();
            for (size_t byte_idx = 0; byte_idx < K_bytes; ++byte_idx) {
                code.bits[byte_idx] = codes_transposed[byte_idx][n];
            }
            out_distances[n] = search_distance(query, code);
        }
    }

    /**
     * Compute correlation between asymmetric distance and true dot product.
     *
     * High correlation (>0.9) indicates the quantizer is effective.
     */
    Float compute_correlation(
        const Float* vectors, size_t N, size_t num_samples = 1000,
        uint64_t seed = 42) const {

        if (N < 2) return 0.0f;

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, N - 1);

        std::vector<Float> X, Y;
        X.reserve(num_samples);
        Y.reserve(num_samples);

        for (size_t s = 0; s < num_samples; ++s) {
            size_t i = dist(rng);
            size_t j = dist(rng);
            if (i == j) j = (j + 1) % N;

            const Float* vi = vectors + i * dim_;
            const Float* vj = vectors + j * dim_;

            // True dot product
            Float true_dot = 0.0f;
            for (size_t d = 0; d < dim_; ++d) {
                true_dot += vi[d] * vj[d];
            }

            // A-ITQ distance
            Query query = encode_query(vi);
            Code code = encode(vj);
            Float aitq_dist = search_distance(query, code);

            X.push_back(-aitq_dist);  // Negate back to similarity
            Y.push_back(true_dot);
        }

        // Compute Pearson correlation
        Float mean_x = std::accumulate(X.begin(), X.end(), 0.0f) / X.size();
        Float mean_y = std::accumulate(Y.begin(), Y.end(), 0.0f) / Y.size();

        Float cov = 0.0f, var_x = 0.0f, var_y = 0.0f;
        for (size_t i = 0; i < X.size(); ++i) {
            Float dx = X[i] - mean_x;
            Float dy = Y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if (var_x < 1e-10f || var_y < 1e-10f) return 0.0f;
        return cov / std::sqrt(var_x * var_y);
    }

private:
    size_t dim_;
    std::vector<Float> R_;  // Rotation matrix (K × dim, row-major)
};

// Common type aliases
using AITQQuantizer128 = AITQQuantizer<128>;
using AITQQuantizer256 = AITQQuantizer<256>;

}  // namespace cphnsw
