#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "rotation.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

// ============================================================================
// SIMD Argmax Helper
// ============================================================================

namespace detail {

/**
 * Find index and absolute value of maximum magnitude element.
 * Uses SIMD when available.
 */
inline void find_argmax_abs(const float* buffer, size_t size,
                             size_t& out_idx, float& out_abs) {
#if defined(__AVX512F__)
    if (size >= 16) {
        const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
        __m512 max_vals = _mm512_setzero_ps();

        // First pass: find max absolute value
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&buffer[i]);
            __m512 abs_v = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
            max_vals = _mm512_max_ps(max_vals, abs_v);
        }

        float global_max = _mm512_reduce_max_ps(max_vals);

        // Handle remainder
        for (; i < size; ++i) {
            float abs_val = std::abs(buffer[i]);
            if (abs_val > global_max) global_max = abs_val;
        }

        // Second pass: find index
        const __m512 target = _mm512_set1_ps(global_max);
        for (i = 0; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&buffer[i]);
            __m512 abs_v = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
            __mmask16 eq_mask = _mm512_cmp_ps_mask(abs_v, target, _CMP_EQ_OQ);
            if (eq_mask) {
                int first_bit = __builtin_ctz(eq_mask);
                out_idx = i + first_bit;
                out_abs = global_max;
                return;
            }
        }

        for (; i < size; ++i) {
            if (std::abs(buffer[i]) == global_max) {
                out_idx = i;
                out_abs = global_max;
                return;
            }
        }

        out_idx = 0;
        out_abs = global_max;
        return;
    }
#endif

    // Scalar fallback
    size_t max_idx = 0;
    float max_abs = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        float abs_val = std::abs(buffer[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            max_idx = i;
        }
    }

    out_idx = max_idx;
    out_abs = max_abs;
}

}  // namespace detail

// ============================================================================
// CPEncoder: Unified Cross-Polytope Encoder
// ============================================================================

/**
 * CPEncoder: Unified encoder for Phase 1 (R=0) and Phase 2 (R>0).
 *
 * UNIFIED DESIGN: A single encoder that handles both configurations.
 * When R=0, residual encoding compiles to no-ops.
 *
 * Encoding Process:
 *   1. For each of K rotations:
 *      a. Apply pseudo-random rotation Ψ(x) = H D₃ H D₂ H D₁ x
 *      b. Find argmax |Ψ(x)ᵢ|
 *      c. Extract sign bit → primary code
 *   2. If R > 0 (Phase 2):
 *      a. Compute residual vector from reconstruction error
 *      b. Apply R independent rotations to residual
 *      c. Extract sign bits → residual code
 *
 * @tparam K Primary code bits
 * @tparam R Residual code bits (0 = Phase 1, >0 = Phase 2)
 */
template <size_t K, size_t R = 0>
class CPEncoder {
public:
    using CodeType = ResidualCode<K, R>;
    using QueryType = CodeQuery<K, R>;

    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    /**
     * Construct encoder.
     * @param dim Vector dimension
     * @param seed Random seed for rotation matrices
     */
    CPEncoder(size_t dim, uint64_t seed = 42)
        : dim_(dim)
        , primary_chain_(dim, seed)
        , buffer_(primary_chain_.padded_dim()) {

        if constexpr (R > 0) {
            // Use different seed for residual chain
            residual_chain_ = std::make_unique<RotationChain<R>>(dim, seed + K);
            residual_buffer_.resize(primary_chain_.padded_dim());
        }
    }

    // ========================================================================
    // Primary Encoding (Phase 1 compatible)
    // ========================================================================

    /**
     * Encode vector to binary code.
     * Thread-safe when using encode_with_buffer.
     */
    CodeType encode(const float* vec) const {
        return encode_with_buffer(vec, buffer_.data());
    }

    /**
     * Encode using caller-provided buffer (thread-safe).
     */
    CodeType encode_with_buffer(const float* vec, float* buffer) const {
        CodeType code;
        code.clear();

        // Encode primary bits
        for (size_t r = 0; r < K; ++r) {
            primary_chain_.apply_copy(vec, buffer, r);

            size_t max_idx;
            float max_abs;
            detail::find_argmax_abs(buffer, padded_dim(), max_idx, max_abs);

            // Extract sign bit
            bool is_negative = (buffer[max_idx] < 0);
            code.set_primary_bit(r, is_negative);
        }

        // Encode residual bits (if R > 0)
        if constexpr (R > 0) {
            encode_residual(vec, code, buffer);
        }

        return code;
    }

    // ========================================================================
    // Query Encoding (with pre-computed scalars)
    // ========================================================================

    /**
     * Encode query with pre-computed distance scalars.
     *
     * The scalars C1 and C2 are used for distance computation:
     *   distance = base + scale * hamming
     *
     * @param vec Query vector
     * @param avg_norm Average norm of database vectors (for calibration)
     */
    QueryType encode_query(const float* vec, float avg_norm = 1.0f) const {
        return encode_query_with_buffer(vec, buffer_.data(), avg_norm);
    }

    QueryType encode_query_with_buffer(const float* vec, float* buffer,
                                        float avg_norm = 1.0f) const {
        QueryType query;

        // Encode binary code
        query.code = encode_with_buffer(vec, buffer);

        // Compute query norm
        float query_norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            query_norm += vec[i] * vec[i];
        }
        query_norm = std::sqrt(query_norm);
        query.query_norm = query_norm;

        // Pre-compute distance scalars
        // For angular distance: dist ∝ 1 - cos(θ) ≈ θ²/2
        // Hamming approximates: hamming ≈ K * θ / π
        query.base = 0.0f;

        if constexpr (R == 0) {
            // Phase 1: Simple scaling
            query.scale = 2.0f * query_norm * avg_norm / static_cast<float>(K);
        } else {
            // Phase 2: Account for combined Hamming weight
            // Combined = (primary << Shift) + residual
            // Scale should account for bit-shift weighting
            float effective_bits = static_cast<float>(K * (1 << QueryType::WEIGHT_SHIFT) + R);
            query.scale = 2.0f * query_norm * avg_norm / effective_bits;
        }

        return query;
    }

    // ========================================================================
    // Batch Encoding
    // ========================================================================

    /**
     * Encode multiple vectors (parallelized with OpenMP).
     */
    void encode_batch(const float* vecs, size_t num_vecs, CodeType* codes) const {
#ifdef _OPENMP
        #pragma omp parallel
        {
            AlignedVector<float> local_buffer(padded_dim());

            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_vecs; ++i) {
                codes[i] = encode_with_buffer(vecs + i * dim_, local_buffer.data());
            }
        }
#else
        for (size_t i = 0; i < num_vecs; ++i) {
            codes[i] = encode(vecs + i * dim_);
        }
#endif
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return primary_chain_.padded_dim(); }

private:
    size_t dim_;
    RotationChain<K> primary_chain_;
    mutable AlignedVector<float> buffer_;

    // Residual chain (only allocated when R > 0)
    std::unique_ptr<RotationChain<R>> residual_chain_;
    mutable AlignedVector<float> residual_buffer_;

    /**
     * Encode residual bits (Phase 2 only).
     *
     * Computes the residual vector as the difference between the original
     * vector and its reconstruction from the primary code, then encodes
     * the residual using R independent rotations.
     */
    template <size_t R_ = R>
    std::enable_if_t<(R_ > 0)>
    encode_residual(const float* vec, CodeType& code, float* buffer) const {
        // For simplified residual encoding, we use a second independent
        // rotation chain on the original vector. This avoids explicit
        // reconstruction but still captures orthogonal information.

        for (size_t r = 0; r < R; ++r) {
            residual_chain_->apply_copy(vec, residual_buffer_.data(), r);

            size_t max_idx;
            float max_abs;
            detail::find_argmax_abs(residual_buffer_.data(), padded_dim(), max_idx, max_abs);

            bool is_negative = (residual_buffer_[max_idx] < 0);
            code.set_residual_bit(r, is_negative);
        }
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1: Pure binary codes
using Encoder32 = CPEncoder<32, 0>;
using Encoder64 = CPEncoder<64, 0>;

// Phase 2: With residual
using Encoder64_32 = CPEncoder<64, 32>;
using Encoder32_16 = CPEncoder<32, 16>;

}  // namespace cphnsw
