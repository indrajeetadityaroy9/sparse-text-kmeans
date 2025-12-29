#pragma once

#include "../core/types.hpp"
#include "rotation_chain.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

#if CPHNSW_HAS_AVX512
#include <immintrin.h>
#endif

namespace cphnsw {

/**
 * SIMD-accelerated argmax of absolute values.
 *
 * Finds index and value of max(|buffer[i]|) over the array.
 * Uses AVX-512 when available for ~4x speedup on large arrays.
 *
 * @param buffer  Input array
 * @param size    Array size
 * @param out_idx Output: index of maximum
 * @param out_abs Output: absolute value of maximum
 */
inline void find_argmax_abs(const Float* buffer, size_t size,
                            size_t& out_idx, Float& out_abs) {
#if CPHNSW_HAS_AVX512
    // AVX-512: Process 16 floats at a time
    if (size >= 16) {
        // Sign mask for absolute value: clear MSB
        const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);

        __m512 max_vals = _mm512_setzero_ps();

        // First pass: find global max absolute value using SIMD reduction
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&buffer[i]);
            // Compute absolute value by clearing sign bit
            __m512 abs_v = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
            max_vals = _mm512_max_ps(max_vals, abs_v);
        }

        // Reduce 16 values to scalar max
        Float global_max = _mm512_reduce_max_ps(max_vals);

        // Handle remainder with scalar
        for (; i < size; ++i) {
            Float abs_val = std::abs(buffer[i]);
            if (abs_val > global_max) {
                global_max = abs_val;
            }
        }

        // Second pass: find index of max (scalar, but array is hot in cache)
        // Use SIMD comparison to find matching indices faster
        const __m512 target = _mm512_set1_ps(global_max);
        for (i = 0; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&buffer[i]);
            __m512 abs_v = _mm512_castsi512_ps(
                _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
            __mmask16 eq_mask = _mm512_cmp_ps_mask(abs_v, target, _CMP_EQ_OQ);
            if (eq_mask) {
                // Found it - get first matching index
                int first_bit = __builtin_ctz(eq_mask);
                out_idx = i + first_bit;
                out_abs = global_max;
                return;
            }
        }

        // Check remainder
        for (; i < size; ++i) {
            if (std::abs(buffer[i]) == global_max) {
                out_idx = i;
                out_abs = global_max;
                return;
            }
        }

        // Shouldn't reach here, but fallback
        out_idx = 0;
        out_abs = global_max;
        return;
    }
#endif

    // Scalar fallback (or for small arrays)
    size_t max_idx = 0;
    Float max_abs = 0.0f;

    for (size_t i = 0; i < size; ++i) {
        Float abs_val = std::abs(buffer[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            max_idx = i;
        }
    }

    out_idx = max_idx;
    out_abs = max_abs;
}

/**
 * Cross-Polytope Encoder
 *
 * Encodes vectors into compact Cross-Polytope hash codes.
 *
 * For each of K rotations:
 *   1. Apply pseudo-random rotation Ψ(x) = H D₃ H D₂ H D₁ x
 *   2. Find argmax |rotated_x[i]|
 *   3. Encode (index, sign) into a single component
 *
 * Template parameters:
 * - ComponentT: uint8_t for d <= 128, uint16_t for d > 128
 * - K: Number of rotations (code width)
 */
template <typename ComponentT, size_t K>
class CPEncoder {
public:
    /**
     * Construct encoder.
     *
     * @param dim   Original vector dimension
     * @param seed  Random seed for rotation matrices
     */
    CPEncoder(size_t dim, uint64_t seed)
        : dim_(dim), rotation_chain_(dim, K, seed) {

        // Pre-allocate work buffer
        buffer_.resize(rotation_chain_.padded_dim());
    }

    /// Get original dimension
    size_t dim() const { return dim_; }

    /// Get padded dimension
    size_t padded_dim() const { return rotation_chain_.padded_dim(); }

    /// Get code width
    static constexpr size_t code_width() { return K; }

    /**
     * Encode vector to Cross-Polytope code.
     *
     * @param vec   Input vector (length >= dim_)
     * @return      Encoded CPCode
     */
    CPCode<ComponentT, K> encode(const Float* vec) const {
        return encode_with_buffer(vec, buffer_.data());
    }

    /**
     * Encode vector to Cross-Polytope code using provided buffer.
     * Thread-safe: Uses caller-provided buffer instead of shared member.
     *
     * @param vec     Input vector (length >= dim_)
     * @param buffer  Work buffer (length >= padded_dim_)
     * @return        Encoded CPCode
     */
    CPCode<ComponentT, K> encode_with_buffer(const Float* vec, Float* buffer) const {
        CPCode<ComponentT, K> code;

        for (size_t r = 0; r < K; ++r) {
            // Apply rotation to copy
            rotation_chain_.apply_copy(vec, buffer, r);

            // Find argmax |buffer[i]| using SIMD-accelerated function
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer, rotation_chain_.padded_dim(), max_idx, max_abs);

            // Encode: (index << 1) | sign_bit
            // sign_bit = 1 if negative, 0 if positive
            bool is_negative = (buffer[max_idx] < 0);
            code.components[r] = CPCode<ComponentT, K>::encode(max_idx, is_negative);
        }

        return code;
    }

    /**
     * Encode vector as CPQuery with full rotated vectors for dot product estimation.
     *
     * This is the PRIMARY encoding method for search queries.
     * Stores the full rotated vectors to enable accurate dot product reconstruction:
     *   Score = sum_r( sign_r * rotated_vec[r][index_r] )
     *
     * @param vec   Input vector (length >= dim_)
     * @return      CPQuery with code and rotated vectors
     */
    CPQuery<ComponentT, K> encode_query(const Float* vec) const {
        CPQuery<ComponentT, K> query;
        size_t pdim = rotation_chain_.padded_dim();

        for (size_t r = 0; r < K; ++r) {
            // Apply rotation
            rotation_chain_.apply_copy(vec, buffer_.data(), r);

            // Store the FULL rotated vector for dot product reconstruction
            query.rotated_vecs[r].resize(pdim);
            std::copy(buffer_.begin(), buffer_.begin() + pdim, query.rotated_vecs[r].begin());

            // Find argmax |buffer[i]| using SIMD-accelerated function
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer_.data(), pdim, max_idx, max_abs);

            // Store primary code component
            bool is_negative = (buffer_[max_idx] < 0);
            query.primary_code.components[r] =
                CPCode<ComponentT, K>::encode(max_idx, is_negative);

            // Store magnitude for multiprobe ranking
            query.magnitudes[r] = max_abs;
            query.original_indices[r] = static_cast<uint32_t>(max_idx);
        }

        return query;
    }

    /**
     * Encode vector with additional probe data for multiprobe search.
     * Alias for encode_query (backwards compatibility).
     *
     * @param vec   Input vector (length >= dim_)
     * @return      CPQuery with code and multiprobe data
     */
    CPQuery<ComponentT, K> encode_with_probe_data(const Float* vec) const {
        return encode_query(vec);
    }

    /**
     * Encode with full sorted indices for advanced multiprobe.
     *
     * For each rotation, returns all indices sorted by absolute value.
     * This allows generating probes ranked by collision probability.
     */
    struct EncodedWithSortedIndices {
        CPCode<ComponentT, K> code;

        // sorted_indices[r][rank] = (index, |value|, is_negative) sorted descending by |value|
        std::array<std::vector<std::tuple<size_t, Float, bool>>, K> sorted_indices;
    };

    EncodedWithSortedIndices encode_with_sorted_indices(const Float* vec) const {
        EncodedWithSortedIndices result;

        for (size_t r = 0; r < K; ++r) {
            // Apply rotation
            rotation_chain_.apply_copy(vec, buffer_.data(), r);

            // Collect all (index, |value|, is_negative) tuples
            auto& sorted = result.sorted_indices[r];
            sorted.clear();
            sorted.reserve(rotation_chain_.padded_dim());

            for (size_t i = 0; i < rotation_chain_.padded_dim(); ++i) {
                sorted.emplace_back(i, std::abs(buffer_[i]), buffer_[i] < 0);
            }

            // Sort by absolute value descending
            std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });

            // Top element is the hash
            size_t max_idx = std::get<0>(sorted[0]);
            bool is_negative = std::get<2>(sorted[0]);
            result.code.components[r] =
                CPCode<ComponentT, K>::encode(max_idx, is_negative);
        }

        return result;
    }

    /**
     * Batch encode multiple vectors.
     *
     * PARALLELIZED with OpenMP when CPHNSW_USE_OPENMP is defined.
     * Uses thread-local buffers to avoid race conditions.
     *
     * @param vecs      Input vectors (row-major, num_vecs x dim)
     * @param num_vecs  Number of vectors
     * @param codes     Output codes (pre-allocated, size num_vecs)
     */
    void encode_batch(const Float* vecs, size_t num_vecs,
                      CPCode<ComponentT, K>* codes) const {
#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel
        {
            // Thread-local buffer to avoid race conditions
            std::vector<Float> local_buffer(rotation_chain_.padded_dim());

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

private:
    size_t dim_;
    RotationChain rotation_chain_;
    mutable std::vector<Float> buffer_;  // Work buffer (thread-local in production)
};

// Common encoder types
using CPEncoder8 = CPEncoder<uint8_t, 16>;    // For d <= 128
using CPEncoder16 = CPEncoder<uint16_t, 16>;  // For d > 128
using CPEncoder32 = CPEncoder<uint8_t, 32>;   // Higher precision

}  // namespace cphnsw
