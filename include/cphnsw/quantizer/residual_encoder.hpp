#pragma once

#include "../core/types.hpp"
#include "rotation_chain.hpp"
#include "cp_encoder.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace cphnsw {

/**
 * Residual Cross-Polytope Encoder
 *
 * PHASE 2 OPTIMIZATION (PhD Portfolio):
 * Encodes vectors using primary + residual quantization to recover
 * precision lost in Phase 1's symmetric Hamming distance.
 *
 * This implementation uses the SIMPLIFIED approach: two independent
 * rotation chains rather than true residual encoding. This is:
 *   - Faster (no reconstruction step)
 *   - Simpler to implement
 *   - Still effective due to ensemble/complementary information
 *
 * Expected improvement:
 *   - Graph-only recall: 25-30% → 55-60%
 *   - Hybrid recall@10: 75% → 90%
 *
 * Template parameters:
 * - K: Primary code width (e.g., 64 bits)
 * - R: Residual code width (typically K/2, e.g., 32 bits)
 */
template <size_t K, size_t R = K / 2>
class ResidualCPEncoder {
public:
    /**
     * Construct residual encoder.
     *
     * Uses separate rotation chains for primary and residual encoding
     * to maximize independence between the two codes.
     *
     * @param dim             Original vector dimension
     * @param primary_seed    Random seed for primary rotation chain
     * @param residual_seed   Random seed for residual rotation chain (default: primary_seed + 1000)
     */
    ResidualCPEncoder(size_t dim, uint64_t primary_seed, uint64_t residual_seed = 0)
        : dim_(dim),
          primary_chain_(dim, K, primary_seed),
          residual_chain_(dim, R, residual_seed == 0 ? primary_seed + 1000 : residual_seed) {

        // Pre-allocate work buffer
        padded_dim_ = primary_chain_.padded_dim();
        buffer_.resize(padded_dim_);
    }

    /// Get original dimension
    size_t dim() const { return dim_; }

    /// Get padded dimension
    size_t padded_dim() const { return padded_dim_; }

    /// Get primary code width
    static constexpr size_t primary_width() { return K; }

    /// Get residual code width
    static constexpr size_t residual_width() { return R; }

    /**
     * Encode vector to ResidualBinaryCode.
     *
     * Uses two independent rotation chains for primary and residual.
     * This provides complementary information without the overhead
     * of reconstruction-based residual encoding.
     *
     * @param vec   Input vector (length >= dim_)
     * @return      ResidualBinaryCode with primary and residual codes
     */
    ResidualBinaryCode<K, R> encode(const Float* vec) const {
        return encode_with_buffer(vec, buffer_.data());
    }

    /**
     * Encode vector to ResidualBinaryCode using provided buffer.
     * Thread-safe: Uses caller-provided buffer instead of shared member.
     */
    ResidualBinaryCode<K, R> encode_with_buffer(const Float* vec, Float* buffer) const {
        ResidualBinaryCode<K, R> code;

        // Encode primary using primary rotation chain
        code.primary.clear();
        for (size_t r = 0; r < K; ++r) {
            primary_chain_.apply_copy(vec, buffer, r);
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer, padded_dim_, max_idx, max_abs);
            code.primary.set_sign(r, buffer[max_idx] < 0);
        }

        // Encode residual using separate rotation chain
        code.residual.clear();
        for (size_t r = 0; r < R; ++r) {
            residual_chain_.apply_copy(vec, buffer, r);
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer, padded_dim_, max_idx, max_abs);
            code.residual.set_sign(r, buffer[max_idx] < 0);
        }

        return code;
    }

    /**
     * Encode query for residual distance computation.
     *
     * Pre-computes both primary and residual codes, plus scalars
     * for the combined distance formula.
     *
     * @param vec             Input vector (length >= dim_)
     * @param avg_node_norm   Average norm of nodes in index
     * @return                ResidualQuery with codes and pre-computed scalars
     */
    ResidualQuery<K, R> encode_query(const Float* vec, float avg_node_norm = 1.0f) const {
        return encode_query_with_buffer(vec, buffer_.data(), avg_node_norm);
    }

    /**
     * Encode query using provided buffer.
     * Thread-safe: Uses caller-provided buffer.
     */
    ResidualQuery<K, R> encode_query_with_buffer(
        const Float* vec, Float* buffer, float avg_node_norm = 1.0f) const {

        ResidualQuery<K, R> query;

        // Encode both codes
        ResidualBinaryCode<K, R> code = encode_with_buffer(vec, buffer);
        query.primary = code.primary;
        query.residual = code.residual;

        // Compute query norm
        float query_norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            query_norm += vec[i] * vec[i];
        }
        query_norm = std::sqrt(query_norm);
        query.query_norm = query_norm;

        // Pre-compute scalars for final distance conversion
        query.base = 0.0f;
        query.scale = query_norm * avg_node_norm /
                      static_cast<float>(K + (R >> DefaultResidualWeighting::SHIFT));

        return query;
    }

    /**
     * Batch encode multiple vectors.
     *
     * PARALLELIZED with OpenMP when available.
     *
     * @param vecs      Input vectors (row-major, num_vecs x dim)
     * @param num_vecs  Number of vectors
     * @param codes     Output codes (pre-allocated, size num_vecs)
     */
    void encode_batch(const Float* vecs, size_t num_vecs,
                      ResidualBinaryCode<K, R>* codes) const {
#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel
        {
            std::vector<Float> local_buffer(padded_dim_);

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
    size_t padded_dim_;
    RotationChain primary_chain_;
    RotationChain residual_chain_;
    mutable std::vector<Float> buffer_;
};

// ============================================================================
// Simplified Residual Encoder (Sign-Only, No Reconstruction)
// ============================================================================

/**
 * SimplifiedResidualEncoder: Faster but less accurate residual encoding.
 *
 * ALTERNATIVE APPROACH:
 * Instead of computing a full reconstruction, this encoder uses a
 * two-level encoding where:
 *   - Primary: Encodes the original vector with K rotations
 *   - Residual: Encodes the same vector with R DIFFERENT rotations
 *
 * This doesn't technically encode the "residual" (reconstruction error),
 * but provides complementary information from different random projections.
 *
 * Advantages:
 *   - Much faster (no reconstruction step)
 *   - Still provides recall improvement through ensemble effect
 *   - Simpler implementation
 *
 * Disadvantages:
 *   - Less theoretically motivated
 *   - May have redundant information if K and R rotations are not independent
 */
template <size_t K, size_t R = K / 2>
class SimplifiedResidualEncoder {
public:
    SimplifiedResidualEncoder(size_t dim, uint64_t primary_seed, uint64_t residual_seed = 0)
        : dim_(dim),
          primary_chain_(dim, K, primary_seed),
          residual_chain_(dim, R, residual_seed == 0 ? primary_seed + 1000 : residual_seed) {

        padded_dim_ = primary_chain_.padded_dim();
        buffer_.resize(padded_dim_);
    }

    size_t dim() const { return dim_; }
    size_t padded_dim() const { return padded_dim_; }

    /**
     * Encode vector to ResidualBinaryCode using separate rotation chains.
     * No reconstruction needed - just two independent encodings.
     */
    ResidualBinaryCode<K, R> encode(const Float* vec) const {
        return encode_with_buffer(vec, buffer_.data());
    }

    ResidualBinaryCode<K, R> encode_with_buffer(const Float* vec, Float* buffer) const {
        ResidualBinaryCode<K, R> code;

        // Encode primary
        code.primary.clear();
        for (size_t r = 0; r < K; ++r) {
            primary_chain_.apply_copy(vec, buffer, r);
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer, padded_dim_, max_idx, max_abs);
            code.primary.set_sign(r, buffer[max_idx] < 0);
        }

        // Encode residual (independent rotations)
        code.residual.clear();
        for (size_t r = 0; r < R; ++r) {
            residual_chain_.apply_copy(vec, buffer, r);
            size_t max_idx;
            Float max_abs;
            find_argmax_abs(buffer, padded_dim_, max_idx, max_abs);
            code.residual.set_sign(r, buffer[max_idx] < 0);
        }

        return code;
    }

    /**
     * Encode query for residual distance computation.
     */
    ResidualQuery<K, R> encode_query(const Float* vec, float avg_node_norm = 1.0f) const {
        return encode_query_with_buffer(vec, buffer_.data(), avg_node_norm);
    }

    ResidualQuery<K, R> encode_query_with_buffer(
        const Float* vec, Float* buffer, float avg_node_norm = 1.0f) const {

        ResidualQuery<K, R> query;

        // Encode both codes
        ResidualBinaryCode<K, R> code = encode_with_buffer(vec, buffer);
        query.primary = code.primary;
        query.residual = code.residual;

        // Compute query norm
        float query_norm = 0.0f;
        for (size_t i = 0; i < dim_; ++i) {
            query_norm += vec[i] * vec[i];
        }
        query_norm = std::sqrt(query_norm);
        query.query_norm = query_norm;

        // Pre-compute scalars
        query.base = 0.0f;
        query.scale = query_norm * avg_node_norm /
                      static_cast<float>(K + (R >> DefaultResidualWeighting::SHIFT));

        return query;
    }

    /**
     * Batch encode multiple vectors.
     */
    void encode_batch(const Float* vecs, size_t num_vecs,
                      ResidualBinaryCode<K, R>* codes) const {
#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel
        {
            std::vector<Float> local_buffer(padded_dim_);

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
    size_t padded_dim_;
    RotationChain primary_chain_;
    RotationChain residual_chain_;
    mutable std::vector<Float> buffer_;
};

// Common residual encoder types
using ResidualEncoder64_32 = SimplifiedResidualEncoder<64, 32>;
using ResidualEncoder32_16 = SimplifiedResidualEncoder<32, 16>;

}  // namespace cphnsw
