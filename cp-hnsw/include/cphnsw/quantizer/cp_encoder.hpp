#pragma once

#include "../core/types.hpp"
#include "rotation_chain.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace cphnsw {

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
        CPCode<ComponentT, K> code;

        for (size_t r = 0; r < K; ++r) {
            // Apply rotation to copy
            rotation_chain_.apply_copy(vec, buffer_.data(), r);

            // Find argmax |buffer[i]|
            size_t max_idx = 0;
            Float max_abs = 0.0f;

            for (size_t i = 0; i < rotation_chain_.padded_dim(); ++i) {
                Float abs_val = std::abs(buffer_[i]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                    max_idx = i;
                }
            }

            // Encode: (index << 1) | sign_bit
            // sign_bit = 1 if negative, 0 if positive
            bool is_negative = (buffer_[max_idx] < 0);
            code.components[r] = CPCode<ComponentT, K>::encode(max_idx, is_negative);
        }

        return code;
    }

    /**
     * Encode vector with additional probe data for multiprobe search.
     *
     * Returns the primary code plus magnitudes and indices needed
     * to generate alternative probe sequences.
     *
     * @param vec   Input vector (length >= dim_)
     * @return      CPQuery with code and multiprobe data
     */
    CPQuery<ComponentT, K> encode_with_probe_data(const Float* vec) const {
        CPQuery<ComponentT, K> query;

        for (size_t r = 0; r < K; ++r) {
            // Apply rotation
            rotation_chain_.apply_copy(vec, buffer_.data(), r);

            // Find argmax |buffer[i]| and store magnitude
            size_t max_idx = 0;
            Float max_abs = 0.0f;

            for (size_t i = 0; i < rotation_chain_.padded_dim(); ++i) {
                Float abs_val = std::abs(buffer_[i]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                    max_idx = i;
                }
            }

            // Store primary code component
            bool is_negative = (buffer_[max_idx] < 0);
            query.primary_code.components[r] =
                CPCode<ComponentT, K>::encode(max_idx, is_negative);

            // Store magnitude for probability ranking
            query.magnitudes[r] = max_abs;
            query.original_indices[r] = static_cast<uint32_t>(max_idx);
        }

        return query;
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
     * @param vecs      Input vectors (row-major, num_vecs x dim)
     * @param num_vecs  Number of vectors
     * @param codes     Output codes (pre-allocated, size num_vecs)
     */
    void encode_batch(const Float* vecs, size_t num_vecs,
                      CPCode<ComponentT, K>* codes) const {
        for (size_t i = 0; i < num_vecs; ++i) {
            codes[i] = encode(vecs + i * dim_);
        }
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
