#pragma once

#include "../core/types.hpp"
#include "quantizer_policy.hpp"
#include "cp_encoder.hpp"
#include "../distance/hamming.hpp"
#include "../graph/flat_graph.hpp"  // For FLASH_MAX_M
#include <memory>

namespace cphnsw {

/**
 * CPFHTPolicy: Cross-Polytope FHT Quantizer as a QuantizerPolicy.
 *
 * Wraps the existing CPEncoder to implement the QuantizerPolicy interface.
 * This enables the same search_layer and insert algorithms to work with
 * both FHT-based and learned (A-ITQ) quantizers.
 *
 * Key difference from A-ITQ:
 * - Uses random rotations (FHT) instead of learned rotations
 * - Stores (index, sign) components instead of binary codes
 * - Includes RaBitQ-style magnitude augmentation
 */
template <typename ComponentT, size_t K>
class CPFHTPolicy : public QuantizerPolicy<CPCode<ComponentT, K>, CPQuery<ComponentT, K>, K> {
public:
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

    /**
     * Construct CP-FHT quantizer.
     *
     * @param dim   Vector dimension
     * @param seed  Random seed for rotation matrices
     */
    CPFHTPolicy(size_t dim, uint64_t seed = 42)
        : encoder_(std::make_shared<Encoder>(dim, seed)) {}

    /**
     * Construct from existing encoder.
     */
    explicit CPFHTPolicy(std::shared_ptr<Encoder> encoder)
        : encoder_(std::move(encoder)) {}

    size_t dim() const override { return encoder_->dim(); }
    size_t padded_dim() const override { return encoder_->padded_dim(); }

    /// Get the underlying encoder
    const Encoder& encoder() const { return *encoder_; }
    Encoder& encoder() { return *encoder_; }

    Code encode(const Float* vec) const override {
        return encoder_->encode(vec);
    }

    Query encode_query(const Float* vec) const override {
        return encoder_->encode_query(vec);
    }

    Code encode_with_buffer(const Float* vec, Float* buffer) const override {
        return encoder_->encode_with_buffer(vec, buffer);
    }

    Query encode_query_with_buffer(const Float* vec, Float* buffer) const override {
        return encoder_->encode_query_with_buffer(vec, buffer);
    }

    /**
     * Compute search distance using asymmetric reconstructed dot product.
     *
     * This is the SAME function used in the current search_layer.
     * Wrapping it in the policy ensures metric consistency.
     */
    float search_distance(const Query& query, const Code& code) const override {
        return asymmetric_search_distance(query, code);
    }

    /**
     * Batch compute distances using SoA transposed layout.
     *
     * Delegates to the existing SIMD-optimized batch functions.
     */
    void batch_search_distance_soa(
        const Query& query,
        const void* codes_transposed_ptr,
        size_t num_neighbors,
        float* out_distances) const override {

        // Cast to the correct transposed layout type
        const ComponentT (*codes_transposed)[FLASH_MAX_M] =
            reinterpret_cast<const ComponentT (*)[FLASH_MAX_M]>(codes_transposed_ptr);

        asymmetric_search_distance_batch_soa<ComponentT, K>(
            query, codes_transposed, num_neighbors, out_distances);
    }

private:
    std::shared_ptr<Encoder> encoder_;
};

// Common type aliases
using CPFHTPolicy8 = CPFHTPolicy<uint8_t, 16>;    // d <= 128
using CPFHTPolicy16 = CPFHTPolicy<uint16_t, 16>;  // d > 128
using CPFHTPolicy32 = CPFHTPolicy<uint8_t, 32>;   // Higher precision

}  // namespace cphnsw
