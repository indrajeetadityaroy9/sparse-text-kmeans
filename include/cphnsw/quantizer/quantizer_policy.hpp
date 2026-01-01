#pragma once

#include "../core/types.hpp"
#include <vector>
#include <cstddef>

namespace cphnsw {

/**
 * QuantizerPolicy: Abstract interface for vector quantization strategies.
 *
 * This policy abstraction enables pluggable quantizers (FHT-based CP-LSH, A-ITQ, etc.)
 * while ensuring the SAME metric is used for both construction and search.
 *
 * CRITICAL INSIGHT (Metric Mismatch Fix):
 * The graph topology must be optimized for the search metric. If you build with
 * FHT distance but search with A-ITQ distance, the graph edges won't align with
 * the search gradient, causing massive recall degradation.
 *
 * Each quantizer policy must provide:
 * - Code/Query types for storage and search
 * - encode(): Vector -> Code (stored in index)
 * - encode_query(): Vector -> Query (used during search)
 * - search_distance(): Query x Code -> float (for graph navigation)
 * - batch_search_distance(): Query x Codes[] -> float[] (SIMD optimized)
 *
 * Template parameters:
 * - CodeT: Compact code type stored in index
 * - QueryT: Query-time structure (may contain more data for asymmetric distance)
 * - K: Code width (number of components/bits)
 */
template <typename CodeT, typename QueryT, size_t K>
class QuantizerPolicy {
public:
    using Code = CodeT;
    using Query = QueryT;
    static constexpr size_t CodeWidth = K;

    virtual ~QuantizerPolicy() = default;

    /// Get original vector dimension
    virtual size_t dim() const = 0;

    /// Get padded dimension (for SIMD alignment)
    virtual size_t padded_dim() const = 0;

    /// Encode vector to compact code (stored in index)
    virtual Code encode(const Float* vec) const = 0;

    /// Encode vector to query structure (used during search)
    virtual Query encode_query(const Float* vec) const = 0;

    /// Thread-safe encode with caller-provided buffer
    virtual Code encode_with_buffer(const Float* vec, Float* buffer) const = 0;

    /// Thread-safe query encode with caller-provided buffer
    virtual Query encode_query_with_buffer(const Float* vec, Float* buffer) const = 0;

    /**
     * Compute search distance between query and code.
     *
     * CRITICAL: This function defines the "metric landscape" that the graph
     * navigates. It MUST be the same function used during construction.
     *
     * Returns: Distance (lower = more similar). Typically -dot_product.
     */
    virtual float search_distance(const Query& query, const Code& code) const = 0;

    /**
     * Batch compute search distances for SIMD optimization.
     *
     * Uses SoA (Struct-of-Arrays) transposed layout for optimal SIMD throughput:
     * codes_transposed[component_k][neighbor_n] is contiguous.
     *
     * @param query              Query structure
     * @param codes_transposed   Transposed codes [K][max_neighbors]
     * @param num_neighbors      Number of neighbors to process
     * @param out_distances      Output array (must have num_neighbors elements)
     */
    virtual void batch_search_distance_soa(
        const Query& query,
        const void* codes_transposed,
        size_t num_neighbors,
        float* out_distances) const = 0;
};

/**
 * QuantizerType: Enum for selecting quantizer at runtime.
 */
enum class QuantizerType {
    CP_FHT,      // Cross-Polytope with Fast Hadamard Transform (random projections)
    AITQ,        // Asymmetric ITQ (learned projections)
    RABITQ       // RaBitQ with magnitude augmentation
};

}  // namespace cphnsw
