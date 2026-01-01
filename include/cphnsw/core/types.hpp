#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace cphnsw {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr size_t CACHE_LINE_SIZE = 64;

// ============================================================================
// Basic Type Definitions
// ============================================================================

/// Node identifier type (supports up to 4B nodes)
using NodeId = uint32_t;
constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();

// NOTE: LayerLevel removed for NSW flatten (single-layer graph)

/// Hamming distance type (discrete, max distance = K, typically < 64)
using HammingDist = uint16_t;

/// Asymmetric distance type (continuous, for search gradient)
using AsymmetricDist = float;

/// Float type for vectors
using Float = float;

// ============================================================================
// Cross-Polytope Code Structures
// ============================================================================

/**
 * CPCode: Compact Cross-Polytope hash code stored in the index.
 *
 * RABITQ-STYLE MAGNITUDE AUGMENTATION (SIGMOD 2024):
 * Standard CP-LSH discards magnitude information, treating all projections
 * as equal. This causes massive variance when vectors have different
 * "confidence" levels in their projections.
 *
 * Fix: Store quantized magnitude (0-255) for each rotation.
 * - Storage: 2K bytes per node (K components + K magnitudes)
 * - Distance: Score = Σ(sign × query_val × node_mag)
 *
 * Template parameters:
 * - ComponentT: uint8_t for d <= 128, uint16_t for d > 128
 *   Each component encodes: (argmax_index << 1) | sign_bit
 * - K: Number of independent rotations (code width)
 *
 * Memory: K * sizeof(ComponentT) + K bytes per node
 * Example: K=32, uint8_t => 64 bytes per node
 */
template <typename ComponentT, size_t K>
struct CPCode {
    std::array<ComponentT, K> components;

    /// Quantized magnitudes for RaBitQ-style correction
    /// Stored as uint8_t (0-255) to minimize storage
    /// 0 = magnitude 0, 255 = max magnitude seen during encoding
    std::array<uint8_t, K> magnitudes;

    /// Decode index from component (upper bits)
    static constexpr size_t decode_index(ComponentT c) {
        return c >> 1;
    }

    /// Decode sign from component (lowest bit: 0=positive, 1=negative)
    static constexpr bool decode_sign_negative(ComponentT c) {
        return c & 1;
    }

    /// Maximum encodable index for this component type
    /// For uint8_t: 7 bits for index = 127 max
    /// For uint16_t: 15 bits for index = 32767 max
    static constexpr size_t max_encodable_index() {
        return (size_t(1) << (sizeof(ComponentT) * 8 - 1)) - 1;
    }

    /// Encode index and sign into component
    /// @throws std::overflow_error if index exceeds max_encodable_index()
    static constexpr ComponentT encode(size_t index, bool is_negative) {
        // Compile-time check for constexpr contexts, runtime check otherwise
        if (index > max_encodable_index()) {
            // In constexpr context, this will cause a compile error
            // At runtime, we assert and potentially throw
#ifndef NDEBUG
            assert(false && "CPCode::encode: index overflow - use larger ComponentT");
#endif
            // For release builds, clamp to prevent silent corruption
            index = max_encodable_index();
        }
        return static_cast<ComponentT>((index << 1) | (is_negative ? 1 : 0));
    }

    /// Safe encode with explicit bounds checking (throws on overflow)
    static ComponentT encode_checked(size_t index, bool is_negative) {
        if (index > max_encodable_index()) {
            throw std::overflow_error(
                "CPCode::encode: index " + std::to_string(index) +
                " exceeds max " + std::to_string(max_encodable_index()) +
                " for ComponentT size " + std::to_string(sizeof(ComponentT)));
        }
        return static_cast<ComponentT>((index << 1) | (is_negative ? 1 : 0));
    }

    /// Quantize magnitude to uint8_t
    /// Input: raw magnitude (typically 0 to max_magnitude)
    /// Scale factor should be pre-computed as 255 / expected_max
    static constexpr uint8_t quantize_magnitude(float mag, float scale = 255.0f) {
        float scaled = mag * scale;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        return static_cast<uint8_t>(scaled);
    }

    /// Dequantize magnitude back to float
    static constexpr float dequantize_magnitude(uint8_t qmag, float scale = 1.0f / 255.0f) {
        return static_cast<float>(qmag) * scale;
    }
};

// Common type aliases
using CPCode8 = CPCode<uint8_t, 16>;    // For d <= 128 (SIFT-1M, GloVe-100)
using CPCode16 = CPCode<uint16_t, 16>;  // For d > 128 (GIST-1M)
using CPCode32 = CPCode<uint8_t, 32>;   // Higher precision variant
using CPCode64 = CPCode<uint8_t, 64>;   // Highest precision variant

/**
 * CPQuery: Query-time structure for asymmetric distance computation.
 * NOT stored in the index - only used during search.
 *
 * CRITICAL: Stores full rotated vectors for dot product reconstruction.
 * The Cross-Polytope code tells us which axis a node lies on; we use
 * the query's value at that axis to estimate the dot product.
 *
 * Memory: K * padded_dim * sizeof(float) per query
 * For K=32, dim=128: 32 * 128 * 4 = 16KB per query (acceptable)
 */
template <typename ComponentT, size_t K>
struct CPQuery {
    CPCode<ComponentT, K> primary_code;

    /// Full rotated vectors for dot product reconstruction
    /// rotated_vecs[r][i] = value at index i after rotation r
    std::array<std::vector<Float>, K> rotated_vecs;

    /// Magnitude of winning coordinate for each rotation (for multiprobe ranking)
    std::array<Float, K> magnitudes;

    /// Original argmax indices before encoding (for multiprobe flips)
    std::array<uint32_t, K> original_indices;
};

// ============================================================================
// Search Result
// ============================================================================

/**
 * SearchResult: A single result from k-NN search.
 * Uses AsymmetricDist (float) for continuous gradient during search.
 */
struct SearchResult {
    NodeId id;
    AsymmetricDist distance;  // Continuous distance for proper gradient descent

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }

    bool operator==(const SearchResult& other) const {
        return id == other.id && distance == other.distance;
    }
};

// ============================================================================
// Index Parameters
// ============================================================================

/**
 * CPHNSWParams: Configuration parameters for the index.
 *
 * Default values optimized for HIGH RECALL use cases (SIFT-like data).
 * For faster builds with lower recall, reduce k, M, and ef_construction.
 */
struct CPHNSWParams {
    /// Original vector dimension
    size_t dim = 0;

    /// Padded dimension (next power of 2 >= dim)
    size_t padded_dim = 0;

    /// Number of rotations (code width K) - higher = more precision
    size_t k = 32;

    /// Max connections per node (NSW: single layer, no M_max0 distinction)
    size_t M = 32;

    /// Search width during construction - deeper search for hybrid edges
    size_t ef_construction = 200;

    /// Keep pruned connections for robustness (essential for connectivity)
    bool keep_pruned = true;

    /// Random seed for reproducibility
    uint64_t seed = 42;

    /// Number of random entry points for search (NSW: replaces hierarchy)
    size_t k_entry = 4;

    /// Detour threshold for rank-based pruning (CAGRA)
    float rank_pruning_alpha = 1.1f;

    /// Compute derived parameters and validate bounds
    void finalize() {
        // ========================================
        // Parameter Validation
        // ========================================

        // Dimension must be positive
        if (dim == 0) {
            throw std::invalid_argument("CPHNSWParams: dim must be > 0");
        }

        // Code width (k) must be positive and reasonable
        if (k == 0) {
            throw std::invalid_argument("CPHNSWParams: k (code width) must be > 0");
        }
        if (k > 256) {
            throw std::invalid_argument("CPHNSWParams: k (code width) must be <= 256");
        }

        // Max connections (M) must be positive and fit in Flash layout
        if (M == 0) {
            throw std::invalid_argument("CPHNSWParams: M (max connections) must be > 0");
        }
        constexpr size_t FLASH_MAX_M_LIMIT = 64;  // From flat_graph.hpp
        if (M > FLASH_MAX_M_LIMIT) {
            throw std::invalid_argument(
                "CPHNSWParams: M must be <= 64 (FLASH_MAX_M limit)");
        }

        // ef_construction should be >= M for good graph quality
        if (ef_construction < M) {
            throw std::invalid_argument(
                "CPHNSWParams: ef_construction must be >= M for good graph quality");
        }
        if (ef_construction > 10000) {
            throw std::invalid_argument(
                "CPHNSWParams: ef_construction must be <= 10000");
        }

        // k_entry must be positive (NSW random entry points)
        if (k_entry == 0) {
            throw std::invalid_argument("CPHNSWParams: k_entry must be > 0");
        }
        if (k_entry > 100) {
            throw std::invalid_argument("CPHNSWParams: k_entry must be <= 100");
        }

        // Pruning alpha must be >= 1.0 (triangle inequality threshold)
        if (rank_pruning_alpha < 1.0f) {
            throw std::invalid_argument(
                "CPHNSWParams: rank_pruning_alpha must be >= 1.0");
        }
        if (rank_pruning_alpha > 3.0f) {
            throw std::invalid_argument(
                "CPHNSWParams: rank_pruning_alpha must be <= 3.0");
        }

        // ========================================
        // Derived Parameters
        // ========================================

        // Pad dimension to next power of 2
        padded_dim = 1;
        while (padded_dim < dim) {
            padded_dim *= 2;
        }
        // NOTE: m_L and M_max0 removed for NSW flatten
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if n is a power of 2
constexpr bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/// Round up to next power of 2
constexpr size_t next_power_of_two(size_t n) {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

}  // namespace cphnsw
