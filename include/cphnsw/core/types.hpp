#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>

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

/// Layer level type (max 255 layers, typically < 16)
using LayerLevel = uint8_t;

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
 * Template parameters:
 * - ComponentT: uint8_t for d <= 128, uint16_t for d > 128
 *   Each component encodes: (argmax_index << 1) | sign_bit
 * - K: Number of independent rotations (code width)
 *
 * Memory: K * sizeof(ComponentT) bytes per node
 * Example: K=16, uint8_t => 16 bytes per node
 */
template <typename ComponentT, size_t K>
struct CPCode {
    std::array<ComponentT, K> components;

    /// Decode index from component (upper bits)
    static constexpr size_t decode_index(ComponentT c) {
        return c >> 1;
    }

    /// Decode sign from component (lowest bit: 0=positive, 1=negative)
    static constexpr bool decode_sign_negative(ComponentT c) {
        return c & 1;
    }

    /// Encode index and sign into component
    static constexpr ComponentT encode(size_t index, bool is_negative) {
        return static_cast<ComponentT>((index << 1) | (is_negative ? 1 : 0));
    }
};

// Common type aliases
using CPCode8 = CPCode<uint8_t, 16>;    // For d <= 128 (SIFT-1M, GloVe-100)
using CPCode16 = CPCode<uint16_t, 16>;  // For d > 128 (GIST-1M)
using CPCode32 = CPCode<uint8_t, 32>;   // Higher precision variant

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

    /// Max connections per node at layers > 0 - higher = more robust graph
    size_t M = 32;

    /// Max connections at layer 0 (typically 2*M)
    size_t M_max0 = 64;

    /// Search width during construction - deeper search for hybrid edges
    size_t ef_construction = 200;

    /// Level multiplier: 1/ln(M)
    double m_L = 0.0;

    /// Keep pruned connections for robustness (essential for connectivity)
    bool keep_pruned = true;

    /// Random seed for reproducibility
    uint64_t seed = 42;

    /// Compute derived parameters
    void finalize() {
        // Pad dimension to next power of 2
        padded_dim = 1;
        while (padded_dim < dim) {
            padded_dim *= 2;
        }

        // Compute level multiplier if not set
        if (m_L <= 0.0) {
            m_L = 1.0 / std::log(static_cast<double>(M));
        }

        // Default M_max0
        if (M_max0 == 0) {
            M_max0 = 2 * M;
        }
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
