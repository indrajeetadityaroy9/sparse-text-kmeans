#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cphnsw {

// ============================================================================
// Configuration Constants
// ============================================================================

#ifndef CPHNSW_CACHE_LINE_SIZE_DEFINED
#define CPHNSW_CACHE_LINE_SIZE_DEFINED
constexpr size_t CACHE_LINE_SIZE = 64;
#endif

// ============================================================================
// Basic Type Definitions
// ============================================================================

#ifndef CPHNSW_NODE_TYPES_DEFINED
#define CPHNSW_NODE_TYPES_DEFINED
/// Node identifier type (supports up to 4B nodes)
using NodeId = uint32_t;
/// Invalid node ID sentinel
constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();
#endif

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

    /// Encode index and sign into component
    static constexpr ComponentT encode(size_t index, bool is_negative) {
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

// ============================================================================
// RaBitQ-Style Binary Code Structures (Phase 1 Optimization)
// ============================================================================

/**
 * BinaryCode: Packed sign bits for XOR + PopCount distance computation.
 *
 * RABITQ OPTIMIZATION (PhD Portfolio):
 * Replace expensive SIMD Gather instructions (10-11 cycles) with pure
 * bitwise operations: XOR + PopCount (1-2 cycles total).
 *
 * Memory layout: K sign bits packed into uint64_t words.
 * - K=64: 1 word (8 bytes)
 * - K=32: 1 word (4 bytes used, padded to 8)
 *
 * CRITICAL: This structure must be 64-byte aligned for AVX-512 loads.
 */
template <size_t K>
struct alignas(64) BinaryCode {
    static constexpr size_t NUM_WORDS = (K + 63) / 64;

    /// Packed sign bits: bit i = 1 if rotation i has negative sign
    uint64_t signs[NUM_WORDS];

    /// Clear all bits
    void clear() {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            signs[i] = 0;
        }
    }

    /// Set sign bit for rotation r
    void set_sign(size_t r, bool is_negative) {
        if (is_negative) {
            signs[r / 64] |= (1ULL << (r % 64));
        }
    }

    /// Get sign bit for rotation r
    bool get_sign(size_t r) const {
        return (signs[r / 64] >> (r % 64)) & 1;
    }
};

/**
 * RaBitQQuery: Query structure for RaBitQ-style distance computation.
 *
 * CRITICAL INSIGHT (Avoiding Hidden Gather Trap):
 * Pre-compute C1, C2 scalars ONCE per query. The distance formula becomes:
 *   Dist = C1 + C2 * Hamming(query, node)
 *
 * This eliminates per-neighbor magnitude lookups inside the hot loop.
 *
 * ASSUMPTION: Normalized vectors (SIFT-1M, GloVe, etc.)
 * For unnormalized data, see thesis defense statement in plan.
 */
template <size_t K>
struct RaBitQQuery {
    /// Binary code (packed sign bits)
    BinaryCode<K> binary;

    /// Pre-computed scalar: related to query norm and average node norm
    /// Dist = c1 + c2 * hamming_distance
    float c1;

    /// Pre-computed scaling factor
    float c2;

    /// Original query norm (for reranking if needed)
    float query_norm;
};

// Common BinaryCode type aliases
using BinaryCode32 = BinaryCode<32>;
using BinaryCode64 = BinaryCode<64>;
using RaBitQQuery32 = RaBitQQuery<32>;
using RaBitQQuery64 = RaBitQQuery<64>;

// ============================================================================
// Residual Quantization Structures (Phase 2 Optimization)
// ============================================================================

/**
 * ResidualBinaryCode: Primary + Residual codes for improved precision.
 *
 * PHASE 2 OPTIMIZATION (PhD Portfolio):
 * Phase 1's symmetric Hamming distance causes recall drop (~10-15%).
 * Residual quantization recovers this by encoding the reconstruction error.
 *
 * Algorithm:
 *   1. Encode primary code from original vector
 *   2. Approximate reconstruction from primary code
 *   3. Compute residual = original - reconstructed
 *   4. Encode residual code from residual vector
 *
 * Distance formula (integer-only until final conversion):
 *   Combined = (primary_hamming << Shift) + residual_hamming
 *
 * Template parameters:
 * - K: Primary code width (e.g., 64 bits)
 * - R: Residual code width (typically K/2, e.g., 32 bits)
 *
 * Memory: (K + R + 63) / 64 * 8 bytes per node
 * Example: K=64, R=32 => 16 bytes per node
 */
template <size_t K, size_t R = K / 2>
struct ResidualBinaryCode {
    static constexpr size_t PRIMARY_WORDS = (K + 63) / 64;
    static constexpr size_t RESIDUAL_WORDS = (R + 63) / 64;

    /// Primary binary code (K sign bits)
    BinaryCode<K> primary;

    /// Residual binary code (R sign bits)
    BinaryCode<R> residual;

    /// Clear all bits
    void clear() {
        primary.clear();
        residual.clear();
    }
};

/**
 * ResidualQuery: Query structure for residual distance computation.
 *
 * Contains both primary and residual binary codes, plus pre-computed
 * scalars for the combined distance formula.
 *
 * CRITICAL: Uses integer-only distance computation until final conversion.
 * This avoids expensive float conversions in the hot loop.
 */
template <size_t K, size_t R = K / 2>
struct ResidualQuery {
    /// Primary binary code
    BinaryCode<K> primary;

    /// Residual binary code
    BinaryCode<R> residual;

    /// Pre-computed scalar for final distance conversion
    /// Final distance = base + scale * combined_hamming
    float base;
    float scale;

    /// Query norm (for reranking if needed)
    float query_norm;
};

/**
 * Bit-shift weighting template for residual distance.
 *
 * TUNABLE WEIGHTING (PhD Portfolio):
 * The relative importance of primary vs residual is controlled by bit shift.
 *
 * | Shift | Alpha (Primary) | Beta (Residual) | Ratio |
 * |-------|-----------------|-----------------|-------|
 * |   1   |      2/3        |      1/3        |  2:1  |
 * |   2   |      4/5        |      1/5        |  4:1  |
 * |   3   |      8/9        |      1/9        |  8:1  |
 * |   4   |     16/17       |     1/17        | 16:1  |
 *
 * Recommendation by K/R ratio:
 * - K=64, R=32: Shift=2 (4:1)
 * - K=64, R=16: Shift=3 (8:1)
 * - K=32, R=32: Shift=1 (2:1)
 * - K=32, R=16: Shift=2 (4:1)
 *
 * PERFORMANCE: vpsll (bit shift) is 1 cycle, tuning is free.
 */
template <int Shift = 2>
struct ResidualWeighting {
    static constexpr int SHIFT = Shift;

    /// Combine primary and residual Hamming distances (integer-only)
    static constexpr uint32_t combine(uint32_t primary, uint32_t residual) {
        return (primary << Shift) + residual;
    }

    /// Effective weight for primary distance
    static constexpr float primary_weight() {
        return static_cast<float>(1 << Shift) / static_cast<float>((1 << Shift) + 1);
    }

    /// Effective weight for residual distance
    static constexpr float residual_weight() {
        return 1.0f / static_cast<float>((1 << Shift) + 1);
    }
};

// Common residual type aliases
using ResidualBinaryCode64_32 = ResidualBinaryCode<64, 32>;
using ResidualBinaryCode32_16 = ResidualBinaryCode<32, 16>;
using ResidualQuery64_32 = ResidualQuery<64, 32>;
using ResidualQuery32_16 = ResidualQuery<32, 16>;

// Default weighting (Shift=2 for 4:1 primary:residual ratio)
using DefaultResidualWeighting = ResidualWeighting<2>;

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
