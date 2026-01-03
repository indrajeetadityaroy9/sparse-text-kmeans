#pragma once

#include "../../core/codes.hpp"
#include <cstdint>

namespace cphnsw {
namespace detail {

// ============================================================================
// Software Popcount (Portable Fallback)
// ============================================================================

inline uint32_t popcount64_software(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<uint32_t>((x * 0x0101010101010101ULL) >> 56);
}

// Use hardware popcount if available
inline uint32_t popcount64(uint64_t x) {
#if defined(__POPCNT__) || defined(__SSE4_2__)
    return static_cast<uint32_t>(__builtin_popcountll(x));
#else
    return popcount64_software(x);
#endif
}

// ============================================================================
// Scalar Hamming Distance: Primary Only (Phase 1 / R=0)
// ============================================================================

/**
 * Compute Hamming distance between two primary codes.
 * Returns popcount(a XOR b).
 */
template <size_t K>
inline uint32_t hamming_primary_scalar(
    const BinaryCodeStorage<K>& a,
    const BinaryCodeStorage<K>& b) {

    uint32_t dist = 0;
    for (size_t w = 0; w < BinaryCodeStorage<K>::NUM_WORDS; ++w) {
        dist += popcount64(a.signs[w] ^ b.signs[w]);
    }
    return dist;
}

// ============================================================================
// Scalar Combined Distance: Primary + Residual (Phase 2 / R>0)
// ============================================================================

/**
 * Compute combined Hamming distance with bit-shift weighting.
 * Returns (primary_hamming << Shift) + residual_hamming.
 */
template <size_t K, size_t R, int Shift>
inline uint32_t hamming_combined_scalar(
    const ResidualCode<K, R>& a,
    const ResidualCode<K, R>& b) {

    // Primary Hamming distance
    uint32_t prim_dist = hamming_primary_scalar<K>(a.primary, b.primary);

    if constexpr (R == 0) {
        // No residual: just return primary
        return prim_dist;
    } else {
        // Residual Hamming distance
        uint32_t res_dist = hamming_primary_scalar<R>(a.residual, b.residual);

        // Combine with bit-shift weighting
        return (prim_dist << Shift) + res_dist;
    }
}

// ============================================================================
// Scalar Distance Computation (Full)
// ============================================================================

/**
 * Compute full distance: base + scale * hamming.
 */
template <size_t K, size_t R, int Shift>
inline DistanceType distance_scalar(
    const CodeQuery<K, R, Shift>& query,
    const ResidualCode<K, R>& code) {

    uint32_t hamming = hamming_combined_scalar<K, R, Shift>(query.code, code);
    return query.base + query.scale * static_cast<float>(hamming);
}

// ============================================================================
// Scalar Batch Distance (SoA Layout)
// ============================================================================

/**
 * Scalar batch distance computation from SoA layout.
 * This is the reference implementation; SIMD versions override this.
 */
template <size_t K, size_t R, int Shift>
void distance_batch_scalar(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    constexpr size_t PRIMARY_WORDS = ResidualCode<K, R>::PRIMARY_WORDS;
    constexpr size_t RESIDUAL_WORDS = ResidualCode<K, R>::RESIDUAL_WORDS;

    for (size_t n = 0; n < count; ++n) {
        // Compute primary Hamming
        uint32_t prim_dist = 0;
        for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
            uint64_t xor_result = query.code.primary.signs[w] ^
                                   soa_layout.primary_transposed[w][n];
            prim_dist += popcount64(xor_result);
        }

        uint32_t combined;
        if constexpr (R == 0) {
            combined = prim_dist;
        } else {
            // Compute residual Hamming
            uint32_t res_dist = 0;
            for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
                uint64_t xor_result = query.code.residual.signs[w] ^
                                       soa_layout.residual_transposed[w][n];
                res_dist += popcount64(xor_result);
            }
            combined = (prim_dist << Shift) + res_dist;
        }

        out_distances[n] = query.base + query.scale * static_cast<float>(combined);
    }
}

/**
 * Scalar batch Hamming computation (returns raw integers).
 */
template <size_t K, size_t R, int Shift>
void hamming_batch_scalar(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    constexpr size_t PRIMARY_WORDS = ResidualCode<K, R>::PRIMARY_WORDS;
    constexpr size_t RESIDUAL_WORDS = ResidualCode<K, R>::RESIDUAL_WORDS;

    for (size_t n = 0; n < count; ++n) {
        // Compute primary Hamming
        uint32_t prim_dist = 0;
        for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
            uint64_t xor_result = query.code.primary.signs[w] ^
                                   soa_layout.primary_transposed[w][n];
            prim_dist += popcount64(xor_result);
        }

        if constexpr (R == 0) {
            out_hamming[n] = prim_dist;
        } else {
            // Compute residual Hamming
            uint32_t res_dist = 0;
            for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
                uint64_t xor_result = query.code.residual.signs[w] ^
                                       soa_layout.residual_transposed[w][n];
                res_dist += popcount64(xor_result);
            }
            out_hamming[n] = (prim_dist << Shift) + res_dist;
        }
    }
}

}  // namespace detail

// ============================================================================
// UnifiedMetricPolicy Implementation (Scalar Fallback)
// ============================================================================

// Default implementations use scalar kernels
// These are overridden by SIMD implementations when available

template <size_t K, size_t R, int Shift>
inline DistanceType UnifiedMetricPolicy<K, R, Shift>::compute_distance(
    const QueryType& query,
    const CodeType& code) {

    return detail::distance_scalar<K, R, Shift>(query, code);
}

template <size_t K, size_t R, int Shift>
inline uint32_t UnifiedMetricPolicy<K, R, Shift>::compute_hamming(
    const CodeType& code_a,
    const CodeType& code_b) {

    return detail::hamming_combined_scalar<K, R, Shift>(code_a, code_b);
}

template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_distance_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    detail::distance_batch_scalar<K, R, Shift>(query, soa_layout, count, out_distances);
}

template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_hamming_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    detail::hamming_batch_scalar<K, R, Shift>(query, soa_layout, count, out_hamming);
}

}  // namespace cphnsw
