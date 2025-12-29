#pragma once

#include "../core/types.hpp"
#include <cstddef>

// SIMD headers
#if defined(__AVX2__)
#define CPHNSW_HAS_AVX2 1
#include <immintrin.h>
#else
#define CPHNSW_HAS_AVX2 0
#endif

#if defined(__SSE4_2__) || defined(__POPCNT__)
#define CPHNSW_HAS_POPCNT 1
#include <nmmintrin.h>
#else
#define CPHNSW_HAS_POPCNT 0
#endif

namespace cphnsw {

/**
 * Cross-Polytope Hamming Distance
 *
 * Computes the number of mismatching components between two CP codes.
 * Distance = K - (number of matching components)
 *
 * A component matches if and only if BOTH the index AND sign match.
 * This is implemented as byte/short equality comparison, not XOR.
 *
 * CRITICAL: For uint16_t components, _mm256_movemask_epi8 returns
 * 2 bits per match, so we must divide popcount by 2.
 */

// ============================================================================
// Scalar Implementations
// ============================================================================

/**
 * Scalar Hamming distance for uint8_t components.
 */
template <size_t K>
inline HammingDist hamming_scalar_u8(const CPCode<uint8_t, K>& a,
                                      const CPCode<uint8_t, K>& b) {
    HammingDist dist = 0;
    for (size_t i = 0; i < K; ++i) {
        if (a.components[i] != b.components[i]) {
            ++dist;
        }
    }
    return dist;
}

/**
 * Scalar Hamming distance for uint16_t components.
 */
template <size_t K>
inline HammingDist hamming_scalar_u16(const CPCode<uint16_t, K>& a,
                                       const CPCode<uint16_t, K>& b) {
    HammingDist dist = 0;
    for (size_t i = 0; i < K; ++i) {
        if (a.components[i] != b.components[i]) {
            ++dist;
        }
    }
    return dist;
}

// ============================================================================
// AVX2 Implementations
// ============================================================================

#if CPHNSW_HAS_AVX2

/**
 * AVX2 Hamming distance for uint8_t components (K <= 32).
 *
 * Uses _mm256_cmpeq_epi8 for byte comparison.
 * movemask returns 1 bit per byte.
 */
inline HammingDist hamming_avx2_u8_32(const uint8_t* a, const uint8_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

    // Compare bytes for equality: 0xFF if equal, 0x00 if not
    __m256i eq = _mm256_cmpeq_epi8(va, vb);

    // Create bitmask: 1 bit per byte (1 if equal)
    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));

    // Count matches (popcount of mask)
    // Distance = K - matches
#if CPHNSW_HAS_POPCNT
    return static_cast<HammingDist>(32 - _mm_popcnt_u32(mask));
#else
    // Software popcount fallback
    uint32_t count = mask;
    count = count - ((count >> 1) & 0x55555555);
    count = (count & 0x33333333) + ((count >> 2) & 0x33333333);
    count = (count + (count >> 4)) & 0x0F0F0F0F;
    count = (count * 0x01010101) >> 24;
    return static_cast<HammingDist>(32 - count);
#endif
}

/**
 * AVX2 Hamming distance for uint8_t components (K = 16).
 *
 * Uses 128-bit SSE for K=16.
 */
inline HammingDist hamming_avx2_u8_16(const uint8_t* a, const uint8_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

    __m128i eq = _mm_cmpeq_epi8(va, vb);
    uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(eq));

#if CPHNSW_HAS_POPCNT
    return static_cast<HammingDist>(16 - _mm_popcnt_u32(mask));
#else
    uint32_t count = mask & 0xFFFF;
    count = count - ((count >> 1) & 0x5555);
    count = (count & 0x3333) + ((count >> 2) & 0x3333);
    count = (count + (count >> 4)) & 0x0F0F;
    count = (count * 0x0101) >> 8;
    return static_cast<HammingDist>(16 - count);
#endif
}

/**
 * AVX2 Hamming distance for uint16_t components (K <= 16).
 *
 * CRITICAL: cmpeq_epi16 produces 0xFFFF (16 bits) for matches.
 * movemask_epi8 returns 2 bits per 16-bit match.
 * Therefore: matches = popcount(mask) / 2
 */
inline HammingDist hamming_avx2_u16_16(const uint16_t* a, const uint16_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

    // Compare 16-bit elements: 0xFFFF if equal, 0x0000 if not
    __m256i eq = _mm256_cmpeq_epi16(va, vb);

    // movemask_epi8 returns 2 bits per 16-bit match!
    uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));

    // Count matches (each match produces 2 bits in mask)
#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(mask) >> 1;  // Divide by 2!
#else
    uint32_t count = mask;
    count = count - ((count >> 1) & 0x55555555);
    count = (count & 0x33333333) + ((count >> 2) & 0x33333333);
    count = (count + (count >> 4)) & 0x0F0F0F0F;
    count = (count * 0x01010101) >> 24;
    uint32_t matches = count >> 1;  // Divide by 2!
#endif

    return static_cast<HammingDist>(16 - matches);
}

/**
 * AVX2 Hamming distance for uint16_t components (K = 8).
 */
inline HammingDist hamming_avx2_u16_8(const uint16_t* a, const uint16_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

    __m128i eq = _mm_cmpeq_epi16(va, vb);
    uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(eq));

#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(mask) >> 1;
#else
    uint32_t count = mask & 0xFFFF;
    count = count - ((count >> 1) & 0x5555);
    count = (count & 0x3333) + ((count >> 2) & 0x3333);
    count = (count + (count >> 4)) & 0x0F0F;
    count = (count * 0x0101) >> 8;
    uint32_t matches = count >> 1;
#endif

    return static_cast<HammingDist>(8 - matches);
}

#endif  // CPHNSW_HAS_AVX2

// ============================================================================
// Dispatcher Templates
// ============================================================================

/**
 * Hamming distance for uint8_t codes with automatic SIMD dispatch.
 */
template <size_t K>
inline HammingDist hamming_distance(const CPCode<uint8_t, K>& a,
                                     const CPCode<uint8_t, K>& b) {
#if CPHNSW_HAS_AVX2
    if constexpr (K == 16) {
        return hamming_avx2_u8_16(a.components.data(), b.components.data());
    } else if constexpr (K == 32) {
        return hamming_avx2_u8_32(a.components.data(), b.components.data());
    } else if constexpr (K <= 32) {
        // Pad to 32 and use AVX2
        alignas(32) uint8_t a_padded[32] = {0};
        alignas(32) uint8_t b_padded[32] = {0};
        for (size_t i = 0; i < K; ++i) {
            a_padded[i] = a.components[i];
            b_padded[i] = b.components[i];
        }
        HammingDist full_dist = hamming_avx2_u8_32(a_padded, b_padded);
        // Padding zeros match in both arrays, so full_dist = 32 - (actual_matches + padding_matches)
        // = 32 - actual_matches - (32 - K) = K - actual_matches, which is correct
        return full_dist;
    } else {
        return hamming_scalar_u8<K>(a, b);
    }
#else
    return hamming_scalar_u8<K>(a, b);
#endif
}

/**
 * Hamming distance for uint16_t codes with automatic SIMD dispatch.
 */
template <size_t K>
inline HammingDist hamming_distance(const CPCode<uint16_t, K>& a,
                                     const CPCode<uint16_t, K>& b) {
#if CPHNSW_HAS_AVX2
    if constexpr (K == 8) {
        return hamming_avx2_u16_8(a.components.data(), b.components.data());
    } else if constexpr (K == 16) {
        return hamming_avx2_u16_16(a.components.data(), b.components.data());
    } else {
        return hamming_scalar_u16<K>(a, b);
    }
#else
    return hamming_scalar_u16<K>(a, b);
#endif
}

// ============================================================================
// Asymmetric Distance (Soft Hamming)
// ============================================================================

/**
 * Asymmetric Distance: Weighted mismatch using query magnitudes.
 *
 * The key insight: Standard Hamming distance has only K+1 discrete values,
 * creating "flat plateaus" where HNSW's greedy descent cannot navigate.
 *
 * Asymmetric distance restores the gradient by weighting each mismatch
 * by the query's confidence (magnitude) in that component:
 *   - Match: cost 0
 *   - Mismatch: cost = query.magnitudes[i]
 *
 * This creates a continuous distance function where:
 *   - Mismatching on a "strong" feature (high magnitude) is heavily penalized
 *   - Mismatching on a "weak" feature (low magnitude, likely noise) is lightly penalized
 *
 * Storage: Unchanged (index still stores compact codes)
 * Query: Uses CPQuery with pre-computed magnitudes
 * Result: Float distance with fine-grained gradient for navigation
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist asymmetric_distance(
    const CPQuery<ComponentT, K>& query,
    const CPCode<ComponentT, K>& node_code) {

    AsymmetricDist dist = 0.0f;

    for (size_t i = 0; i < K; ++i) {
        // If codes don't match, add magnitude penalty
        if (query.primary_code.components[i] != node_code.components[i]) {
            dist += query.magnitudes[i];
        }
    }

    return dist;
}

/**
 * SIMD-optimized asymmetric distance for K=16, uint8_t components.
 * Falls back to scalar for other configurations.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_distance<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const CPCode<uint8_t, 16>& node_code) {

    // Load codes (16 bytes each)
    __m128i q_codes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(query.primary_code.components.data()));
    __m128i n_codes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(node_code.components.data()));

    // Compare for equality: 0xFF if equal, 0x00 if not
    __m128i eq = _mm_cmpeq_epi8(q_codes, n_codes);

    // Get mismatch mask: bit set where NOT equal
    uint32_t mismatch_mask = static_cast<uint32_t>(~_mm_movemask_epi8(eq)) & 0xFFFF;

    // Accumulate magnitudes for mismatched positions
    AsymmetricDist dist = 0.0f;
    while (mismatch_mask) {
        int idx = __builtin_ctz(mismatch_mask);  // Find lowest set bit
        dist += query.magnitudes[idx];
        mismatch_mask &= (mismatch_mask - 1);    // Clear lowest set bit
    }

    return dist;
}
#endif

/**
 * SIMD-optimized asymmetric distance for K=32, uint8_t components.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_distance<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const CPCode<uint8_t, 32>& node_code) {

    // Load codes (32 bytes each)
    __m256i q_codes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(query.primary_code.components.data()));
    __m256i n_codes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(node_code.components.data()));

    // Compare for equality
    __m256i eq = _mm256_cmpeq_epi8(q_codes, n_codes);

    // Get mismatch mask
    uint32_t mismatch_mask = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq));

    // Accumulate magnitudes for mismatched positions
    AsymmetricDist dist = 0.0f;
    while (mismatch_mask) {
        int idx = __builtin_ctz(mismatch_mask);
        dist += query.magnitudes[idx];
        mismatch_mask &= (mismatch_mask - 1);
    }

    return dist;
}
#endif

// ============================================================================
// Reconstructed Dot Product (Asymmetric Distance V2)
// ============================================================================

/**
 * Estimate dot product using Cross-Polytope codes.
 *
 * THE KEY INSIGHT: The node's code tells us which axis it lies closest to
 * in the rotated space. We use the query's actual value at that axis
 * to estimate the dot product.
 *
 * For each rotation r:
 *   - Node code encodes (index_r, sign_r) meaning "I'm closest to sign_r * e_{index_r}"
 *   - Query has full rotated vector y_r
 *   - Contribution to dot product: sign_r * y_r[index_r]
 *
 * Score = sum_r( sign_r * query.rotated_vecs[r][index_r] )
 *
 * This is an UNBIASED ESTIMATOR of the cosine similarity!
 * Higher score = more similar vectors.
 *
 * For HNSW (which minimizes distance), use negative score.
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist estimate_dot_product(
    const CPQuery<ComponentT, K>& query,
    const CPCode<ComponentT, K>& node_code) {

    float score = 0.0f;

    for (size_t r = 0; r < K; ++r) {
        // Decode the code component
        ComponentT raw = node_code.components[r];
        size_t idx = CPCode<ComponentT, K>::decode_index(raw);
        bool is_negative = CPCode<ComponentT, K>::decode_sign_negative(raw);

        // Look up query's value at this index
        float val = query.rotated_vecs[r][idx];

        // Apply sign and accumulate
        if (is_negative) {
            score -= val;
        } else {
            score += val;
        }
    }

    // Return NEGATIVE score for HNSW (which minimizes distance)
    // Higher dot product = lower distance
    return -score;
}

/**
 * SIMD-optimized dot product estimation for K=16, uint8_t.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist estimate_dot_product<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const CPCode<uint8_t, 16>& node_code) {

    float score = 0.0f;

    // Process all 16 rotations
    for (size_t r = 0; r < 16; ++r) {
        uint8_t raw = node_code.components[r];
        size_t idx = raw >> 1;           // Upper 7 bits = index
        bool is_negative = raw & 1;      // LSB = sign

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;
}
#endif

/**
 * SIMD-optimized dot product estimation for K=32, uint8_t.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist estimate_dot_product<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const CPCode<uint8_t, 32>& node_code) {

    float score = 0.0f;

    // Process all 32 rotations
    for (size_t r = 0; r < 32; ++r) {
        uint8_t raw = node_code.components[r];
        size_t idx = raw >> 1;
        bool is_negative = raw & 1;

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;
}
#endif

}  // namespace cphnsw
