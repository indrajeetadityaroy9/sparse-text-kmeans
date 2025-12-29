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

}  // namespace cphnsw
