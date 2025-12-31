#pragma once

#include "../core/types.hpp"
#include <cstddef>

// SIMD headers
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define CPHNSW_HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define CPHNSW_HAS_AVX512 0
#define CPHNSW_HAS_AVX2 1
#include <immintrin.h>
#else
#define CPHNSW_HAS_AVX512 0
#define CPHNSW_HAS_AVX2 0
#endif

// Ensure CPHNSW_HAS_AVX2 is defined when AVX512 is available
#if CPHNSW_HAS_AVX512 && !defined(CPHNSW_HAS_AVX2)
#define CPHNSW_HAS_AVX2 1
#endif

#if defined(__SSE4_2__) || defined(__POPCNT__)
#define CPHNSW_HAS_POPCNT 1
#include <nmmintrin.h>
#else
#define CPHNSW_HAS_POPCNT 0
#endif

namespace cphnsw {

// ============================================================================
// Software Popcount Helpers (for platforms without POPCNT instruction)
// ============================================================================

namespace detail {

/**
 * Software popcount for 16-bit values.
 */
inline uint32_t popcount16_fallback(uint32_t x) {
    x = x & 0xFFFF;
    x = x - ((x >> 1) & 0x5555);
    x = (x & 0x3333) + ((x >> 2) & 0x3333);
    x = (x + (x >> 4)) & 0x0F0F;
    return (x * 0x0101) >> 8;
}

/**
 * Software popcount for 32-bit values.
 */
inline uint32_t popcount32_fallback(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    return (x * 0x01010101) >> 24;
}

/**
 * Software popcount for 64-bit values.
 */
inline uint64_t popcount64_fallback(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (x * 0x0101010101010101ULL) >> 56;
}

}  // namespace detail

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
// AVX-512 Implementations
// ============================================================================

#if CPHNSW_HAS_AVX512

/**
 * AVX-512 Hamming distance for uint8_t components (K = 64).
 *
 * Uses _mm512_cmpeq_epi8_mask for direct mask comparison.
 * Mask has 1 bit per byte match, use popcount64.
 */
inline HammingDist hamming_avx512_u8_64(const uint8_t* a, const uint8_t* b) {
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));

    // Compare bytes for equality: returns 64-bit mask (1 bit per byte)
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(va, vb);

    // Count matches using popcount
#if CPHNSW_HAS_POPCNT
    uint64_t matches = _mm_popcnt_u64(eq_mask);
#else
    uint64_t x = eq_mask;
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    uint64_t matches = (x * 0x0101010101010101ULL) >> 56;
#endif

    return static_cast<HammingDist>(64 - matches);
}

/**
 * AVX-512 Hamming distance for uint8_t components (K = 32).
 *
 * Uses lower 32 bytes of 512-bit registers.
 */
inline HammingDist hamming_avx512_u8_32(const uint8_t* a, const uint8_t* b) {
    // Load 32 bytes into 256-bit portion of 512-bit register
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

    // Promote to 512-bit for mask comparison
    __m512i va512 = _mm512_castsi256_si512(va);
    __m512i vb512 = _mm512_castsi256_si512(vb);

    // Compare first 32 bytes
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(va512, vb512);

    // Only count lower 32 bits of mask
#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(static_cast<uint32_t>(eq_mask & 0xFFFFFFFF));
#else
    uint32_t x = static_cast<uint32_t>(eq_mask & 0xFFFFFFFF);
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    uint32_t matches = (x * 0x01010101) >> 24;
#endif

    return static_cast<HammingDist>(32 - matches);
}

/**
 * AVX-512 Hamming distance for uint8_t components (K = 16).
 *
 * Uses 128-bit portion of 512-bit register.
 */
inline HammingDist hamming_avx512_u8_16(const uint8_t* a, const uint8_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

    // Promote to 512-bit for mask comparison
    __m512i va512 = _mm512_castsi128_si512(va);
    __m512i vb512 = _mm512_castsi128_si512(vb);

    // Compare first 16 bytes
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(va512, vb512);

    // Only count lower 16 bits of mask
#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(static_cast<uint32_t>(eq_mask & 0xFFFF));
#else
    uint32_t x = static_cast<uint32_t>(eq_mask & 0xFFFF);
    x = x - ((x >> 1) & 0x5555);
    x = (x & 0x3333) + ((x >> 2) & 0x3333);
    x = (x + (x >> 4)) & 0x0F0F;
    uint32_t matches = (x * 0x0101) >> 8;
#endif

    return static_cast<HammingDist>(16 - matches);
}

/**
 * AVX-512 Hamming distance for uint16_t components (K = 32).
 */
inline HammingDist hamming_avx512_u16_32(const uint16_t* a, const uint16_t* b) {
    __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
    __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));

    // Compare 16-bit elements: returns 32-bit mask (1 bit per 16-bit element)
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(va, vb);

#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(eq_mask);
#else
    uint32_t x = eq_mask;
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    uint32_t matches = (x * 0x01010101) >> 24;
#endif

    return static_cast<HammingDist>(32 - matches);
}

/**
 * AVX-512 Hamming distance for uint16_t components (K = 16).
 */
inline HammingDist hamming_avx512_u16_16(const uint16_t* a, const uint16_t* b) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

    // Promote to 512-bit for mask comparison
    __m512i va512 = _mm512_castsi256_si512(va);
    __m512i vb512 = _mm512_castsi256_si512(vb);

    // Compare first 16 16-bit elements
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(va512, vb512);

#if CPHNSW_HAS_POPCNT
    uint32_t matches = _mm_popcnt_u32(eq_mask & 0xFFFF);
#else
    uint32_t x = eq_mask & 0xFFFF;
    x = x - ((x >> 1) & 0x5555);
    x = (x & 0x3333) + ((x >> 2) & 0x3333);
    x = (x + (x >> 4)) & 0x0F0F;
    uint32_t matches = (x * 0x0101) >> 8;
#endif

    return static_cast<HammingDist>(16 - matches);
}

#endif  // CPHNSW_HAS_AVX512

// ============================================================================
// Dispatcher Templates
// ============================================================================

/**
 * Hamming distance for uint8_t codes with automatic SIMD dispatch.
 * Prefers AVX-512 > AVX2 > scalar.
 */
template <size_t K>
inline HammingDist hamming_distance(const CPCode<uint8_t, K>& a,
                                     const CPCode<uint8_t, K>& b) {
#if CPHNSW_HAS_AVX512
    if constexpr (K == 64) {
        return hamming_avx512_u8_64(a.components.data(), b.components.data());
    } else if constexpr (K == 32) {
        return hamming_avx512_u8_32(a.components.data(), b.components.data());
    } else if constexpr (K == 16) {
        return hamming_avx512_u8_16(a.components.data(), b.components.data());
    } else if constexpr (K <= 64) {
        // Pad to 64 and use AVX-512
        alignas(64) uint8_t a_padded[64] = {0};
        alignas(64) uint8_t b_padded[64] = {0};
        for (size_t i = 0; i < K; ++i) {
            a_padded[i] = a.components[i];
            b_padded[i] = b.components[i];
        }
        HammingDist full_dist = hamming_avx512_u8_64(a_padded, b_padded);
        return full_dist;
    } else {
        return hamming_scalar_u8<K>(a, b);
    }
#elif CPHNSW_HAS_AVX2
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
 * Prefers AVX-512 > AVX2 > scalar.
 */
template <size_t K>
inline HammingDist hamming_distance(const CPCode<uint16_t, K>& a,
                                     const CPCode<uint16_t, K>& b) {
#if CPHNSW_HAS_AVX512
    if constexpr (K == 32) {
        return hamming_avx512_u16_32(a.components.data(), b.components.data());
    } else if constexpr (K == 16) {
        return hamming_avx512_u16_16(a.components.data(), b.components.data());
    } else if constexpr (K <= 32) {
        // Pad to 32 and use AVX-512
        alignas(64) uint16_t a_padded[32] = {0};
        alignas(64) uint16_t b_padded[32] = {0};
        for (size_t i = 0; i < K; ++i) {
            a_padded[i] = a.components[i];
            b_padded[i] = b.components[i];
        }
        HammingDist full_dist = hamming_avx512_u16_32(a_padded, b_padded);
        return full_dist;
    } else {
        return hamming_scalar_u16<K>(a, b);
    }
#elif CPHNSW_HAS_AVX2
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
// Asymmetric Search Distance (Reconstructed Dot Product)
// ============================================================================

/**
 * Computes asymmetric distance for HNSW navigation.
 *
 * Returns NEGATIVE of reconstructed dot product.
 * Lower value = more similar (compatible with HNSW min-heaps).
 *
 * Formula: distance = -Σᵣ sign_r × query.rotated_vecs[r][node.index_r]
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
 * This is an UNBIASED ESTIMATOR of the cosine similarity!
 *
 * CRITICAL: Returns negative score to convert MIPS to Min-Distance.
 *           This allows HNSW's min-heap navigation to work correctly.
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist asymmetric_search_distance(
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

    // CRITICAL: Negate for min-heap compatibility
    // Higher dot product = lower distance = more similar
    return -score;
}

/**
 * SIMD-optimized asymmetric search distance for K=16, uint8_t.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance<uint8_t, 16>(
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
 * SIMD-optimized asymmetric search distance for K=32, uint8_t.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance<uint8_t, 32>(
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
