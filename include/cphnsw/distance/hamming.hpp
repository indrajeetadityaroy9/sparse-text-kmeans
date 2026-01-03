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
 * Uses the query's rotated vectors and the node's CP code to estimate
 * the dot product. This is more accurate than symmetric Hamming distance
 * because it uses continuous query values rather than discretized codes.
 *
 * Formula: distance = -Σᵣ sign_r × query.rotated_vecs[r][idx_r]
 *
 * The node's CP code encodes: argmax_r |rotated_n[r]| and its sign.
 * We look up the query's value at that same index and weight by sign.
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
 * Scalar specialization of asymmetric search distance for K=16, uint8_t.
 *
 * NOTE: This is a SCALAR implementation with inline decode optimization.
 * True SIMD vectorization of the score accumulation is not feasible here
 * because each rotation requires a different index lookup into rotated_vecs.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const CPCode<uint8_t, 16>& node_code) {

    float score = 0.0f;

    // Process all 16 rotations (scalar - index lookups prevent SIMD)
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
 * Scalar specialization of asymmetric search distance for K=32, uint8_t.
 *
 * NOTE: This is a SCALAR implementation with inline decode optimization.
 * True SIMD vectorization of the score accumulation is not feasible here
 * because each rotation requires a different index lookup into rotated_vecs.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const CPCode<uint8_t, 32>& node_code) {

    float score = 0.0f;

    // Process all 32 rotations (scalar - index lookups prevent SIMD)
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

/**
 * AVX-512 specialization of asymmetric search distance for K=64, uint8_t.
 * Processes 64 rotations with loop unrolling for better ILP.
 */
#if CPHNSW_HAS_AVX512
template <>
inline AsymmetricDist asymmetric_search_distance<uint8_t, 64>(
    const CPQuery<uint8_t, 64>& query,
    const CPCode<uint8_t, 64>& node_code) {

    float score = 0.0f;

    // Process all 64 rotations with 4x unrolling for better ILP
    for (size_t r = 0; r < 64; r += 4) {
        for (size_t i = 0; i < 4; ++i) {
            uint8_t raw = node_code.components[r + i];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;
            float val = query.rotated_vecs[r + i][idx];
            score += is_negative ? -val : val;
        }
    }

    return -score;
}
#endif

// ============================================================================
// Flash Layout: SIMD Batch Distance Computation (SoA Transposed)
// ============================================================================

/**
 * SIMD Batch Asymmetric Search Distance using SoA Transposed Layout.
 *
 * This is the KEY OPTIMIZATION that enables true SIMD parallelism for
 * asymmetric distance computation. By transposing the code layout to SoA,
 * we can process multiple neighbors in parallel.
 *
 * Algorithm (for 8 neighbors at once):
 *   scores[0..7] = 0
 *   for k in 0..K-1:
 *       raw_bytes[0..7] = block.codes_transposed[k][0..7]  // CONTIGUOUS load!
 *       indices[0..7] = raw_bytes >> 1
 *       signs[0..7] = raw_bytes & 1
 *       vals[0..7] = gather(query.rotated_vecs[k], indices)
 *       scores[0..7] += signs ? -vals : vals
 *   distances[0..7] = -scores
 *
 * @param query              Query with rotated vectors
 * @param codes_transposed   Pointer to transposed codes [K][num_neighbors]
 * @param num_neighbors      Number of neighbors to process
 * @param out_distances      Output array (must have num_neighbors elements)
 */

// Forward declaration for NeighborBlock template
template <typename ComponentT, size_t K> struct NeighborBlock;

/**
 * Scalar fallback for batch distance with SoA layout.
 */
template <typename ComponentT, size_t K>
inline void asymmetric_search_distance_batch_soa(
    const CPQuery<ComponentT, K>& query,
    const ComponentT codes_transposed[K][64],  // K rows, up to 64 neighbors
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    for (size_t n = 0; n < num_neighbors; ++n) {
        float score = 0.0f;

        for (size_t k = 0; k < K; ++k) {
            ComponentT raw = codes_transposed[k][n];
            size_t idx = CPCode<ComponentT, K>::decode_index(raw);
            bool is_negative = CPCode<ComponentT, K>::decode_sign_negative(raw);

            float val = query.rotated_vecs[k][idx];
            score += is_negative ? -val : val;
        }

        out_distances[n] = -score;
    }
}

// ============================================================================
// AVX2 SIMD Batch Implementation (8 neighbors at once)
// ============================================================================

#if CPHNSW_HAS_AVX2

/**
 * AVX2 SIMD batch distance for K=32, uint8_t - processes 8 neighbors at once.
 *
 * Uses _mm256_i32gather_ps for parallel lookups into rotated_vecs.
 * Expected speedup: 3-5x over scalar.
 */
inline void asymmetric_search_distance_batch_soa_avx2_8(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t codes_transposed[32][64],
    size_t start_idx,
    float* out_distances) {  // Output for 8 distances

    // Accumulator for 8 scores
    __m256 scores = _mm256_setzero_ps();

    // Process all 32 rotations
    for (size_t k = 0; k < 32; ++k) {
        // 1. Load 8 bytes contiguously: codes_transposed[k][start_idx..start_idx+7]
        __m128i raw_bytes_128 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&codes_transposed[k][start_idx]));

        // 2. Zero-extend 8 bytes to 8 x 32-bit integers
        __m256i raw_32 = _mm256_cvtepu8_epi32(raw_bytes_128);

        // 3. Extract indices (raw >> 1) and signs (raw & 1)
        __m256i indices = _mm256_srli_epi32(raw_32, 1);
        __m256i signs = _mm256_and_si256(raw_32, _mm256_set1_epi32(1));

        // 4. Gather values from query.rotated_vecs[k]
        // Base address for this rotation's vector
        const float* base = query.rotated_vecs[k].data();
        __m256 vals = _mm256_i32gather_ps(base, indices, 4);  // scale = 4 (sizeof float)

        // 5. Apply signs: if sign bit is 1, negate the value
        // Create mask: -1.0f where sign=1, +1.0f where sign=0
        __m256 sign_mask = _mm256_castsi256_ps(
            _mm256_slli_epi32(signs, 31));  // Move sign bit to MSB position
        // XOR with vals to flip sign where needed (IEEE 754 sign bit flip)
        vals = _mm256_xor_ps(vals, sign_mask);

        // 6. Accumulate
        scores = _mm256_add_ps(scores, vals);
    }

    // 7. Negate scores for min-heap compatibility and store
    __m256 neg_scores = _mm256_xor_ps(scores,
        _mm256_set1_ps(-0.0f));  // Flip sign bit
    _mm256_storeu_ps(out_distances, neg_scores);
}

/**
 * AVX2 batch wrapper that processes neighbors in groups of 8.
 */
template <>
inline void asymmetric_search_distance_batch_soa<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t codes_transposed[32][64],
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    size_t n = 0;

    // Process 8 neighbors at a time
    for (; n + 8 <= num_neighbors; n += 8) {
        asymmetric_search_distance_batch_soa_avx2_8(
            query, codes_transposed, n, out_distances + n);
    }

    // Handle remaining neighbors (scalar fallback)
    for (; n < num_neighbors; ++n) {
        float score = 0.0f;
        for (size_t k = 0; k < 32; ++k) {
            uint8_t raw = codes_transposed[k][n];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;
            float val = query.rotated_vecs[k][idx];
            score += is_negative ? -val : val;
        }
        out_distances[n] = -score;
    }
}

/**
 * K=16 version
 */
inline void asymmetric_search_distance_batch_soa_avx2_8_k16(
    const CPQuery<uint8_t, 16>& query,
    const uint8_t codes_transposed[16][64],
    size_t start_idx,
    float* out_distances) {

    __m256 scores = _mm256_setzero_ps();

    for (size_t k = 0; k < 16; ++k) {
        __m128i raw_bytes_128 = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&codes_transposed[k][start_idx]));
        __m256i raw_32 = _mm256_cvtepu8_epi32(raw_bytes_128);
        __m256i indices = _mm256_srli_epi32(raw_32, 1);
        __m256i signs = _mm256_and_si256(raw_32, _mm256_set1_epi32(1));

        const float* base = query.rotated_vecs[k].data();
        __m256 vals = _mm256_i32gather_ps(base, indices, 4);

        __m256 sign_mask = _mm256_castsi256_ps(_mm256_slli_epi32(signs, 31));
        vals = _mm256_xor_ps(vals, sign_mask);
        scores = _mm256_add_ps(scores, vals);
    }

    __m256 neg_scores = _mm256_xor_ps(scores, _mm256_set1_ps(-0.0f));
    _mm256_storeu_ps(out_distances, neg_scores);
}

template <>
inline void asymmetric_search_distance_batch_soa<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const uint8_t codes_transposed[16][64],
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    size_t n = 0;
    for (; n + 8 <= num_neighbors; n += 8) {
        asymmetric_search_distance_batch_soa_avx2_8_k16(
            query, codes_transposed, n, out_distances + n);
    }
    for (; n < num_neighbors; ++n) {
        float score = 0.0f;
        for (size_t k = 0; k < 16; ++k) {
            uint8_t raw = codes_transposed[k][n];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;
            float val = query.rotated_vecs[k][idx];
            score += is_negative ? -val : val;
        }
        out_distances[n] = -score;
    }
}

#endif  // CPHNSW_HAS_AVX2

// ============================================================================
// AVX-512 SIMD Batch Implementation (16 neighbors at once)
// ============================================================================

#if CPHNSW_HAS_AVX512

/**
 * AVX-512 SIMD batch distance for K=32 - processes 16 neighbors at once.
 *
 * Uses _mm512_i32gather_ps for parallel lookups.
 * Expected speedup: 5-8x over scalar.
 */
inline void asymmetric_search_distance_batch_soa_avx512_16(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t codes_transposed[32][64],
    size_t start_idx,
    float* out_distances) {

    __m512 scores = _mm512_setzero_ps();

    for (size_t k = 0; k < 32; ++k) {
        // Load 16 bytes contiguously
        __m128i raw_bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&codes_transposed[k][start_idx]));

        // Zero-extend 16 bytes to 16 x 32-bit integers
        __m512i raw_32 = _mm512_cvtepu8_epi32(raw_bytes);

        // Extract indices and signs
        __m512i indices = _mm512_srli_epi32(raw_32, 1);
        __m512i signs = _mm512_and_si512(raw_32, _mm512_set1_epi32(1));

        // Gather values
        const float* base = query.rotated_vecs[k].data();
        __m512 vals = _mm512_i32gather_ps(indices, base, 4);

        // Apply signs
        __m512 sign_mask = _mm512_castsi512_ps(_mm512_slli_epi32(signs, 31));
        vals = _mm512_xor_ps(vals, sign_mask);

        scores = _mm512_add_ps(scores, vals);
    }

    // Negate and store
    __m512 neg_scores = _mm512_xor_ps(scores, _mm512_set1_ps(-0.0f));
    _mm512_storeu_ps(out_distances, neg_scores);
}

/**
 * AVX-512 batch wrapper for K=32.
 */
inline void asymmetric_search_distance_batch_soa_avx512(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t codes_transposed[32][64],
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    size_t n = 0;

    // Process 16 neighbors at a time
    for (; n + 16 <= num_neighbors; n += 16) {
        asymmetric_search_distance_batch_soa_avx512_16(
            query, codes_transposed, n, out_distances + n);
    }

    // Handle remaining with AVX2 (8 at a time)
#if CPHNSW_HAS_AVX2
    for (; n + 8 <= num_neighbors; n += 8) {
        asymmetric_search_distance_batch_soa_avx2_8(
            query, codes_transposed, n, out_distances + n);
    }
#endif

    // Scalar remainder
    for (; n < num_neighbors; ++n) {
        float score = 0.0f;
        for (size_t k = 0; k < 32; ++k) {
            uint8_t raw = codes_transposed[k][n];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;
            float val = query.rotated_vecs[k][idx];
            score += is_negative ? -val : val;
        }
        out_distances[n] = -score;
    }
}

/**
 * AVX-512 SIMD batch distance for K=64 - processes 16 neighbors at once.
 * Uses _mm512_i32gather_ps for parallel lookups into rotated_vecs.
 */
inline void asymmetric_search_distance_batch_soa_avx512_16_k64(
    const CPQuery<uint8_t, 64>& query,
    const uint8_t codes_transposed[64][64],
    size_t start_idx,
    float* out_distances) {

    __m512 scores = _mm512_setzero_ps();

    for (size_t k = 0; k < 64; ++k) {
        // Load 16 bytes contiguously
        __m128i raw_bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&codes_transposed[k][start_idx]));

        // Zero-extend 16 bytes to 16 x 32-bit integers
        __m512i raw_32 = _mm512_cvtepu8_epi32(raw_bytes);

        // Extract indices and signs
        __m512i indices = _mm512_srli_epi32(raw_32, 1);
        __m512i signs = _mm512_and_si512(raw_32, _mm512_set1_epi32(1));

        // Gather values
        const float* base = query.rotated_vecs[k].data();
        __m512 vals = _mm512_i32gather_ps(indices, base, 4);

        // Apply signs
        __m512 sign_mask = _mm512_castsi512_ps(_mm512_slli_epi32(signs, 31));
        vals = _mm512_xor_ps(vals, sign_mask);

        scores = _mm512_add_ps(scores, vals);
    }

    // Negate and store
    __m512 neg_scores = _mm512_xor_ps(scores, _mm512_set1_ps(-0.0f));
    _mm512_storeu_ps(out_distances, neg_scores);
}

/**
 * AVX-512 batch wrapper for K=64.
 */
inline void asymmetric_search_distance_batch_soa_avx512_k64(
    const CPQuery<uint8_t, 64>& query,
    const uint8_t codes_transposed[64][64],
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    size_t n = 0;

    // Process 16 neighbors at a time
    for (; n + 16 <= num_neighbors; n += 16) {
        asymmetric_search_distance_batch_soa_avx512_16_k64(
            query, codes_transposed, n, out_distances + n);
    }

    // Scalar remainder
    for (; n < num_neighbors; ++n) {
        float score = 0.0f;
        for (size_t k = 0; k < 64; ++k) {
            uint8_t raw = codes_transposed[k][n];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;
            float val = query.rotated_vecs[k][idx];
            score += is_negative ? -val : val;
        }
        out_distances[n] = -score;
    }
}

#endif  // CPHNSW_HAS_AVX512

// ============================================================================
// Legacy AoS Batch API (for backward compatibility)
// ============================================================================

/**
 * Legacy batch distance with AoS layout (DEPRECATED).
 * Kept for backward compatibility. New code should use SoA batch functions.
 */
template <typename ComponentT, size_t K>
inline void asymmetric_search_distance_batch(
    const CPQuery<ComponentT, K>& query,
    const ComponentT* neighbor_codes,  // AoS: [N0_code][N1_code]...
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    for (size_t n = 0; n < num_neighbors; ++n) {
        float score = 0.0f;
        const ComponentT* code = neighbor_codes + n * K;

        for (size_t r = 0; r < K; ++r) {
            ComponentT raw = code[r];
            size_t idx = CPCode<ComponentT, K>::decode_index(raw);
            bool is_negative = CPCode<ComponentT, K>::decode_sign_negative(raw);

            float val = query.rotated_vecs[r][idx];
            score += is_negative ? -val : val;
        }

        out_distances[n] = -score;
    }
}

/**
 * Scalar specialization of legacy batch distance for K=32.
 */
#if CPHNSW_HAS_AVX2
template <>
inline void asymmetric_search_distance_batch<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t* neighbor_codes,
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    // Process each neighbor (scalar - index lookups prevent SIMD)
    for (size_t n = 0; n < num_neighbors; ++n) {
        float score = 0.0f;
        const uint8_t* code = neighbor_codes + n * 32;

        // Process all 32 rotations
        for (size_t r = 0; r < 32; ++r) {
            uint8_t raw = code[r];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;

            float val = query.rotated_vecs[r][idx];
            score += is_negative ? -val : val;
        }

        out_distances[n] = -score;
    }
}
#endif

/**
 * Scalar specialization of batch distance for K=16, uint8_t components.
 *
 * NOTE: This is a SCALAR implementation. See K=32 version for rationale.
 */
#if CPHNSW_HAS_AVX2
template <>
inline void asymmetric_search_distance_batch<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const uint8_t* neighbor_codes,
    size_t num_neighbors,
    AsymmetricDist* out_distances) {

    // Process each neighbor (scalar - index lookups prevent SIMD)
    for (size_t n = 0; n < num_neighbors; ++n) {
        float score = 0.0f;
        const uint8_t* code = neighbor_codes + n * 16;

        for (size_t r = 0; r < 16; ++r) {
            uint8_t raw = code[r];
            size_t idx = raw >> 1;
            bool is_negative = raw & 1;

            float val = query.rotated_vecs[r][idx];
            score += is_negative ? -val : val;
        }

        out_distances[n] = -score;
    }
}
#endif

/**
 * Compute asymmetric search distance from raw code pointer.
 *
 * FLASH CONVENIENCE: Allows computing distance directly from
 * the contiguous code storage in NeighborBlock without creating
 * a CPCode object.
 *
 * @param query  Query with rotated vectors
 * @param code   Pointer to K components of neighbor code
 * @return       Asymmetric distance (negative dot product)
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist asymmetric_search_distance_ptr(
    const CPQuery<ComponentT, K>& query,
    const ComponentT* code) {

    float score = 0.0f;

    for (size_t r = 0; r < K; ++r) {
        ComponentT raw = code[r];
        size_t idx = CPCode<ComponentT, K>::decode_index(raw);
        bool is_negative = CPCode<ComponentT, K>::decode_sign_negative(raw);

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;
}

/**
 * Scalar specialization for K=32, uint8_t.
 *
 * NOTE: This is a SCALAR implementation. See asymmetric_search_distance
 * for rationale on why SIMD is not applicable here.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance_ptr<uint8_t, 32>(
    const CPQuery<uint8_t, 32>& query,
    const uint8_t* code) {

    float score = 0.0f;

    // Scalar loop - index lookups prevent SIMD vectorization
    for (size_t r = 0; r < 32; ++r) {
        uint8_t raw = code[r];
        size_t idx = raw >> 1;
        bool is_negative = raw & 1;

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;
}
#endif

/**
 * Scalar specialization for K=16, uint8_t.
 *
 * NOTE: This is a SCALAR implementation. See asymmetric_search_distance
 * for rationale on why SIMD is not applicable here.
 */
#if CPHNSW_HAS_AVX2
template <>
inline AsymmetricDist asymmetric_search_distance_ptr<uint8_t, 16>(
    const CPQuery<uint8_t, 16>& query,
    const uint8_t* code) {

    float score = 0.0f;

    // Scalar loop - index lookups prevent SIMD vectorization
    for (size_t r = 0; r < 16; ++r) {
        uint8_t raw = code[r];
        size_t idx = raw >> 1;
        bool is_negative = raw & 1;

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;
}
#endif

// ============================================================================
// RaBitQ-Style Binary Hamming Distance (Phase 1 Optimization)
// ============================================================================

/**
 * RABITQ DISTANCE COMPUTATION (PhD Portfolio)
 *
 * Replaces expensive SIMD Gather instructions with pure bitwise operations:
 * - Current: _mm256_i32gather_ps = 10-11 cycles per instruction
 * - New: XOR + PopCount = 1-2 cycles total
 *
 * Distance formula (for normalized vectors):
 *   Dist = C1 + C2 * Hamming(query, node)
 *
 * Where C1, C2 are pre-computed ONCE per query (no per-neighbor lookups).
 */

// ============================================================================
// Scalar Reference Implementation (Development Step 1)
// ============================================================================

/**
 * Scalar Hamming distance between two BinaryCode structures.
 * Uses software popcount - portable but slow.
 * Used for correctness verification before SIMD optimization.
 */
template <size_t K>
inline uint32_t rabitq_hamming_scalar(const BinaryCode<K>& a,
                                       const BinaryCode<K>& b) {
    uint32_t dist = 0;
    for (size_t i = 0; i < BinaryCode<K>::NUM_WORDS; ++i) {
        uint64_t xor_result = a.signs[i] ^ b.signs[i];
#if CPHNSW_HAS_POPCNT
        dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
#else
        dist += static_cast<uint32_t>(detail::popcount64_fallback(xor_result));
#endif
    }
    return dist;
}

/**
 * Scalar RaBitQ distance using pre-computed scalars.
 * Dist = c1 + c2 * hamming_distance
 *
 * This is the CORRECT approach - no per-neighbor magnitude lookups!
 */
template <size_t K>
inline AsymmetricDist rabitq_distance_scalar(const RaBitQQuery<K>& query,
                                              const BinaryCode<K>& node) {
    uint32_t hamming = rabitq_hamming_scalar(query.binary, node);
    return query.c1 + query.c2 * static_cast<float>(hamming);
}

// ============================================================================
// AVX-512 Optimized Implementation
// ============================================================================

#if CPHNSW_HAS_AVX512

/**
 * Harley-Seal software popcount for AVX-512 (Skylake-X fallback).
 * Used when VPOPCNTDQ extension is not available.
 * Expected speedup: 2x over Gather (vs 3x with native VPOPCNTDQ).
 */
#ifndef __AVX512VPOPCNTDQ__
inline __m512i harley_seal_popcnt_epi64(__m512i v) {
    // Split into 256-bit halves
    __m256i lo = _mm512_extracti64x4_epi64(v, 0);
    __m256i hi = _mm512_extracti64x4_epi64(v, 1);

    // Use scalar popcount for each 64-bit word
    alignas(64) uint64_t arr[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + 4), hi);

    for (int i = 0; i < 8; ++i) {
        arr[i] = static_cast<uint64_t>(__builtin_popcountll(arr[i]));
    }

    return _mm512_load_si512(arr);
}
#endif

/**
 * BATCH KERNEL: Process 8 neighbors in parallel with VERTICAL ACCUMULATION.
 *
 * CRITICAL: Do NOT reduce after each neighbor - kills instruction throughput!
 * Horizontal reductions are slow (3-6 cycles) and break the pipeline.
 *
 * Memory layout requirement: SoA transposed
 *   signs_transposed[word][neighbor] allows loading word 0 for all 8 neighbors
 *   with a single _mm512_loadu_si512 instruction.
 *
 * @param query_signs       Query binary code (K/64 words)
 * @param signs_transposed  SoA layout [NUM_WORDS][64] for up to 64 neighbors
 * @param num_words         Number of 64-bit words per code (K/64)
 * @param neighbor_offset   Start index in the block (0, 8, 16, ...)
 * @param out_distances     Output array for 8 Hamming distances
 */
template <size_t K>
inline void rabitq_hamming_batch8_avx512(
    const uint64_t* query_signs,
    const uint64_t signs_transposed[][64],
    size_t neighbor_offset,
    uint32_t* out_distances) {

    constexpr size_t NUM_WORDS = (K + 63) / 64;

    // Accumulator for 8 neighbors (stays in register across all K words)
    __m512i sums = _mm512_setzero_si512();

    // VERTICAL ACCUMULATION: Loop through K words WITHOUT reducing
    for (size_t w = 0; w < NUM_WORDS; ++w) {
        // 1. Broadcast query word (same for all 8 neighbors)
        __m512i q = _mm512_set1_epi64(static_cast<long long>(query_signs[w]));

        // 2. Load 8 neighbors' word w (contiguous due to SoA layout)
        __m512i n = _mm512_loadu_si512(&signs_transposed[w][neighbor_offset]);

        // 3. XOR to find differing bits
        __m512i xor_result = _mm512_xor_si512(q, n);

        // 4. PopCount (use native or Harley-Seal fallback)
#ifdef __AVX512VPOPCNTDQ__
        __m512i popc = _mm512_popcnt_epi64(xor_result);
#else
        __m512i popc = harley_seal_popcnt_epi64(xor_result);
#endif

        // 5. Accumulate (no reduction yet!)
        sums = _mm512_add_epi64(sums, popc);
    }

    // ONLY NOW extract 8 distances (single horizontal operation at end)
    alignas(64) uint64_t results[8];
    _mm512_storeu_si512(results, sums);

    for (int i = 0; i < 8; ++i) {
        out_distances[i] = static_cast<uint32_t>(results[i]);
    }
}

/**
 * Process all neighbors in a block using batch8 kernel.
 * Handles the full 64-neighbor block with 8 iterations.
 */
template <size_t K>
inline void rabitq_hamming_block_avx512(
    const RaBitQQuery<K>& query,
    const uint64_t signs_transposed[][64],
    size_t neighbor_count,
    AsymmetricDist* out_distances) {

    // Process in batches of 8
    alignas(64) uint32_t batch_hamming[8];

    for (size_t offset = 0; offset < neighbor_count; offset += 8) {
        size_t batch_size = std::min(size_t(8), neighbor_count - offset);

        rabitq_hamming_batch8_avx512<K>(
            query.binary.signs,
            signs_transposed,
            offset,
            batch_hamming);

        // Convert Hamming to final distance using pre-computed scalars
        for (size_t i = 0; i < batch_size; ++i) {
            out_distances[offset + i] =
                query.c1 + query.c2 * static_cast<float>(batch_hamming[i]);
        }
    }
}

#endif  // CPHNSW_HAS_AVX512

// ============================================================================
// AVX2 Fallback Implementation (for dev laptops)
// ============================================================================

#if CPHNSW_HAS_AVX2 && !CPHNSW_HAS_AVX512

/**
 * AVX2 batch kernel - processes 4 neighbors at a time.
 * Used on machines without AVX-512 support.
 */
template <size_t K>
inline void rabitq_hamming_batch4_avx2(
    const uint64_t* query_signs,
    const uint64_t signs_transposed[][64],
    size_t neighbor_offset,
    uint32_t* out_distances) {

    constexpr size_t NUM_WORDS = (K + 63) / 64;

    // Accumulator for 4 neighbors
    __m256i sums = _mm256_setzero_si256();

    for (size_t w = 0; w < NUM_WORDS; ++w) {
        // Broadcast query word
        __m256i q = _mm256_set1_epi64x(static_cast<long long>(query_signs[w]));

        // Load 4 neighbors' word w
        __m256i n = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&signs_transposed[w][neighbor_offset]));

        // XOR
        __m256i xor_result = _mm256_xor_si256(q, n);

        // Software popcount per 64-bit word
        alignas(32) uint64_t arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), xor_result);

        alignas(32) uint64_t popc_arr[4];
        for (int i = 0; i < 4; ++i) {
            popc_arr[i] = static_cast<uint64_t>(__builtin_popcountll(arr[i]));
        }

        __m256i popc = _mm256_load_si256(reinterpret_cast<const __m256i*>(popc_arr));
        sums = _mm256_add_epi64(sums, popc);
    }

    // Extract results
    alignas(32) uint64_t results[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), sums);

    for (int i = 0; i < 4; ++i) {
        out_distances[i] = static_cast<uint32_t>(results[i]);
    }
}

#endif  // CPHNSW_HAS_AVX2 && !CPHNSW_HAS_AVX512

// ============================================================================
// Residual Quantization Distance (Phase 2 Optimization)
// ============================================================================

/**
 * RESIDUAL DISTANCE COMPUTATION (PhD Portfolio)
 *
 * Phase 2 optimization: Recover precision lost in Phase 1's symmetric
 * Hamming distance by adding residual quantization.
 *
 * Key insight: Keep everything in INTEGER registers until final conversion.
 * Float conversions are expensive (~5 cycles each).
 *
 * Combined distance formula (integer-only):
 *   combined = (primary_hamming << Shift) + residual_hamming
 *
 * Where Shift controls the relative weighting:
 *   - Shift=1: alpha=2/3, beta=1/3 (2:1)
 *   - Shift=2: alpha=4/5, beta=1/5 (4:1) [DEFAULT]
 *   - Shift=3: alpha=8/9, beta=1/9 (8:1)
 *   - Shift=4: alpha=16/17, beta=1/17 (16:1)
 *
 * TUNING: vpsll (bit shift) is 1 cycle - tuning is essentially free.
 */

// ============================================================================
// Scalar Reference Implementation (for correctness verification)
// ============================================================================

/**
 * Scalar residual Hamming distance.
 * Computes combined distance using bit-shift weighting.
 *
 * @param q_prim  Query primary code (K bits)
 * @param q_res   Query residual code (R bits)
 * @param n_prim  Node primary code (K bits)
 * @param n_res   Node residual code (R bits)
 * @param Shift   Bit-shift for weighting (template parameter)
 * @return        Combined distance as uint32_t
 */
template <size_t K, size_t R, int Shift = 2>
inline uint32_t residual_distance_integer_scalar(
    const BinaryCode<K>& q_prim, const BinaryCode<R>& q_res,
    const BinaryCode<K>& n_prim, const BinaryCode<R>& n_res) {

    // Compute primary Hamming distance
    uint32_t prim_dist = 0;
    for (size_t i = 0; i < BinaryCode<K>::NUM_WORDS; ++i) {
        uint64_t xor_result = q_prim.signs[i] ^ n_prim.signs[i];
#if CPHNSW_HAS_POPCNT
        prim_dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
#else
        prim_dist += static_cast<uint32_t>(detail::popcount64_fallback(xor_result));
#endif
    }

    // Compute residual Hamming distance
    uint32_t res_dist = 0;
    for (size_t i = 0; i < BinaryCode<R>::NUM_WORDS; ++i) {
        uint64_t xor_result = q_res.signs[i] ^ n_res.signs[i];
#if CPHNSW_HAS_POPCNT
        res_dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
#else
        res_dist += static_cast<uint32_t>(detail::popcount64_fallback(xor_result));
#endif
    }

    // Combine with bit-shift weighting (integer-only!)
    return (prim_dist << Shift) + res_dist;
}

/**
 * Scalar residual distance using ResidualQuery structure.
 * Converts to float only at the very end.
 */
template <size_t K, size_t R, int Shift = 2>
inline AsymmetricDist residual_distance_scalar(
    const ResidualQuery<K, R>& query,
    const ResidualBinaryCode<K, R>& node) {

    uint32_t combined = residual_distance_integer_scalar<K, R, Shift>(
        query.primary, query.residual,
        node.primary, node.residual);

    // Convert to float ONCE at the end
    return query.base + query.scale * static_cast<float>(combined);
}

// ============================================================================
// AVX-512 Optimized Residual Distance
// ============================================================================

#if CPHNSW_HAS_AVX512

/**
 * AVX-512 residual distance for single query-node pair.
 *
 * Computes both primary and residual Hamming in parallel,
 * then combines with bit-shift weighting.
 *
 * NOTE: For K=64, R=32, this uses 1+1=2 words, fits in registers.
 */
template <size_t K, size_t R, int Shift = 2>
inline uint32_t residual_distance_integer_avx512(
    const BinaryCode<K>& q_prim, const BinaryCode<R>& q_res,
    const BinaryCode<K>& n_prim, const BinaryCode<R>& n_res) {

    constexpr size_t PRIM_WORDS = BinaryCode<K>::NUM_WORDS;
    constexpr size_t RES_WORDS = BinaryCode<R>::NUM_WORDS;

    uint32_t prim_dist = 0;
    uint32_t res_dist = 0;

    // Process primary code words
    for (size_t w = 0; w < PRIM_WORDS; ++w) {
        uint64_t xor_result = q_prim.signs[w] ^ n_prim.signs[w];
#if defined(__AVX512VPOPCNTDQ__)
        // Use AVX-512 VPOPCNTDQ if available
        __m128i v = _mm_set_epi64x(0, static_cast<long long>(xor_result));
        __m128i popc = _mm_popcnt_epi64(v);
        prim_dist += static_cast<uint32_t>(_mm_extract_epi64(popc, 0));
#elif CPHNSW_HAS_POPCNT
        prim_dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
#else
        prim_dist += static_cast<uint32_t>(detail::popcount64_fallback(xor_result));
#endif
    }

    // Process residual code words
    for (size_t w = 0; w < RES_WORDS; ++w) {
        uint64_t xor_result = q_res.signs[w] ^ n_res.signs[w];
#if defined(__AVX512VPOPCNTDQ__)
        __m128i v = _mm_set_epi64x(0, static_cast<long long>(xor_result));
        __m128i popc = _mm_popcnt_epi64(v);
        res_dist += static_cast<uint32_t>(_mm_extract_epi64(popc, 0));
#elif CPHNSW_HAS_POPCNT
        res_dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
#else
        res_dist += static_cast<uint32_t>(detail::popcount64_fallback(xor_result));
#endif
    }

    // Combine with bit-shift weighting
    return (prim_dist << Shift) + res_dist;
}

/**
 * AVX-512 batch residual distance - processes 8 neighbors at once.
 *
 * Uses SoA transposed layout for both primary and residual codes.
 * Computes combined distance for 8 neighbors in parallel.
 *
 * Memory layout requirement:
 *   primary_transposed[PRIM_WORDS][64] - primary signs in SoA
 *   residual_transposed[RES_WORDS][64] - residual signs in SoA
 *
 * @param query             ResidualQuery with primary/residual binary codes
 * @param prim_transposed   Primary signs in SoA layout [PRIM_WORDS][64]
 * @param res_transposed    Residual signs in SoA layout [RES_WORDS][64]
 * @param neighbor_offset   Starting neighbor index (0, 8, 16, ...)
 * @param out_distances     Output array for 8 combined distances
 */
template <size_t K, size_t R, int Shift = 2>
inline void residual_hamming_batch8_avx512(
    const ResidualQuery<K, R>& query,
    const uint64_t prim_transposed[][64],
    const uint64_t res_transposed[][64],
    size_t neighbor_offset,
    uint32_t* out_combined) {

    constexpr size_t PRIM_WORDS = BinaryCode<K>::NUM_WORDS;
    constexpr size_t RES_WORDS = BinaryCode<R>::NUM_WORDS;

    // Accumulators for primary and residual (8 neighbors each)
    __m512i prim_sums = _mm512_setzero_si512();
    __m512i res_sums = _mm512_setzero_si512();

    // Process primary code words
    for (size_t w = 0; w < PRIM_WORDS; ++w) {
        __m512i q = _mm512_set1_epi64(static_cast<long long>(query.primary.signs[w]));
        __m512i n = _mm512_loadu_si512(&prim_transposed[w][neighbor_offset]);
        __m512i xor_result = _mm512_xor_si512(q, n);

#ifdef __AVX512VPOPCNTDQ__
        __m512i popc = _mm512_popcnt_epi64(xor_result);
#else
        __m512i popc = harley_seal_popcnt_epi64(xor_result);
#endif
        prim_sums = _mm512_add_epi64(prim_sums, popc);
    }

    // Process residual code words
    for (size_t w = 0; w < RES_WORDS; ++w) {
        __m512i q = _mm512_set1_epi64(static_cast<long long>(query.residual.signs[w]));
        __m512i n = _mm512_loadu_si512(&res_transposed[w][neighbor_offset]);
        __m512i xor_result = _mm512_xor_si512(q, n);

#ifdef __AVX512VPOPCNTDQ__
        __m512i popc = _mm512_popcnt_epi64(xor_result);
#else
        __m512i popc = harley_seal_popcnt_epi64(xor_result);
#endif
        res_sums = _mm512_add_epi64(res_sums, popc);
    }

    // Combine with bit-shift weighting: (primary << Shift) + residual
    __m512i shifted_prim = _mm512_slli_epi64(prim_sums, Shift);
    __m512i combined = _mm512_add_epi64(shifted_prim, res_sums);

    // Extract 8 results
    alignas(64) uint64_t results[8];
    _mm512_storeu_si512(results, combined);

    for (int i = 0; i < 8; ++i) {
        out_combined[i] = static_cast<uint32_t>(results[i]);
    }
}

/**
 * Process full residual neighbor block using batch8 kernel.
 *
 * Converts combined integer distances to float at the very end.
 *
 * @param query              ResidualQuery structure
 * @param prim_transposed    Primary signs SoA [PRIM_WORDS][64]
 * @param res_transposed     Residual signs SoA [RES_WORDS][64]
 * @param neighbor_count     Number of neighbors to process
 * @param out_distances      Output float distances
 */
template <size_t K, size_t R, int Shift = 2>
inline void residual_distance_block_avx512(
    const ResidualQuery<K, R>& query,
    const uint64_t prim_transposed[][64],
    const uint64_t res_transposed[][64],
    size_t neighbor_count,
    AsymmetricDist* out_distances) {

    alignas(64) uint32_t batch_combined[8];

    for (size_t offset = 0; offset < neighbor_count; offset += 8) {
        size_t batch_size = std::min(size_t(8), neighbor_count - offset);

        residual_hamming_batch8_avx512<K, R, Shift>(
            query,
            prim_transposed,
            res_transposed,
            offset,
            batch_combined);

        // Convert to float ONCE at the end
        for (size_t i = 0; i < batch_size; ++i) {
            out_distances[offset + i] =
                query.base + query.scale * static_cast<float>(batch_combined[i]);
        }
    }
}

#endif  // CPHNSW_HAS_AVX512

// ============================================================================
// AVX2 Fallback for Residual Distance
// ============================================================================

#if CPHNSW_HAS_AVX2 && !CPHNSW_HAS_AVX512

/**
 * AVX2 batch residual distance - processes 4 neighbors at once.
 */
template <size_t K, size_t R, int Shift = 2>
inline void residual_hamming_batch4_avx2(
    const ResidualQuery<K, R>& query,
    const uint64_t prim_transposed[][64],
    const uint64_t res_transposed[][64],
    size_t neighbor_offset,
    uint32_t* out_combined) {

    constexpr size_t PRIM_WORDS = BinaryCode<K>::NUM_WORDS;
    constexpr size_t RES_WORDS = BinaryCode<R>::NUM_WORDS;

    __m256i prim_sums = _mm256_setzero_si256();
    __m256i res_sums = _mm256_setzero_si256();

    // Process primary code words
    for (size_t w = 0; w < PRIM_WORDS; ++w) {
        __m256i q = _mm256_set1_epi64x(static_cast<long long>(query.primary.signs[w]));
        __m256i n = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&prim_transposed[w][neighbor_offset]));
        __m256i xor_result = _mm256_xor_si256(q, n);

        // Software popcount
        alignas(32) uint64_t arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), xor_result);
        alignas(32) uint64_t popc_arr[4];
        for (int i = 0; i < 4; ++i) {
            popc_arr[i] = static_cast<uint64_t>(__builtin_popcountll(arr[i]));
        }
        __m256i popc = _mm256_load_si256(reinterpret_cast<const __m256i*>(popc_arr));
        prim_sums = _mm256_add_epi64(prim_sums, popc);
    }

    // Process residual code words
    for (size_t w = 0; w < RES_WORDS; ++w) {
        __m256i q = _mm256_set1_epi64x(static_cast<long long>(query.residual.signs[w]));
        __m256i n = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&res_transposed[w][neighbor_offset]));
        __m256i xor_result = _mm256_xor_si256(q, n);

        alignas(32) uint64_t arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), xor_result);
        alignas(32) uint64_t popc_arr[4];
        for (int i = 0; i < 4; ++i) {
            popc_arr[i] = static_cast<uint64_t>(__builtin_popcountll(arr[i]));
        }
        __m256i popc = _mm256_load_si256(reinterpret_cast<const __m256i*>(popc_arr));
        res_sums = _mm256_add_epi64(res_sums, popc);
    }

    // Combine with bit-shift weighting
    __m256i shifted_prim = _mm256_slli_epi64(prim_sums, Shift);
    __m256i combined = _mm256_add_epi64(shifted_prim, res_sums);

    alignas(32) uint64_t results[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), combined);

    for (int i = 0; i < 4; ++i) {
        out_combined[i] = static_cast<uint32_t>(results[i]);
    }
}

#endif  // CPHNSW_HAS_AVX2 && !CPHNSW_HAS_AVX512

// ============================================================================
// Dispatcher for Residual Distance
// ============================================================================

/**
 * Residual distance with automatic SIMD dispatch.
 * Chooses best implementation based on available hardware.
 */
template <size_t K, size_t R, int Shift = 2>
inline AsymmetricDist residual_distance(
    const ResidualQuery<K, R>& query,
    const ResidualBinaryCode<K, R>& node) {

#if CPHNSW_HAS_AVX512
    uint32_t combined = residual_distance_integer_avx512<K, R, Shift>(
        query.primary, query.residual,
        node.primary, node.residual);
#else
    uint32_t combined = residual_distance_integer_scalar<K, R, Shift>(
        query.primary, query.residual,
        node.primary, node.residual);
#endif

    return query.base + query.scale * static_cast<float>(combined);
}

// Common type aliases for residual distance
template <int Shift = 2>
using ResidualDistanceFunc64_32 = AsymmetricDist(*)(
    const ResidualQuery<64, 32>&,
    const ResidualBinaryCode<64, 32>&);

template <int Shift = 2>
using ResidualDistanceFunc32_16 = AsymmetricDist(*)(
    const ResidualQuery<32, 16>&,
    const ResidualBinaryCode<32, 16>&);

}  // namespace cphnsw
