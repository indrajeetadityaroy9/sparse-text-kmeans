#pragma once

#include "../../core/codes.hpp"
#include <immintrin.h>
#include <cstdint>
#include <algorithm>

namespace cphnsw {
namespace detail {

// ============================================================================
// AVX-512 VPOPCNT Fallback (for Skylake-X without VPOPCNTDQ)
// ============================================================================

#ifndef __AVX512VPOPCNTDQ__

/**
 * Harley-Seal popcount for AVX-512 (software fallback).
 * Used on Skylake-X which lacks VPOPCNTDQ.
 */
inline __m512i harley_seal_popcnt_epi64(__m512i v) {
    // Extract to scalar, popcount, repack
    // This is slower than native VPOPCNTDQ but still faster than scalar loop
    alignas(64) uint64_t arr[8];
    _mm512_storeu_si512(arr, v);

    for (int i = 0; i < 8; ++i) {
        arr[i] = static_cast<uint64_t>(__builtin_popcountll(arr[i]));
    }

    return _mm512_load_si512(arr);
}

#endif

// ============================================================================
// AVX-512 Batch Hamming Distance: Processes 8 neighbors at once
// ============================================================================

/**
 * AVX-512 vertical accumulation kernel for Hamming distance.
 *
 * KEY OPTIMIZATION: Vertical accumulation - loop through K words WITHOUT
 * horizontal reduction until the very end. Horizontal reductions are slow
 * (3-6 cycles) and break the pipeline.
 *
 * @tparam K Primary code bits
 * @tparam R Residual code bits (0 = Phase 1)
 * @tparam Shift Bit-shift weighting
 * @param query Query codes
 * @param soa_layout Transposed code storage
 * @param neighbor_offset Starting neighbor index (must be aligned to 8)
 * @param out_combined Output: 8 combined Hamming distances
 */
template <size_t K, size_t R, int Shift>
inline void hamming_batch8_avx512(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t neighbor_offset,
    uint32_t* out_combined) {

    constexpr size_t PRIMARY_WORDS = ResidualCode<K, R>::PRIMARY_WORDS;
    constexpr size_t RESIDUAL_WORDS = ResidualCode<K, R>::RESIDUAL_WORDS;

    // Accumulators stay in registers across all K words (vertical accumulation)
    __m512i prim_sums = _mm512_setzero_si512();
    __m512i res_sums = _mm512_setzero_si512();

    // Process primary code words
    for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
        // Broadcast query word to all 8 lanes
        __m512i q = _mm512_set1_epi64(
            static_cast<long long>(query.code.primary.signs[w]));

        // Load 8 neighbors' word w (contiguous in SoA layout!)
        __m512i n = _mm512_loadu_si512(
            &soa_layout.primary_transposed[w][neighbor_offset]);

        // XOR to find differing bits
        __m512i xor_result = _mm512_xor_si512(q, n);

        // PopCount (native VPOPCNTDQ or Harley-Seal fallback)
#ifdef __AVX512VPOPCNTDQ__
        __m512i popc = _mm512_popcnt_epi64(xor_result);
#else
        __m512i popc = harley_seal_popcnt_epi64(xor_result);
#endif

        // Accumulate (no reduction yet - stays vertical!)
        prim_sums = _mm512_add_epi64(prim_sums, popc);
    }

    // Process residual code words (if R > 0)
    if constexpr (R > 0) {
        for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
            __m512i q = _mm512_set1_epi64(
                static_cast<long long>(query.code.residual.signs[w]));

            __m512i n = _mm512_loadu_si512(
                &soa_layout.residual_transposed[w][neighbor_offset]);

            __m512i xor_result = _mm512_xor_si512(q, n);

#ifdef __AVX512VPOPCNTDQ__
            __m512i popc = _mm512_popcnt_epi64(xor_result);
#else
            __m512i popc = harley_seal_popcnt_epi64(xor_result);
#endif

            res_sums = _mm512_add_epi64(res_sums, popc);
        }
    }

    // Combine with bit-shift weighting: (primary << Shift) + residual
    __m512i combined;
    if constexpr (R == 0) {
        combined = prim_sums;
    } else {
        __m512i shifted_prim = _mm512_slli_epi64(prim_sums, Shift);
        combined = _mm512_add_epi64(shifted_prim, res_sums);
    }

    // ONLY NOW extract 8 distances (single horizontal operation at end)
    alignas(64) uint64_t results[8];
    _mm512_storeu_si512(results, combined);

    for (int i = 0; i < 8; ++i) {
        out_combined[i] = static_cast<uint32_t>(results[i]);
    }
}

/**
 * AVX-512 batch distance computation.
 * Processes 8 neighbors at a time with vertical accumulation.
 */
template <size_t K, size_t R, int Shift>
void distance_batch_avx512(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    alignas(64) uint32_t batch_hamming[8];
    size_t n = 0;

    // Process 8 neighbors at a time
    for (; n + 8 <= count; n += 8) {
        hamming_batch8_avx512<K, R, Shift>(query, soa_layout, n, batch_hamming);

        // Convert Hamming to float distances using AVX-512
        __m256i hamming_i32 = _mm256_setr_epi32(
            batch_hamming[0], batch_hamming[1], batch_hamming[2], batch_hamming[3],
            batch_hamming[4], batch_hamming[5], batch_hamming[6], batch_hamming[7]);

        __m256 hamming_f = _mm256_cvtepi32_ps(hamming_i32);
        __m256 scale = _mm256_set1_ps(query.scale);
        __m256 base = _mm256_set1_ps(query.base);
        __m256 distances = _mm256_fmadd_ps(hamming_f, scale, base);

        _mm256_storeu_ps(out_distances + n, distances);
    }

    // Handle remainder with scalar
    for (; n < count; ++n) {
        uint32_t prim_dist = 0;
        for (size_t w = 0; w < ResidualCode<K, R>::PRIMARY_WORDS; ++w) {
            prim_dist += popcount64(
                query.code.primary.signs[w] ^ soa_layout.primary_transposed[w][n]);
        }

        uint32_t combined;
        if constexpr (R == 0) {
            combined = prim_dist;
        } else {
            uint32_t res_dist = 0;
            for (size_t w = 0; w < ResidualCode<K, R>::RESIDUAL_WORDS; ++w) {
                res_dist += popcount64(
                    query.code.residual.signs[w] ^ soa_layout.residual_transposed[w][n]);
            }
            combined = (prim_dist << Shift) + res_dist;
        }

        out_distances[n] = query.base + query.scale * static_cast<float>(combined);
    }
}

/**
 * AVX-512 batch Hamming computation (returns raw integers).
 */
template <size_t K, size_t R, int Shift>
void hamming_batch_avx512(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    alignas(64) uint32_t batch_hamming[8];
    size_t n = 0;

    // Process 8 neighbors at a time
    for (; n + 8 <= count; n += 8) {
        hamming_batch8_avx512<K, R, Shift>(query, soa_layout, n, batch_hamming);

        for (size_t i = 0; i < 8; ++i) {
            out_hamming[n + i] = batch_hamming[i];
        }
    }

    // Handle remainder with scalar
    for (; n < count; ++n) {
        uint32_t prim_dist = 0;
        for (size_t w = 0; w < ResidualCode<K, R>::PRIMARY_WORDS; ++w) {
            prim_dist += popcount64(
                query.code.primary.signs[w] ^ soa_layout.primary_transposed[w][n]);
        }

        if constexpr (R == 0) {
            out_hamming[n] = prim_dist;
        } else {
            uint32_t res_dist = 0;
            for (size_t w = 0; w < ResidualCode<K, R>::RESIDUAL_WORDS; ++w) {
                res_dist += popcount64(
                    query.code.residual.signs[w] ^ soa_layout.residual_transposed[w][n]);
            }
            out_hamming[n] = (prim_dist << Shift) + res_dist;
        }
    }
}

}  // namespace detail

// ============================================================================
// Override UnifiedMetricPolicy batch methods with AVX-512 implementations
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)

// When AVX-512 is available, use the highest performance kernels
template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_distance_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    detail::distance_batch_avx512<K, R, Shift>(query, soa_layout, count, out_distances);
}

template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_hamming_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    detail::hamming_batch_avx512<K, R, Shift>(query, soa_layout, count, out_hamming);
}

#endif  // __AVX512F__ && __AVX512BW__

}  // namespace cphnsw
