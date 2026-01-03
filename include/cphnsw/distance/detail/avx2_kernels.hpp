#pragma once

#include "../../core/codes.hpp"
#include <immintrin.h>
#include <cstdint>

namespace cphnsw {
namespace detail {

// ============================================================================
// AVX2 Batch Hamming Distance: Processes 4 neighbors at once
// ============================================================================

/**
 * AVX2 vertical accumulation kernel for Hamming distance.
 *
 * Processes 4 neighbors in parallel using 256-bit registers.
 * Uses software popcount (no VPOPCNT in AVX2).
 *
 * @tparam K Primary code bits
 * @tparam R Residual code bits (0 = Phase 1)
 * @tparam Shift Bit-shift weighting
 * @param query Query codes (primary + optional residual)
 * @param soa_layout Transposed code storage
 * @param neighbor_offset Starting neighbor index (must be aligned to 4)
 * @param out_combined Output: 4 combined Hamming distances
 */
template <size_t K, size_t R, int Shift>
inline void hamming_batch4_avx2(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t neighbor_offset,
    uint32_t* out_combined) {

    constexpr size_t PRIMARY_WORDS = ResidualCode<K, R>::PRIMARY_WORDS;
    constexpr size_t RESIDUAL_WORDS = ResidualCode<K, R>::RESIDUAL_WORDS;

    // Accumulators for primary and residual distances
    __m256i prim_sums = _mm256_setzero_si256();
    __m256i res_sums = _mm256_setzero_si256();

    // Process primary code words
    for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
        // Broadcast query word to all 4 lanes
        __m256i q = _mm256_set1_epi64x(
            static_cast<long long>(query.code.primary.signs[w]));

        // Load 4 neighbors' word w (contiguous in SoA layout)
        __m256i n = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                &soa_layout.primary_transposed[w][neighbor_offset]));

        // XOR to find differing bits
        __m256i xor_result = _mm256_xor_si256(q, n);

        // Software popcount per 64-bit word (AVX2 has no VPOPCNT)
        alignas(32) uint64_t xor_arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(xor_arr), xor_result);

        alignas(32) uint64_t popc_arr[4];
        for (int i = 0; i < 4; ++i) {
            popc_arr[i] = static_cast<uint64_t>(__builtin_popcountll(xor_arr[i]));
        }

        __m256i popc = _mm256_load_si256(reinterpret_cast<const __m256i*>(popc_arr));
        prim_sums = _mm256_add_epi64(prim_sums, popc);
    }

    // Process residual code words (if R > 0)
    if constexpr (R > 0) {
        for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
            __m256i q = _mm256_set1_epi64x(
                static_cast<long long>(query.code.residual.signs[w]));

            __m256i n = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    &soa_layout.residual_transposed[w][neighbor_offset]));

            __m256i xor_result = _mm256_xor_si256(q, n);

            alignas(32) uint64_t xor_arr[4];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(xor_arr), xor_result);

            alignas(32) uint64_t popc_arr[4];
            for (int i = 0; i < 4; ++i) {
                popc_arr[i] = static_cast<uint64_t>(__builtin_popcountll(xor_arr[i]));
            }

            __m256i popc = _mm256_load_si256(reinterpret_cast<const __m256i*>(popc_arr));
            res_sums = _mm256_add_epi64(res_sums, popc);
        }
    }

    // Combine with bit-shift weighting
    __m256i combined;
    if constexpr (R == 0) {
        combined = prim_sums;
    } else {
        __m256i shifted_prim = _mm256_slli_epi64(prim_sums, Shift);
        combined = _mm256_add_epi64(shifted_prim, res_sums);
    }

    // Extract 4 results
    alignas(32) uint64_t results[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), combined);

    for (int i = 0; i < 4; ++i) {
        out_combined[i] = static_cast<uint32_t>(results[i]);
    }
}

/**
 * AVX2 batch distance computation.
 * Processes 4 neighbors at a time with vertical accumulation.
 */
template <size_t K, size_t R, int Shift>
void distance_batch_avx2(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    alignas(32) uint32_t batch_hamming[4];
    size_t n = 0;

    // Process 4 neighbors at a time
    for (; n + 4 <= count; n += 4) {
        hamming_batch4_avx2<K, R, Shift>(query, soa_layout, n, batch_hamming);

        // Convert Hamming to float distances
        for (size_t i = 0; i < 4; ++i) {
            out_distances[n + i] =
                query.base + query.scale * static_cast<float>(batch_hamming[i]);
        }
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
 * AVX2 batch Hamming computation (returns raw integers).
 */
template <size_t K, size_t R, int Shift>
void hamming_batch_avx2(
    const CodeQuery<K, R, Shift>& query,
    const CodeSoALayout<ResidualCode<K, R>, 64>& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    alignas(32) uint32_t batch_hamming[4];
    size_t n = 0;

    // Process 4 neighbors at a time
    for (; n + 4 <= count; n += 4) {
        hamming_batch4_avx2<K, R, Shift>(query, soa_layout, n, batch_hamming);

        for (size_t i = 0; i < 4; ++i) {
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
// Override UnifiedMetricPolicy batch methods with AVX2 implementations
// ============================================================================

#if defined(__AVX2__) && !defined(__AVX512F__)

// When AVX2 is available but not AVX-512, use AVX2 batch kernels
template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_distance_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    DistanceType* out_distances) {

    detail::distance_batch_avx2<K, R, Shift>(query, soa_layout, count, out_distances);
}

template <size_t K, size_t R, int Shift>
void UnifiedMetricPolicy<K, R, Shift>::compute_hamming_batch(
    const QueryType& query,
    const SoALayoutType& soa_layout,
    size_t count,
    uint32_t* out_hamming) {

    detail::hamming_batch_avx2<K, R, Shift>(query, soa_layout, count, out_hamming);
}

#endif  // __AVX2__ && !__AVX512F__

}  // namespace cphnsw
