#pragma once

#include <cstddef>
#include <cassert>

// SIMD detection
#if defined(__AVX512F__)
#define CPHNSW_FHT_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define CPHNSW_FHT_AVX512 0
#define CPHNSW_FHT_AVX2 1
#include <immintrin.h>
#else
#define CPHNSW_FHT_AVX512 0
#define CPHNSW_FHT_AVX2 0
#endif

namespace cphnsw {

// ============================================================================
// Fast Hadamard Transform
// ============================================================================

/**
 * Fast Hadamard Transform (FHT)
 *
 * Computes H * x in-place where H is the Walsh-Hadamard matrix.
 * The transform is its own inverse (up to scaling by 1/sqrt(n)).
 *
 * Mathematical definition (recursive):
 *   H_1 = [1]
 *   H_{2n} = [[H_n, H_n], [H_n, -H_n]]
 *
 * Complexity: O(n log n) time, O(1) auxiliary space
 *
 * Note: No normalization is applied. The transform increases magnitude
 * by sqrt(n), but since we use argmax (scale-invariant), this is fine.
 */

namespace detail {

inline bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

}  // namespace detail

// ============================================================================
// Scalar Implementation (Reference/Fallback)
// ============================================================================

/**
 * Scalar Fast Hadamard Transform using classic butterfly algorithm.
 */
inline void fht_scalar(float* vec, size_t len) {
    assert(detail::is_power_of_two(len) && "FHT requires power-of-2 length");

    for (size_t h = 1; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; ++j) {
                float x = vec[j];
                float y = vec[j + h];
                vec[j] = x + y;
                vec[j + h] = x - y;
            }
        }
    }
}

// ============================================================================
// AVX2 Implementation
// ============================================================================

#if CPHNSW_FHT_AVX2 || CPHNSW_FHT_AVX512

/**
 * AVX2 Fast Hadamard Transform.
 * Processes 8 floats at a time.
 */
inline void fht_avx2(float* vec, size_t len) {
    assert(detail::is_power_of_two(len) && "FHT requires power-of-2 length");

    if (len < 8) {
        fht_scalar(vec, len);
        return;
    }

    constexpr size_t SIMD_WIDTH = 8;

    // Phase 1: Intra-register butterfly for h < 8
    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m256 v = _mm256_loadu_ps(&vec[i]);

        // h = 1: Swap adjacent pairs
        __m256 v_swap1 = _mm256_permute_ps(v, 0b10110001);
        __m256 v_add1 = _mm256_add_ps(v, v_swap1);
        __m256 v_sub1 = _mm256_sub_ps(v, v_swap1);
        v = _mm256_blend_ps(v_add1, v_sub1, 0b10101010);

        // h = 2: Swap pairs of pairs
        __m256 v_swap2 = _mm256_permute_ps(v, 0b01001110);
        __m256 v_add2 = _mm256_add_ps(v, v_swap2);
        __m256 v_sub2 = _mm256_sub_ps(v, v_swap2);
        v = _mm256_blend_ps(v_add2, v_sub2, 0b11001100);

        // h = 4: Swap 128-bit lanes
        __m256 v_swap4 = _mm256_permute2f128_ps(v, v, 0x01);
        __m256 v_add4 = _mm256_add_ps(v, v_swap4);
        __m256 v_sub4 = _mm256_sub_ps(v, v_swap4);
        v = _mm256_blend_ps(v_add4, v_sub4, 0b11110000);

        _mm256_storeu_ps(&vec[i], v);
    }

    // Phase 2: Cross-register butterfly for h >= 8
    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m256 x = _mm256_loadu_ps(&vec[j]);
                __m256 y = _mm256_loadu_ps(&vec[j + h]);
                _mm256_storeu_ps(&vec[j], _mm256_add_ps(x, y));
                _mm256_storeu_ps(&vec[j + h], _mm256_sub_ps(x, y));
            }
        }
    }
}

#endif  // CPHNSW_FHT_AVX2 || CPHNSW_FHT_AVX512

// ============================================================================
// AVX-512 Implementation
// ============================================================================

#if CPHNSW_FHT_AVX512

/**
 * AVX-512 Fast Hadamard Transform.
 * Processes 16 floats at a time.
 */
inline void fht_avx512(float* vec, size_t len) {
    assert(detail::is_power_of_two(len) && "FHT requires power-of-2 length");

    if (len < 16) {
        fht_avx2(vec, len);
        return;
    }

    constexpr size_t SIMD_WIDTH = 16;

    // Phase 1: Intra-register butterfly for h < 16
    for (size_t i = 0; i < len; i += SIMD_WIDTH) {
        __m512 v = _mm512_loadu_ps(&vec[i]);

        // h = 1
        __m512i idx1 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        __m512 v_swap1 = _mm512_permutexvar_ps(idx1, v);
        __m512 v_add1 = _mm512_add_ps(v, v_swap1);
        __m512 v_sub1 = _mm512_sub_ps(v, v_swap1);
        v = _mm512_mask_blend_ps(0xAAAA, v_add1, v_sub1);

        // h = 2
        __m512i idx2 = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
        __m512 v_swap2 = _mm512_permutexvar_ps(idx2, v);
        __m512 v_add2 = _mm512_add_ps(v, v_swap2);
        __m512 v_sub2 = _mm512_sub_ps(v, v_swap2);
        v = _mm512_mask_blend_ps(0xCCCC, v_add2, v_sub2);

        // h = 4
        __m512i idx4 = _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
        __m512 v_swap4 = _mm512_permutexvar_ps(idx4, v);
        __m512 v_add4 = _mm512_add_ps(v, v_swap4);
        __m512 v_sub4 = _mm512_sub_ps(v, v_swap4);
        v = _mm512_mask_blend_ps(0xF0F0, v_add4, v_sub4);

        // h = 8
        __m512 v_swap8 = _mm512_shuffle_f32x4(v, v, 0x4E);
        __m512 v_add8 = _mm512_add_ps(v, v_swap8);
        __m512 v_sub8 = _mm512_sub_ps(v, v_swap8);
        v = _mm512_mask_blend_ps(0xFF00, v_add8, v_sub8);

        _mm512_storeu_ps(&vec[i], v);
    }

    // Phase 2: Cross-register butterfly for h >= 16
    for (size_t h = SIMD_WIDTH; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += SIMD_WIDTH) {
                __m512 x = _mm512_loadu_ps(&vec[j]);
                __m512 y = _mm512_loadu_ps(&vec[j + h]);
                _mm512_storeu_ps(&vec[j], _mm512_add_ps(x, y));
                _mm512_storeu_ps(&vec[j + h], _mm512_sub_ps(x, y));
            }
        }
    }
}

#endif  // CPHNSW_FHT_AVX512

// ============================================================================
// Dispatcher
// ============================================================================

/**
 * Fast Hadamard Transform with automatic SIMD dispatch.
 */
inline void fht(float* vec, size_t len) {
#if CPHNSW_FHT_AVX512
    fht_avx512(vec, len);
#elif CPHNSW_FHT_AVX2
    fht_avx2(vec, len);
#else
    fht_scalar(vec, len);
#endif
}

}  // namespace cphnsw
