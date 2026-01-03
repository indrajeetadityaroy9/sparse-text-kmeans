#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include <cstddef>
#include <type_traits>

namespace cphnsw {

// ============================================================================
// Distance Type
// ============================================================================

using DistanceType = float;

// ============================================================================
// Metric Policy Concept (C++20 concept, with SFINAE fallback for C++17)
// ============================================================================

/**
 * MetricPolicy: Interface for distance computation strategies.
 *
 * A valid MetricPolicy must provide:
 *
 * Types:
 *   - CodeType: The binary code storage type (e.g., ResidualCode<K, R>)
 *   - QueryType: The query structure with pre-computed scalars
 *   - SoALayoutType: Transposed storage for batch processing
 *
 * Static Methods:
 *   - compute_distance(query, code) -> DistanceType
 *   - compute_distance_batch(query, soa_layout, count, out_distances)
 *   - compute_hamming(code_a, code_b) -> uint32_t (raw Hamming distance)
 *
 * This design allows the SearchEngine to be generic over the distance metric,
 * with compile-time dispatch to the appropriate SIMD kernels.
 */

// Forward declarations
template <size_t K, size_t R, int Shift>
struct ResidualMetricPolicy;

// ============================================================================
// Unified Metric Policy: Handles both Phase 1 (R=0) and Phase 2 (R>0)
// ============================================================================

/**
 * UnifiedMetricPolicy: Single policy that adapts to R=0 or R>0.
 *
 * When R=0: Becomes equivalent to RaBitQ (XOR + PopCount on primary only)
 * When R>0: Full residual distance with bit-shift weighting
 *
 * @tparam K Primary code bits
 * @tparam R Residual code bits (0 = Phase 1 mode)
 * @tparam Shift Bit-shift for weighting (default 2 = 4:1 ratio)
 */
template <size_t K, size_t R = 0, int Shift = 2>
struct UnifiedMetricPolicy {
    // Type definitions
    using CodeType = ResidualCode<K, R>;
    using QueryType = CodeQuery<K, R, Shift>;
    using SoALayoutType = CodeSoALayout<CodeType, 64>;

    // Configuration
    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int WEIGHT_SHIFT = Shift;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    // ========================================================================
    // Single Distance Computation
    // ========================================================================

    /**
     * Compute distance between query and a single code.
     *
     * When R=0: distance = base + scale * popcount(query.primary XOR code.primary)
     * When R>0: distance = base + scale * ((primary_hamming << Shift) + residual_hamming)
     */
    static inline DistanceType compute_distance(
        const QueryType& query,
        const CodeType& code);

    /**
     * Compute raw Hamming distance (combined if R>0).
     *
     * When R=0: Returns popcount(query.primary XOR code.primary)
     * When R>0: Returns (primary_hamming << Shift) + residual_hamming
     */
    static inline uint32_t compute_hamming(
        const CodeType& code_a,
        const CodeType& code_b);

    // ========================================================================
    // Batch Distance Computation (SIMD-optimized)
    // ========================================================================

    /**
     * Compute distances to multiple codes stored in SoA layout.
     *
     * This is the HOT PATH for search. Uses SIMD to process 8 (AVX2) or
     * 16 (AVX-512) codes simultaneously.
     *
     * @param query The query with pre-computed scalars
     * @param soa_layout Transposed code storage
     * @param count Number of codes to process
     * @param out_distances Output array (must have at least count elements)
     */
    static void compute_distance_batch(
        const QueryType& query,
        const SoALayoutType& soa_layout,
        size_t count,
        DistanceType* out_distances);

    /**
     * Compute raw Hamming distances for a batch (returns integers).
     *
     * Used when you need the raw Hamming values before scaling.
     */
    static void compute_hamming_batch(
        const QueryType& query,
        const SoALayoutType& soa_layout,
        size_t count,
        uint32_t* out_hamming);

    // ========================================================================
    // Prefetching Hints
    // ========================================================================

    /**
     * Prefetch code storage for upcoming distance computation.
     */
    static inline void prefetch_code(const CodeType* code) {
        cphnsw::prefetch(code, 3);  // L1 cache
    }

    /**
     * Prefetch SoA layout for upcoming batch computation.
     */
    static inline void prefetch_soa(const SoALayoutType* soa, size_t start_idx) {
        // Prefetch primary signs for the batch
        for (size_t w = 0; w < CodeType::PRIMARY_WORDS; ++w) {
            cphnsw::prefetch(&soa->primary_transposed[w][start_idx], 3);
        }
        // Prefetch residual signs if present
        if constexpr (R > 0) {
            for (size_t w = 0; w < CodeType::RESIDUAL_WORDS; ++w) {
                cphnsw::prefetch(&soa->residual_transposed[w][start_idx], 3);
            }
        }
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1: Pure RaBitQ (no residual)
template <size_t K>
using RaBitQPolicy = UnifiedMetricPolicy<K, 0, 0>;

// Phase 2: With residual, default 4:1 weighting
template <size_t K, size_t R>
using ResidualPolicy = UnifiedMetricPolicy<K, R, 2>;

// Common instantiations
using Policy32 = RaBitQPolicy<32>;
using Policy64 = RaBitQPolicy<64>;
using Policy64_32 = ResidualPolicy<64, 32>;
using Policy32_16 = ResidualPolicy<32, 16>;

// ============================================================================
// Metric Policy Traits (for compile-time introspection)
// ============================================================================

template <typename Policy>
struct MetricPolicyTraits {
    using CodeType = typename Policy::CodeType;
    using QueryType = typename Policy::QueryType;
    using SoALayoutType = typename Policy::SoALayoutType;

    static constexpr size_t primary_bits = Policy::PRIMARY_BITS;
    static constexpr size_t residual_bits = Policy::RESIDUAL_BITS;
    static constexpr bool has_residual = Policy::HAS_RESIDUAL;
};

// ============================================================================
// SFINAE helper to check if a type is a valid MetricPolicy
// ============================================================================

namespace detail {

template <typename T, typename = void>
struct is_metric_policy_impl : std::false_type {};

template <typename T>
struct is_metric_policy_impl<T, std::void_t<
    typename T::CodeType,
    typename T::QueryType,
    typename T::SoALayoutType,
    decltype(T::compute_distance(
        std::declval<typename T::QueryType>(),
        std::declval<typename T::CodeType>()
    ))
>> : std::true_type {};

}  // namespace detail

template <typename T>
inline constexpr bool is_metric_policy_v = detail::is_metric_policy_impl<T>::value;

}  // namespace cphnsw

// Include implementations (keep headers clean, implementations in detail/)
#include "detail/scalar_kernels.hpp"

#if defined(__AVX2__)
#include "detail/avx2_kernels.hpp"
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include "detail/avx512_kernels.hpp"
#endif
