#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace cphnsw {

// ============================================================================
// Unified Binary Code Storage
// ============================================================================

/**
 * BinaryCodeStorage: Low-level storage for packed sign bits.
 *
 * This is the fundamental building block for all code types.
 * Stores K sign bits packed into 64-bit words, aligned for SIMD access.
 *
 * @tparam Bits Number of sign bits to store
 */
template <size_t Bits>
struct alignas(64) BinaryCodeStorage {
    static constexpr size_t NUM_BITS = Bits;
    static constexpr size_t NUM_WORDS = (Bits + 63) / 64;

    uint64_t signs[NUM_WORDS];

    void clear() {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            signs[i] = 0;
        }
    }

    void set_bit(size_t idx, bool value) {
        if (value) {
            signs[idx / 64] |= (1ULL << (idx % 64));
        } else {
            signs[idx / 64] &= ~(1ULL << (idx % 64));
        }
    }

    bool get_bit(size_t idx) const {
        return (signs[idx / 64] >> (idx % 64)) & 1;
    }

    bool operator==(const BinaryCodeStorage& other) const {
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            if (signs[i] != other.signs[i]) return false;
        }
        return true;
    }
};

// ============================================================================
// Unified Code Type: ResidualCode<K, R>
// ============================================================================

/**
 * ResidualCode: Unified code type supporting both Phase 1 and Phase 2.
 *
 * KEY DESIGN PRINCIPLE: When R=0, the residual storage is empty and
 * all residual-related operations compile to no-ops. This means
 * ResidualCode<K, 0> is functionally equivalent to BinaryCode<K>.
 *
 * Memory Layout (for R > 0):
 *   - Primary: K bits packed into (K+63)/64 words
 *   - Residual: R bits packed into (R+63)/64 words
 *
 * Memory Layout (for R = 0):
 *   - Primary only: K bits packed into (K+63)/64 words
 *   - Residual storage: empty (zero size)
 *
 * @tparam K Primary code bits (e.g., 32, 64)
 * @tparam R Residual code bits (e.g., 0, 16, 32). R=0 disables residual.
 */
template <size_t K, size_t R = 0>
struct ResidualCode {
    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    static constexpr size_t PRIMARY_WORDS = (K + 63) / 64;
    static constexpr size_t RESIDUAL_WORDS = (R > 0) ? ((R + 63) / 64) : 0;
    static constexpr size_t TOTAL_WORDS = PRIMARY_WORDS + RESIDUAL_WORDS;

    // Storage for primary bits (always present)
    BinaryCodeStorage<K> primary;

    // Storage for residual bits (empty when R=0)
    // Use conditional type to avoid allocating space when R=0
    struct EmptyStorage {};
    using ResidualStorageType = std::conditional_t<(R > 0), BinaryCodeStorage<R>, EmptyStorage>;
    [[no_unique_address]] ResidualStorageType residual;

    void clear() {
        primary.clear();
        if constexpr (R > 0) {
            residual.clear();
        }
    }

    // Primary bit access
    void set_primary_bit(size_t idx, bool value) {
        primary.set_bit(idx, value);
    }

    bool get_primary_bit(size_t idx) const {
        return primary.get_bit(idx);
    }

    // Residual bit access (only valid when R > 0)
    template <size_t R_ = R>
    std::enable_if_t<(R_ > 0)> set_residual_bit(size_t idx, bool value) {
        residual.set_bit(idx, value);
    }

    template <size_t R_ = R>
    std::enable_if_t<(R_ > 0), bool> get_residual_bit(size_t idx) const {
        return residual.get_bit(idx);
    }

    bool operator==(const ResidualCode& other) const {
        if (!(primary == other.primary)) return false;
        if constexpr (R > 0) {
            if (!(residual == other.residual)) return false;
        }
        return true;
    }
};

// ============================================================================
// Type Aliases for Common Configurations (Unified API)
// ============================================================================

// Note: These use "Code" suffix to avoid conflict with legacy BinaryCode<K>
// Phase 1: Pure binary codes (no residual)
using Code32 = ResidualCode<32, 0>;
using Code64 = ResidualCode<64, 0>;

// Phase 2: Binary codes with residual
using Code64_32 = ResidualCode<64, 32>;
using Code32_16 = ResidualCode<32, 16>;

// ============================================================================
// Query Structure: Pre-computed scalars for distance computation
// ============================================================================

/**
 * CodeQuery: Query-time structure with pre-computed distance scalars.
 *
 * The distance formula is:
 *   distance = base + scale * combined_hamming
 *
 * Where combined_hamming = (primary_hamming << Shift) + residual_hamming
 *
 * @tparam K Primary code bits
 * @tparam R Residual code bits (0 = Phase 1 mode)
 * @tparam Shift Bit-shift for primary/residual weighting (default 2 = 4:1)
 */
template <size_t K, size_t R = 0, int Shift = 2>
struct CodeQuery {
    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int WEIGHT_SHIFT = Shift;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    // Binary codes for XOR + PopCount
    ResidualCode<K, R> code;

    // Pre-computed scalars (computed once per query)
    float base;   // Constant offset
    float scale;  // Per-bit scaling factor

    // Query norm (for potential reranking)
    float query_norm;

    // Compute weight ratios at compile time
    static constexpr float primary_weight() {
        if constexpr (R == 0) {
            return 1.0f;
        } else {
            return static_cast<float>(1 << Shift) / static_cast<float>((1 << Shift) + 1);
        }
    }

    static constexpr float residual_weight() {
        if constexpr (R == 0) {
            return 0.0f;
        } else {
            return 1.0f / static_cast<float>((1 << Shift) + 1);
        }
    }
};

// ============================================================================
// SoA (Structure-of-Arrays) Layout for SIMD Batch Processing
// ============================================================================

/**
 * CodeSoALayout: Transposed storage for SIMD-friendly batch distance computation.
 *
 * Instead of storing codes as:
 *   [N0: word0, word1][N1: word0, word1]...  (AoS)
 *
 * We store them as:
 *   [word0: N0, N1, N2...][word1: N0, N1, N2...]  (SoA)
 *
 * This enables contiguous SIMD loads for batch processing.
 *
 * @tparam CodeT The code type (ResidualCode<K, R>)
 * @tparam MaxNeighbors Maximum neighbors per block (typically 64)
 */
template <typename CodeT, size_t MaxNeighbors = 64>
struct CodeSoALayout {
    static constexpr size_t PRIMARY_WORDS = CodeT::PRIMARY_WORDS;
    static constexpr size_t RESIDUAL_WORDS = CodeT::RESIDUAL_WORDS;
    static constexpr size_t MAX_N = MaxNeighbors;

    // Transposed primary signs: [word_idx][neighbor_idx]
    alignas(64) uint64_t primary_transposed[PRIMARY_WORDS][MAX_N];

    // Transposed residual signs (empty when R=0)
    struct EmptyArray {};
    using ResidualArrayType = std::conditional_t<
        (CodeT::RESIDUAL_BITS > 0),
        uint64_t[RESIDUAL_WORDS][MAX_N],
        EmptyArray
    >;
    [[no_unique_address]] alignas(64) ResidualArrayType residual_transposed;

    void clear() {
        std::memset(primary_transposed, 0, sizeof(primary_transposed));
        if constexpr (CodeT::RESIDUAL_BITS > 0) {
            std::memset(residual_transposed, 0, sizeof(residual_transposed));
        }
    }

    // Store a code at the given neighbor index
    void store(size_t neighbor_idx, const CodeT& code) {
        for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
            primary_transposed[w][neighbor_idx] = code.primary.signs[w];
        }
        if constexpr (CodeT::RESIDUAL_BITS > 0) {
            for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
                residual_transposed[w][neighbor_idx] = code.residual.signs[w];
            }
        }
    }

    // Load a code from the given neighbor index
    void load(size_t neighbor_idx, CodeT& code) const {
        for (size_t w = 0; w < PRIMARY_WORDS; ++w) {
            code.primary.signs[w] = primary_transposed[w][neighbor_idx];
        }
        if constexpr (CodeT::RESIDUAL_BITS > 0) {
            for (size_t w = 0; w < RESIDUAL_WORDS; ++w) {
                code.residual.signs[w] = residual_transposed[w][neighbor_idx];
            }
        }
    }
};

// ============================================================================
// Compile-time code traits
// ============================================================================

template <typename CodeT>
struct CodeTraits {
    static constexpr size_t primary_bits = CodeT::PRIMARY_BITS;
    static constexpr size_t residual_bits = CodeT::RESIDUAL_BITS;
    static constexpr bool has_residual = CodeT::HAS_RESIDUAL;
    static constexpr size_t total_bits = primary_bits + residual_bits;
    static constexpr size_t storage_bytes = CodeT::TOTAL_WORDS * sizeof(uint64_t);
};

}  // namespace cphnsw
