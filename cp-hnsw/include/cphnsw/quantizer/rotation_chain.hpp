#pragma once

#include "../core/types.hpp"
#include "hadamard.hpp"
#include <vector>
#include <random>
#include <cstring>

namespace cphnsw {

/**
 * Pseudo-Random Rotation Chain
 *
 * Implements the transformation: Ψ(x) = H D₃ H D₂ H D₁ x
 *
 * From Andoni et al. "Practical and Optimal LSH for Angular Distance":
 * Three applications of H*D (Hadamard * Diagonal signs) approximate
 * the effect of a true Gaussian random rotation, but in O(d log d) time
 * instead of O(d²).
 *
 * Each rotation uses independent random diagonal matrices D₁, D₂, D₃
 * where D_i is a diagonal matrix with random ±1 entries.
 */
class RotationChain {
public:
    /**
     * Construct rotation chain for k independent rotations.
     *
     * @param dim           Original vector dimension
     * @param num_rotations Number of independent rotations (k)
     * @param seed          Random seed for reproducibility
     */
    RotationChain(size_t dim, size_t num_rotations, uint64_t seed)
        : dim_(dim), num_rotations_(num_rotations) {

        // Pad to power of 2 for FHT
        padded_dim_ = next_power_of_two(dim);

        // Pre-generate sign matrices for all rotations
        generate_signs(seed);
    }

    /// Get padded dimension (power of 2)
    size_t padded_dim() const { return padded_dim_; }

    /// Get original dimension
    size_t dim() const { return dim_; }

    /// Get number of rotations
    size_t num_rotations() const { return num_rotations_; }

    /**
     * Apply a single rotation in-place.
     *
     * Computes: vec = H D₃ H D₂ H D₁ vec
     *
     * @param vec            Vector to rotate (length >= padded_dim_)
     * @param rotation_index Which rotation to apply [0, num_rotations)
     */
    void apply(Float* vec, size_t rotation_index) const {
        assert(rotation_index < num_rotations_);

        const auto& signs = signs_[rotation_index];

        // D₁ * x
        apply_diagonal(vec, signs[0].data());

        // H * D₁ * x
        fht(vec, padded_dim_);

        // D₂ * H * D₁ * x
        apply_diagonal(vec, signs[1].data());

        // H * D₂ * H * D₁ * x
        fht(vec, padded_dim_);

        // D₃ * H * D₂ * H * D₁ * x
        apply_diagonal(vec, signs[2].data());

        // H * D₃ * H * D₂ * H * D₁ * x
        fht(vec, padded_dim_);
    }

    /**
     * Apply rotation to a copy of the input vector.
     *
     * @param input          Input vector (length >= dim_)
     * @param output         Output buffer (length >= padded_dim_)
     * @param rotation_index Which rotation to apply
     */
    void apply_copy(const Float* input, Float* output, size_t rotation_index) const {
        // Copy and zero-pad
        std::memcpy(output, input, dim_ * sizeof(Float));
        std::memset(output + dim_, 0, (padded_dim_ - dim_) * sizeof(Float));

        // Apply rotation in-place
        apply(output, rotation_index);
    }

private:
    size_t dim_;
    size_t padded_dim_;
    size_t num_rotations_;

    // signs_[rotation_index][layer_0_1_2][dim_index] = ±1
    std::vector<std::array<std::vector<int8_t>, 3>> signs_;

    /**
     * Generate random sign matrices for all rotations.
     */
    void generate_signs(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);

        signs_.resize(num_rotations_);

        for (size_t r = 0; r < num_rotations_; ++r) {
            for (int layer = 0; layer < 3; ++layer) {
                signs_[r][layer].resize(padded_dim_);
                for (size_t i = 0; i < padded_dim_; ++i) {
                    signs_[r][layer][i] = coin(rng) ? 1 : -1;
                }
            }
        }
    }

    /**
     * Apply diagonal sign matrix in-place: vec[i] *= signs[i]
     */
    void apply_diagonal(Float* vec, const int8_t* signs) const {
#if CPHNSW_HAS_AVX2
        apply_diagonal_avx2(vec, signs);
#else
        apply_diagonal_scalar(vec, signs);
#endif
    }

    /**
     * Scalar diagonal multiplication.
     */
    void apply_diagonal_scalar(Float* vec, const int8_t* signs) const {
        for (size_t i = 0; i < padded_dim_; ++i) {
            vec[i] *= static_cast<Float>(signs[i]);
        }
    }

#if CPHNSW_HAS_AVX2
    /**
     * AVX2 diagonal multiplication.
     * Processes 8 floats at a time.
     */
    void apply_diagonal_avx2(Float* vec, const int8_t* signs) const {
        // Convert signs to float for multiplication
        // Process in chunks of 8
        size_t i = 0;
        for (; i + 8 <= padded_dim_; i += 8) {
            // Load 8 floats
            __m256 v = _mm256_loadu_ps(&vec[i]);

            // Load 8 int8 signs and convert to float
            // First load as 64-bit integer, then expand
            __m128i signs_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&signs[i]));

            // Expand int8 to int32
            __m256i signs_i32 = _mm256_cvtepi8_epi32(signs_i8);

            // Convert int32 to float
            __m256 signs_f = _mm256_cvtepi32_ps(signs_i32);

            // Multiply
            __m256 result = _mm256_mul_ps(v, signs_f);

            // Store
            _mm256_storeu_ps(&vec[i], result);
        }

        // Handle remainder
        for (; i < padded_dim_; ++i) {
            vec[i] *= static_cast<Float>(signs[i]);
        }
    }
#endif
};

}  // namespace cphnsw
