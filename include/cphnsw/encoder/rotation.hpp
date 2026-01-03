#pragma once

#include "../core/memory.hpp"
#include "transform/fht.hpp"
#include <vector>
#include <array>
#include <random>
#include <memory>
#include <cstring>

namespace cphnsw {

// ============================================================================
// RotationStrategy: Abstract interface for vector rotation
// ============================================================================

/**
 * RotationStrategy: Interface for pseudo-random rotation schemes.
 *
 * The Cross-Polytope LSH requires rotating vectors before argmax selection.
 * Different rotation strategies trade off between:
 *   - Speed: FHT is O(d log d), dense matrix is O(d²)
 *   - Quality: Learned rotations can improve quantization quality
 *
 * Implementations:
 *   - RandomHadamard: Standard random rotation (Phase 1/2)
 *   - LearnedDiagonal: Data-dependent learned rotations (Phase 3)
 */
class RotationStrategy {
public:
    virtual ~RotationStrategy() = default;

    /**
     * Apply rotation in-place.
     * @param x Vector to rotate (must be padded_dim() length)
     */
    virtual void apply(float* x) const = 0;

    /**
     * Apply rotation to a copy.
     * @param input Input vector (original_dim() length)
     * @param output Output buffer (padded_dim() length)
     */
    virtual void apply_copy(const float* input, float* output) const = 0;

    /**
     * Get original dimension.
     */
    virtual size_t original_dim() const = 0;

    /**
     * Get padded dimension (power of 2 for FHT).
     */
    virtual size_t padded_dim() const = 0;
};

// ============================================================================
// RandomHadamardRotation: Standard rotation chain (H D₃ H D₂ H D₁)
// ============================================================================

/**
 * RandomHadamardRotation: The standard Cross-Polytope rotation scheme.
 *
 * Implements: Ψ(x) = H D₃ H D₂ H D₁ x
 *
 * Where:
 *   - H is the Walsh-Hadamard matrix
 *   - D₁, D₂, D₃ are diagonal matrices with random ±1 entries
 *
 * From Andoni et al. "Practical and Optimal LSH for Angular Distance":
 * Three applications of H*D approximate a true Gaussian random rotation
 * in O(d log d) time instead of O(d²).
 */
class RandomHadamardRotation : public RotationStrategy {
public:
    static constexpr size_t NUM_LAYERS = 3;

    /**
     * Construct rotation with random signs.
     * @param dim Original vector dimension
     * @param seed Random seed for reproducibility
     */
    RandomHadamardRotation(size_t dim, uint64_t seed)
        : original_dim_(dim)
        , padded_dim_(next_power_of_two(dim)) {

        generate_signs(seed);
    }

    void apply(float* x) const override {
        // D₁ * x
        apply_diagonal(x, signs_[0].data());
        fht(x, padded_dim_);

        // D₂ * H * D₁ * x
        apply_diagonal(x, signs_[1].data());
        fht(x, padded_dim_);

        // D₃ * H * D₂ * H * D₁ * x
        apply_diagonal(x, signs_[2].data());
        fht(x, padded_dim_);
    }

    void apply_copy(const float* input, float* output) const override {
        // Copy and zero-pad
        std::memcpy(output, input, original_dim_ * sizeof(float));
        std::memset(output + original_dim_, 0,
                    (padded_dim_ - original_dim_) * sizeof(float));

        apply(output);
    }

    size_t original_dim() const override { return original_dim_; }
    size_t padded_dim() const override { return padded_dim_; }

    /**
     * Get the sign matrices (for serialization/debugging).
     */
    const std::array<std::vector<int8_t>, NUM_LAYERS>& get_signs() const {
        return signs_;
    }

private:
    size_t original_dim_;
    size_t padded_dim_;
    std::array<std::vector<int8_t>, NUM_LAYERS> signs_;

    static size_t next_power_of_two(size_t n) {
        size_t p = 1;
        while (p < n) p *= 2;
        return p;
    }

    void generate_signs(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> coin(0, 1);

        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            signs_[layer].resize(padded_dim_);
            for (size_t i = 0; i < padded_dim_; ++i) {
                signs_[layer][i] = coin(rng) ? 1 : -1;
            }
        }
    }

    void apply_diagonal(float* x, const int8_t* signs) const {
        for (size_t i = 0; i < padded_dim_; ++i) {
            x[i] *= static_cast<float>(signs[i]);
        }
    }
};

// ============================================================================
// RotationChain: Multiple independent rotations for K-bit codes
// ============================================================================

/**
 * RotationChain: Manages K independent rotation instances.
 *
 * Each rotation uses different random seeds to ensure independence.
 * Used by the encoder to generate K independent hash values.
 *
 * @tparam K Number of rotations (code width)
 */
template <size_t K>
class RotationChain {
public:
    /**
     * Construct chain with random Hadamard rotations.
     * @param dim Original vector dimension
     * @param base_seed Base seed (each rotation uses base_seed + rotation_index)
     */
    RotationChain(size_t dim, uint64_t base_seed) {
        rotations_.reserve(K);
        for (size_t r = 0; r < K; ++r) {
            rotations_.push_back(
                std::make_unique<RandomHadamardRotation>(dim, base_seed + r));
        }
    }

    /**
     * Apply rotation r in-place.
     */
    void apply(float* x, size_t rotation_index) const {
        rotations_[rotation_index]->apply(x);
    }

    /**
     * Apply rotation r to a copy.
     */
    void apply_copy(const float* input, float* output, size_t rotation_index) const {
        rotations_[rotation_index]->apply_copy(input, output);
    }

    size_t original_dim() const {
        return rotations_.empty() ? 0 : rotations_[0]->original_dim();
    }

    size_t padded_dim() const {
        return rotations_.empty() ? 0 : rotations_[0]->padded_dim();
    }

    static constexpr size_t num_rotations() { return K; }

    /**
     * Replace a rotation with a custom strategy (for Phase 3 learned rotations).
     */
    void set_rotation(size_t index, std::unique_ptr<RotationStrategy> rotation) {
        if (index < K) {
            rotations_[index] = std::move(rotation);
        }
    }

private:
    std::vector<std::unique_ptr<RotationStrategy>> rotations_;
};

// ============================================================================
// LearnedDiagonalRotation: Phase 3 learned rotation (placeholder)
// ============================================================================

/**
 * LearnedDiagonalRotation: Data-dependent learned rotation (Phase 3).
 *
 * Instead of random ±1 diagonals, uses learned diagonal values from
 * PyTorch training with Gumbel-Softmax relaxation.
 *
 * The structure is the same as RandomHadamardRotation (H D₃ H D₂ H D₁),
 * but the diagonals are continuous values instead of binary.
 *
 * NOTE: This is a placeholder. Full implementation requires loading
 * weights from a trained PyTorch model.
 */
class LearnedDiagonalRotation : public RotationStrategy {
public:
    static constexpr size_t NUM_LAYERS = 3;

    /**
     * Construct with pre-trained diagonal weights.
     * @param dim Original vector dimension
     * @param diagonals Three diagonal matrices (each of length padded_dim)
     */
    LearnedDiagonalRotation(
        size_t dim,
        std::array<std::vector<float>, NUM_LAYERS> diagonals)
        : original_dim_(dim)
        , padded_dim_(next_power_of_two(dim))
        , diagonals_(std::move(diagonals)) {}

    /**
     * Load from file (placeholder).
     */
    static std::unique_ptr<LearnedDiagonalRotation> load(
        const std::string& path, size_t dim);

    void apply(float* x) const override {
        // D₁ * x
        apply_diagonal(x, diagonals_[0].data());
        fht(x, padded_dim_);

        // D₂ * H * D₁ * x
        apply_diagonal(x, diagonals_[1].data());
        fht(x, padded_dim_);

        // D₃ * H * D₂ * H * D₁ * x
        apply_diagonal(x, diagonals_[2].data());
        fht(x, padded_dim_);
    }

    void apply_copy(const float* input, float* output) const override {
        std::memcpy(output, input, original_dim_ * sizeof(float));
        std::memset(output + original_dim_, 0,
                    (padded_dim_ - original_dim_) * sizeof(float));
        apply(output);
    }

    size_t original_dim() const override { return original_dim_; }
    size_t padded_dim() const override { return padded_dim_; }

private:
    size_t original_dim_;
    size_t padded_dim_;
    std::array<std::vector<float>, NUM_LAYERS> diagonals_;

    static size_t next_power_of_two(size_t n) {
        size_t p = 1;
        while (p < n) p *= 2;
        return p;
    }

    void apply_diagonal(float* x, const float* diag) const {
        for (size_t i = 0; i < padded_dim_; ++i) {
            x[i] *= diag[i];
        }
    }
};

}  // namespace cphnsw
