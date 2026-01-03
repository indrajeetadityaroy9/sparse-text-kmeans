#include <gtest/gtest.h>
#include "cphnsw/encoder/transform/fht.hpp"
#include <vector>
#include <cmath>
#include <random>

using namespace cphnsw;
using Float = float;  // Compatibility alias

// Test that FHT is its own inverse (up to scaling)
TEST(HadamardTest, SelfInverse) {
    std::vector<Float> vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<Float> original = vec;

    // Apply FHT twice
    fht(vec.data(), vec.size());
    fht(vec.data(), vec.size());

    // Result should be original * n
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_NEAR(vec[i], original[i] * vec.size(), 1e-4f);
    }
}

// Test FHT correctness against naive implementation
TEST(HadamardTest, CorrectnessSmall) {
    // For n=4, Hadamard matrix is:
    // [1  1  1  1]
    // [1 -1  1 -1]
    // [1  1 -1 -1]
    // [1 -1 -1  1]

    std::vector<Float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    fht(vec.data(), vec.size());

    // Expected: H * [1,2,3,4]^T
    std::vector<Float> expected = {
        1 + 2 + 3 + 4,   // 10
        1 - 2 + 3 - 4,   // -2
        1 + 2 - 3 - 4,   // -4
        1 - 2 - 3 + 4    // 0
    };

    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_NEAR(vec[i], expected[i], 1e-5f);
    }
}

// Test FHT preserves L2 norm (up to scaling)
TEST(HadamardTest, NormPreservation) {
    std::mt19937_64 rng(42);
    std::normal_distribution<Float> dist(0.0f, 1.0f);

    for (size_t len : {8, 16, 32, 64, 128, 256}) {
        std::vector<Float> vec(len);
        Float norm_before = 0.0f;

        for (size_t i = 0; i < len; ++i) {
            vec[i] = dist(rng);
            norm_before += vec[i] * vec[i];
        }
        norm_before = std::sqrt(norm_before);

        fht(vec.data(), len);

        Float norm_after = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            norm_after += vec[i] * vec[i];
        }
        norm_after = std::sqrt(norm_after);

        // After FHT, norm should be scaled by sqrt(n)
        Float expected_norm = norm_before * std::sqrt(static_cast<Float>(len));
        EXPECT_NEAR(norm_after, expected_norm, expected_norm * 1e-4f);
    }
}

// Test that scalar and SIMD implementations match
#if CPHNSW_HAS_AVX2
TEST(HadamardTest, ScalarAVX2Match) {
    std::mt19937_64 rng(42);
    std::normal_distribution<Float> dist(0.0f, 1.0f);

    for (size_t len : {8, 16, 32, 64, 128}) {
        std::vector<Float> vec_scalar(len);
        std::vector<Float> vec_avx2(len);

        for (size_t i = 0; i < len; ++i) {
            Float val = dist(rng);
            vec_scalar[i] = val;
            vec_avx2[i] = val;
        }

        fht_scalar(vec_scalar.data(), len);
        fht_avx2(vec_avx2.data(), len);

        for (size_t i = 0; i < len; ++i) {
            EXPECT_NEAR(vec_scalar[i], vec_avx2[i], 1e-4f)
                << "Mismatch at index " << i << " for len=" << len;
        }
    }
}
#endif
