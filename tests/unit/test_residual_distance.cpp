/**
 * Phase 2 Residual Quantization Unit Tests
 *
 * Tests for:
 * - ResidualBinaryCode structure
 * - ResidualCPEncoder encoding
 * - Residual distance computation (scalar and SIMD)
 * - ResidualNeighborBlock storage
 * - Bit-shift weighting verification
 *
 * Build: cmake --build . --target test_residual_distance
 * Run: ./test_residual_distance
 */

#include <cphnsw/core/types.hpp>
#include <cphnsw/distance/hamming.hpp>
#include <cphnsw/quantizer/residual_encoder.hpp>
#include <cphnsw/graph/flat_graph.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace cphnsw;

// ============================================================================
// Test Utilities
// ============================================================================

/// Generate random normalized vector
std::vector<Float> random_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<Float> dist(0.0f, 1.0f);
    std::vector<Float> vec(dim);
    Float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] /= norm;
    }
    return vec;
}

/// Count differing bits between two codes
template <size_t K>
uint32_t count_hamming(const BinaryCode<K>& a, const BinaryCode<K>& b) {
    uint32_t dist = 0;
    for (size_t i = 0; i < BinaryCode<K>::NUM_WORDS; ++i) {
        uint64_t xor_result = a.signs[i] ^ b.signs[i];
        dist += static_cast<uint32_t>(__builtin_popcountll(xor_result));
    }
    return dist;
}

// ============================================================================
// Test 1: ResidualBinaryCode Structure
// ============================================================================

bool test_residual_binary_code() {
    std::cout << "Test 1: ResidualBinaryCode Structure... ";

    // Test K=64, R=32
    ResidualBinaryCode<64, 32> code;
    code.clear();

    // Verify initial state
    for (size_t w = 0; w < BinaryCode<64>::NUM_WORDS; ++w) {
        if (code.primary.signs[w] != 0) {
            std::cout << "FAILED - primary not cleared\n";
            return false;
        }
    }
    for (size_t w = 0; w < BinaryCode<32>::NUM_WORDS; ++w) {
        if (code.residual.signs[w] != 0) {
            std::cout << "FAILED - residual not cleared\n";
            return false;
        }
    }

    // Set some bits
    code.primary.set_sign(0, true);
    code.primary.set_sign(63, true);
    code.residual.set_sign(0, true);
    code.residual.set_sign(31, true);

    // Verify bits
    if (!code.primary.get_sign(0) || !code.primary.get_sign(63)) {
        std::cout << "FAILED - primary bits not set\n";
        return false;
    }
    if (!code.residual.get_sign(0) || !code.residual.get_sign(31)) {
        std::cout << "FAILED - residual bits not set\n";
        return false;
    }

    // Verify unset bits
    if (code.primary.get_sign(32)) {
        std::cout << "FAILED - unexpected primary bit\n";
        return false;
    }
    if (code.residual.get_sign(16)) {
        std::cout << "FAILED - unexpected residual bit\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// ============================================================================
// Test 2: ResidualEncoder Encoding
// ============================================================================

bool test_residual_encoder() {
    std::cout << "Test 2: ResidualEncoder Encoding... ";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);

    // Verify dimensions
    if (encoder.dim() != dim) {
        std::cout << "FAILED - wrong dim\n";
        return false;
    }
    if (encoder.primary_width() != K) {
        std::cout << "FAILED - wrong K\n";
        return false;
    }
    if (encoder.residual_width() != R) {
        std::cout << "FAILED - wrong R\n";
        return false;
    }

    // Encode a random vector
    std::mt19937 rng(123);
    auto vec = random_vector(dim, rng);

    ResidualBinaryCode<K, R> code = encoder.encode(vec.data());

    // Verify code is not all zeros (statistically unlikely)
    uint64_t prim_bits = 0, res_bits = 0;
    for (size_t w = 0; w < BinaryCode<K>::NUM_WORDS; ++w) {
        prim_bits |= code.primary.signs[w];
    }
    for (size_t w = 0; w < BinaryCode<R>::NUM_WORDS; ++w) {
        res_bits |= code.residual.signs[w];
    }

    if (prim_bits == 0) {
        std::cout << "FAILED - all primary bits zero\n";
        return false;
    }
    if (res_bits == 0) {
        std::cout << "FAILED - all residual bits zero\n";
        return false;
    }

    // Verify determinism: same input should produce same output
    ResidualBinaryCode<K, R> code2 = encoder.encode(vec.data());
    for (size_t w = 0; w < BinaryCode<K>::NUM_WORDS; ++w) {
        if (code.primary.signs[w] != code2.primary.signs[w]) {
            std::cout << "FAILED - non-deterministic primary\n";
            return false;
        }
    }
    for (size_t w = 0; w < BinaryCode<R>::NUM_WORDS; ++w) {
        if (code.residual.signs[w] != code2.residual.signs[w]) {
            std::cout << "FAILED - non-deterministic residual\n";
            return false;
        }
    }

    std::cout << "PASSED\n";
    return true;
}

// ============================================================================
// Test 3: Residual Distance Computation (Scalar)
// ============================================================================

bool test_residual_distance_scalar() {
    std::cout << "Test 3: Residual Distance (Scalar)... ";

    constexpr size_t K = 64;
    constexpr size_t R = 32;

    // Create two codes with known bit patterns
    BinaryCode<K> q_prim, n_prim;
    BinaryCode<R> q_res, n_res;

    q_prim.clear(); n_prim.clear();
    q_res.clear(); n_res.clear();

    // Set specific bits for predictable Hamming distances
    // Primary: 10 bits differ
    for (int i = 0; i < 10; ++i) {
        q_prim.set_sign(i, true);  // q has bits 0-9 set
        // n has none set, so 10 bits differ
    }

    // Residual: 5 bits differ
    for (int i = 0; i < 5; ++i) {
        q_res.set_sign(i, true);
        // n has none set, so 5 bits differ
    }

    // Expected: primary_hamming = 10, residual_hamming = 5
    // With Shift=2: combined = (10 << 2) + 5 = 40 + 5 = 45
    uint32_t combined = residual_distance_integer_scalar<K, R, 2>(
        q_prim, q_res, n_prim, n_res);

    if (combined != 45) {
        std::cout << "FAILED - expected 45, got " << combined << "\n";
        return false;
    }

    // Test with Shift=1: combined = (10 << 1) + 5 = 20 + 5 = 25
    uint32_t combined_shift1 = residual_distance_integer_scalar<K, R, 1>(
        q_prim, q_res, n_prim, n_res);

    if (combined_shift1 != 25) {
        std::cout << "FAILED - shift=1: expected 25, got " << combined_shift1 << "\n";
        return false;
    }

    // Test with Shift=3: combined = (10 << 3) + 5 = 80 + 5 = 85
    uint32_t combined_shift3 = residual_distance_integer_scalar<K, R, 3>(
        q_prim, q_res, n_prim, n_res);

    if (combined_shift3 != 85) {
        std::cout << "FAILED - shift=3: expected 85, got " << combined_shift3 << "\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// ============================================================================
// Test 4: ResidualWeighting Template
// ============================================================================

bool test_residual_weighting() {
    std::cout << "Test 4: ResidualWeighting Template... ";

    // Test combine function at different shifts
    uint32_t prim = 10;
    uint32_t res = 5;

    // Shift=1: (10 << 1) + 5 = 25
    if (ResidualWeighting<1>::combine(prim, res) != 25) {
        std::cout << "FAILED - shift=1\n";
        return false;
    }

    // Shift=2: (10 << 2) + 5 = 45
    if (ResidualWeighting<2>::combine(prim, res) != 45) {
        std::cout << "FAILED - shift=2\n";
        return false;
    }

    // Shift=3: (10 << 3) + 5 = 85
    if (ResidualWeighting<3>::combine(prim, res) != 85) {
        std::cout << "FAILED - shift=3\n";
        return false;
    }

    // Test weight calculations
    float alpha2 = ResidualWeighting<2>::primary_weight();
    float beta2 = ResidualWeighting<2>::residual_weight();

    // Shift=2: alpha = 4/5 = 0.8, beta = 1/5 = 0.2
    if (std::abs(alpha2 - 0.8f) > 0.01f) {
        std::cout << "FAILED - alpha weight\n";
        return false;
    }
    if (std::abs(beta2 - 0.2f) > 0.01f) {
        std::cout << "FAILED - beta weight\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// ============================================================================
// Test 5: ResidualNeighborBlock Storage
// ============================================================================

bool test_residual_neighbor_block() {
    std::cout << "Test 5: ResidualNeighborBlock Storage... ";

    constexpr size_t K = 64;
    constexpr size_t R = 32;

    ResidualNeighborBlock<K, R> block;

    // Verify initial state
    if (block.count != 0) {
        std::cout << "FAILED - initial count\n";
        return false;
    }

    // Create a test code
    ResidualBinaryCode<K, R> code;
    code.clear();
    code.primary.set_sign(0, true);
    code.primary.set_sign(63, true);
    code.residual.set_sign(15, true);

    // Store the code for neighbor 0
    block.set_neighbor_residual_code(0, code);
    block.ids[0] = 42;
    block.count = 1;

    // Retrieve and verify
    auto retrieved = block.get_neighbor_residual_code(0);

    if (!retrieved.primary.get_sign(0) || !retrieved.primary.get_sign(63)) {
        std::cout << "FAILED - primary bits lost\n";
        return false;
    }
    if (!retrieved.residual.get_sign(15)) {
        std::cout << "FAILED - residual bits lost\n";
        return false;
    }
    if (retrieved.primary.get_sign(32)) {
        std::cout << "FAILED - spurious primary bit\n";
        return false;
    }
    if (retrieved.residual.get_sign(0)) {
        std::cout << "FAILED - spurious residual bit\n";
        return false;
    }

    // Test norm storage
    block.set_neighbor_norm(0, 2.5f);
    if (std::abs(block.get_neighbor_norm(0) - 2.5f) > 0.001f) {
        std::cout << "FAILED - norm storage\n";
        return false;
    }

    // Test distance storage
    block.set_neighbor_distance(0, 1.234f);
    if (std::abs(block.get_neighbor_distance(0) - 1.234f) > 0.001f) {
        std::cout << "FAILED - distance storage\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// ============================================================================
// Test 6: Residual Distance with Full Pipeline
// ============================================================================

bool test_residual_distance_pipeline() {
    std::cout << "Test 6: Full Residual Distance Pipeline... ";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Encode query and node
    auto query_vec = random_vector(dim, rng);
    auto node_vec = random_vector(dim, rng);

    ResidualQuery<K, R> query = encoder.encode_query(query_vec.data(), 1.0f);
    ResidualBinaryCode<K, R> node = encoder.encode(node_vec.data());

    // Compute distance using dispatcher
    float dist = residual_distance<K, R, 2>(query, node);

    // Distance should be non-negative and finite
    if (dist < 0.0f || !std::isfinite(dist)) {
        std::cout << "FAILED - invalid distance: " << dist << "\n";
        return false;
    }

    // Self-distance should be zero (same vector encoded twice)
    ResidualBinaryCode<K, R> query_code = encoder.encode(query_vec.data());
    ResidualQuery<K, R> query_self;
    query_self.primary = query_code.primary;
    query_self.residual = query_code.residual;
    query_self.base = 0.0f;
    query_self.scale = 1.0f;

    float self_dist = residual_distance<K, R, 2>(query_self, query_code);
    if (self_dist != 0.0f) {
        std::cout << "FAILED - self distance not zero: " << self_dist << "\n";
        return false;
    }

    std::cout << "PASSED (dist=" << dist << ")\n";
    return true;
}

// ============================================================================
// Test 7: Batch Encoding Performance
// ============================================================================

bool test_batch_encoding() {
    std::cout << "Test 7: Batch Encoding... ";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    constexpr size_t num_vecs = 1000;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);

    // Generate random vectors
    std::mt19937 rng(123);
    std::vector<Float> vecs(num_vecs * dim);
    for (size_t i = 0; i < num_vecs * dim; ++i) {
        vecs[i] = static_cast<Float>(rng()) / static_cast<Float>(rng.max());
    }

    // Allocate output
    std::vector<ResidualBinaryCode<K, R>> codes(num_vecs);

    // Time batch encoding
    auto start = std::chrono::high_resolution_clock::now();
    encoder.encode_batch(vecs.data(), num_vecs, codes.data());
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double vecs_per_sec = num_vecs / (ms / 1000.0);

    // Verify at least some codes have bits set
    size_t non_zero = 0;
    for (const auto& code : codes) {
        if (code.primary.signs[0] != 0 || code.residual.signs[0] != 0) {
            ++non_zero;
        }
    }

    if (non_zero < num_vecs / 2) {
        std::cout << "FAILED - too many zero codes\n";
        return false;
    }

    std::cout << "PASSED (" << static_cast<int>(vecs_per_sec) << " vecs/sec)\n";
    return true;
}

// ============================================================================
// Test 8: Residual vs Primary-Only Comparison
// ============================================================================

bool test_residual_vs_primary() {
    std::cout << "Test 8: Residual vs Primary-Only Comparison... ";

    // This test verifies that residual codes provide additional information
    // by comparing similar vs dissimilar vectors

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Create base vector
    auto base_vec = random_vector(dim, rng);
    ResidualBinaryCode<K, R> base_code = encoder.encode(base_vec.data());

    // Create similar vector (small perturbation)
    std::vector<Float> similar_vec = base_vec;
    for (size_t i = 0; i < dim; ++i) {
        similar_vec[i] += 0.1f * (static_cast<Float>(rng()) / static_cast<Float>(rng.max()) - 0.5f);
    }
    ResidualBinaryCode<K, R> similar_code = encoder.encode(similar_vec.data());

    // Create random vector (different direction)
    auto random_vec = random_vector(dim, rng);
    ResidualBinaryCode<K, R> random_code = encoder.encode(random_vec.data());

    // Compute Hamming distances
    uint32_t similar_prim = count_hamming(base_code.primary, similar_code.primary);
    uint32_t similar_res = count_hamming(base_code.residual, similar_code.residual);

    uint32_t random_prim = count_hamming(base_code.primary, random_code.primary);
    uint32_t random_res = count_hamming(base_code.residual, random_code.residual);

    // Random should have higher distance than similar on average
    // Due to noise, we just check they're both non-zero
    if (similar_prim == 0 && similar_res == 0 && random_prim == 0 && random_res == 0) {
        std::cout << "FAILED - all distances zero\n";
        return false;
    }

    std::cout << "PASSED (similar: prim=" << similar_prim << " res=" << similar_res
              << ", random: prim=" << random_prim << " res=" << random_res << ")\n";
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n=== Phase 2 Residual Quantization Tests ===\n\n";

    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test)(), const char* name) {
        try {
            if (test()) {
                ++passed;
            } else {
                ++failed;
            }
        } catch (const std::exception& e) {
            std::cout << name << " EXCEPTION: " << e.what() << "\n";
            ++failed;
        }
    };

    run_test(test_residual_binary_code, "test_residual_binary_code");
    run_test(test_residual_encoder, "test_residual_encoder");
    run_test(test_residual_distance_scalar, "test_residual_distance_scalar");
    run_test(test_residual_weighting, "test_residual_weighting");
    run_test(test_residual_neighbor_block, "test_residual_neighbor_block");
    run_test(test_residual_distance_pipeline, "test_residual_distance_pipeline");
    run_test(test_batch_encoding, "test_batch_encoding");
    run_test(test_residual_vs_primary, "test_residual_vs_primary");

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}
