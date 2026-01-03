/**
 * Unified Distance Computation Unit Tests
 *
 * Verifies the Policy-Based Design for both Phase 1 (RaBitQ) and Phase 2 (Residual):
 * 1. ResidualCode<K, R> structure and bit operations
 * 2. CodeQuery<K, R, Shift> with pre-computed scalars
 * 3. UnifiedMetricPolicy<K, R, Shift> distance computation
 * 4. Bit-shift weighting logic for Phase 2
 * 5. Batch distance computation correctness
 *
 * This test replaces the legacy test_rabitq_distance.cpp and validates
 * that the refactoring preserves mathematical correctness.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <cmath>

// Include NEW unified headers only
#include "cphnsw/api/config.hpp"
#include "cphnsw/core/codes.hpp"
#include "cphnsw/distance/metric_policy.hpp"

using namespace cphnsw;

// Test configuration
constexpr size_t K32 = 32;
constexpr size_t K64 = 64;
constexpr size_t R32 = 32;
constexpr size_t R16 = 16;

// ============================================================================
// Test 1: ResidualCode<K, R> Bit Operations
// ============================================================================

bool test_residual_code_operations() {
    std::cout << "\n[Test 1] ResidualCode<K, R> Bit Operations..." << std::endl;

    // Test Phase 1 code (R=0)
    {
        ResidualCode<64, 0> code;
        code.clear();

        // Set some bits
        code.set_primary_bit(0, true);
        code.set_primary_bit(63, true);
        code.set_primary_bit(32, true);

        // Verify
        if (!code.get_primary_bit(0) || !code.get_primary_bit(63) || !code.get_primary_bit(32)) {
            std::cout << "  FAIL: Primary bits not set correctly" << std::endl;
            return false;
        }
        if (code.get_primary_bit(1) || code.get_primary_bit(31)) {
            std::cout << "  FAIL: Unexpected bits set" << std::endl;
            return false;
        }
        std::cout << "  Phase 1 (R=0) primary bits: OK" << std::endl;

        // Verify R=0 has no residual storage (compile-time check)
        static_assert(!ResidualCode<64, 0>::HAS_RESIDUAL, "R=0 should have no residual");
        static_assert(ResidualCode<64, 0>::RESIDUAL_WORDS == 0, "R=0 should have 0 residual words");
        std::cout << "  Phase 1 (R=0) no residual storage: OK" << std::endl;
    }

    // Test Phase 2 code (R>0)
    {
        ResidualCode<64, 32> code;
        code.clear();

        // Set primary bits
        code.set_primary_bit(0, true);
        code.set_primary_bit(63, true);

        // Set residual bits
        code.set_residual_bit(0, true);
        code.set_residual_bit(31, true);

        // Verify primary
        if (!code.get_primary_bit(0) || !code.get_primary_bit(63)) {
            std::cout << "  FAIL: Primary bits not set correctly" << std::endl;
            return false;
        }

        // Verify residual
        if (!code.get_residual_bit(0) || !code.get_residual_bit(31)) {
            std::cout << "  FAIL: Residual bits not set correctly" << std::endl;
            return false;
        }
        if (code.get_residual_bit(1) || code.get_residual_bit(15)) {
            std::cout << "  FAIL: Unexpected residual bits set" << std::endl;
            return false;
        }
        std::cout << "  Phase 2 (R=32) primary + residual bits: OK" << std::endl;

        // Verify R>0 has residual storage
        static_assert(ResidualCode<64, 32>::HAS_RESIDUAL, "R>0 should have residual");
        static_assert(ResidualCode<64, 32>::RESIDUAL_WORDS == 1, "R=32 should have 1 residual word");
        std::cout << "  Phase 2 (R=32) residual storage present: OK" << std::endl;
    }

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 2: Phase 1 (RaBitQ) Distance Computation
// ============================================================================

bool test_phase1_distance() {
    std::cout << "\n[Test 2] Phase 1 (RaBitQ) Distance Computation..." << std::endl;

    using Policy = UnifiedMetricPolicy<64, 0, 0>;  // K=64, R=0
    using CodeType = Policy::CodeType;
    using QueryType = Policy::QueryType;

    // Create two codes with known Hamming distance
    CodeType code_a, code_b;
    code_a.clear();
    code_b.clear();

    // Set 10 bits differently between code_a and code_b
    for (size_t i = 0; i < 10; ++i) {
        code_a.set_primary_bit(i, true);
        // code_b has these bits as 0 (from clear)
    }

    // Expected Hamming distance = 10
    uint32_t hamming = Policy::compute_hamming(code_a, code_b);
    if (hamming != 10) {
        std::cout << "  FAIL: Expected Hamming=10, got " << hamming << std::endl;
        return false;
    }
    std::cout << "  Hamming distance (10 bits differ): " << hamming << " OK" << std::endl;

    // Create a query from code_a
    QueryType query;
    query.code = code_a;
    query.base = 0.0f;
    query.scale = 1.0f;  // distance = hamming directly
    query.query_norm = 1.0f;

    // Distance from query (code_a) to code_b
    float dist = Policy::compute_distance(query, code_b);
    if (std::abs(dist - 10.0f) > 1e-6f) {
        std::cout << "  FAIL: Expected distance=10.0, got " << dist << std::endl;
        return false;
    }
    std::cout << "  Distance computation: " << dist << " OK" << std::endl;

    // Test with scaling
    query.base = 5.0f;
    query.scale = 0.5f;  // distance = 5 + 0.5 * 10 = 10
    dist = Policy::compute_distance(query, code_b);
    float expected = 5.0f + 0.5f * 10.0f;
    if (std::abs(dist - expected) > 1e-6f) {
        std::cout << "  FAIL: Expected distance=" << expected << ", got " << dist << std::endl;
        return false;
    }
    std::cout << "  Distance with base+scale: " << dist << " OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 3: Phase 2 (Residual) Bit-Shift Weighting
// ============================================================================

bool test_phase2_bitshift_weighting() {
    std::cout << "\n[Test 3] Phase 2 Bit-Shift Weighting..." << std::endl;

    // K=64, R=32, Shift=2 (4:1 weighting)
    using Policy = UnifiedMetricPolicy<64, 32, 2>;
    using CodeType = Policy::CodeType;
    using QueryType = Policy::QueryType;

    CodeType code_a, code_b;
    code_a.clear();
    code_b.clear();

    // Scenario 1: Only primary differs by 1 bit
    // Expected: (1 << 2) + 0 = 4
    {
        code_a.set_primary_bit(0, true);
        // code_b primary bit 0 is 0

        uint32_t hamming = Policy::compute_hamming(code_a, code_b);
        if (hamming != 4) {
            std::cout << "  FAIL: 1 primary bit diff, expected combined=4, got " << hamming << std::endl;
            return false;
        }
        std::cout << "  1 primary bit diff -> combined=" << hamming << " OK" << std::endl;

        code_a.set_primary_bit(0, false);  // Reset
    }

    // Scenario 2: Only residual differs by 1 bit
    // Expected: (0 << 2) + 1 = 1
    {
        code_a.set_residual_bit(0, true);
        // code_b residual bit 0 is 0

        uint32_t hamming = Policy::compute_hamming(code_a, code_b);
        if (hamming != 1) {
            std::cout << "  FAIL: 1 residual bit diff, expected combined=1, got " << hamming << std::endl;
            return false;
        }
        std::cout << "  1 residual bit diff -> combined=" << hamming << " OK" << std::endl;

        code_a.set_residual_bit(0, false);  // Reset
    }

    // Scenario 3: 3 primary + 5 residual
    // Expected: (3 << 2) + 5 = 12 + 5 = 17
    {
        code_a.set_primary_bit(0, true);
        code_a.set_primary_bit(1, true);
        code_a.set_primary_bit(2, true);

        code_a.set_residual_bit(0, true);
        code_a.set_residual_bit(1, true);
        code_a.set_residual_bit(2, true);
        code_a.set_residual_bit(3, true);
        code_a.set_residual_bit(4, true);

        uint32_t hamming = Policy::compute_hamming(code_a, code_b);
        uint32_t expected = (3 << 2) + 5;  // 17
        if (hamming != expected) {
            std::cout << "  FAIL: 3 prim + 5 res, expected combined=" << expected
                      << ", got " << hamming << std::endl;
            return false;
        }
        std::cout << "  3 primary + 5 residual -> combined=" << hamming << " OK" << std::endl;
    }

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 4: Batch Distance Computation
// ============================================================================

bool test_batch_distance() {
    std::cout << "\n[Test 4] Batch Distance Computation..." << std::endl;

    using Policy = UnifiedMetricPolicy<64, 0, 0>;  // Phase 1
    using CodeType = Policy::CodeType;
    using QueryType = Policy::QueryType;
    using SoALayout = Policy::SoALayoutType;

    const size_t NUM_NEIGHBORS = 16;

    // Create query
    QueryType query;
    query.code.clear();
    query.code.set_primary_bit(0, true);  // Query has bit 0 set
    query.base = 0.0f;
    query.scale = 1.0f;
    query.query_norm = 1.0f;

    // Create neighbor codes with varying Hamming distances
    std::vector<CodeType> neighbors(NUM_NEIGHBORS);
    std::vector<float> expected_distances(NUM_NEIGHBORS);

    for (size_t n = 0; n < NUM_NEIGHBORS; ++n) {
        neighbors[n].clear();
        // Each neighbor has 'n' bits different from query
        // Query has bit 0 set, so neighbor has bit 0 clear (1 diff)
        // Plus we set bits 1..n in neighbor (n more diffs)
        for (size_t b = 1; b <= n; ++b) {
            neighbors[n].set_primary_bit(b, true);
        }
        expected_distances[n] = static_cast<float>(1 + n);  // 1 from bit 0, n from bits 1..n
    }

    // Populate SoA layout
    SoALayout soa_layout;
    for (size_t n = 0; n < NUM_NEIGHBORS; ++n) {
        for (size_t w = 0; w < CodeType::PRIMARY_WORDS; ++w) {
            soa_layout.primary_transposed[w][n] = neighbors[n].primary.signs[w];
        }
    }

    // Compute batch distances
    std::vector<float> out_distances(NUM_NEIGHBORS);
    Policy::compute_distance_batch(query, soa_layout, NUM_NEIGHBORS, out_distances.data());

    // Verify
    bool all_correct = true;
    for (size_t n = 0; n < NUM_NEIGHBORS; ++n) {
        if (std::abs(out_distances[n] - expected_distances[n]) > 1e-6f) {
            std::cout << "  FAIL: neighbor " << n << " expected=" << expected_distances[n]
                      << " got=" << out_distances[n] << std::endl;
            all_correct = false;
        }
    }

    if (!all_correct) {
        return false;
    }

    std::cout << "  Batch distances match individual computation: OK" << std::endl;

    // Verify batch matches individual computation
    for (size_t n = 0; n < NUM_NEIGHBORS; ++n) {
        float individual = Policy::compute_distance(query, neighbors[n]);
        if (std::abs(out_distances[n] - individual) > 1e-6f) {
            std::cout << "  FAIL: Batch/individual mismatch at n=" << n << std::endl;
            return false;
        }
    }
    std::cout << "  Batch matches individual: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 5: Zero-Overhead Abstraction Verification
// ============================================================================

bool test_zero_overhead_abstraction() {
    std::cout << "\n[Test 5] Zero-Overhead Abstraction (R=0 vs R>0 sizes)..." << std::endl;

    // Verify that ResidualCode<K, 0> has the same size as primary-only storage
    constexpr size_t phase1_size = sizeof(ResidualCode<64, 0>);
    constexpr size_t primary_only_size = sizeof(BinaryCodeStorage<64>);

    std::cout << "  ResidualCode<64, 0> size: " << phase1_size << " bytes" << std::endl;
    std::cout << "  BinaryCodeStorage<64> size: " << primary_only_size << " bytes" << std::endl;

    // They should be very close (may differ due to alignment)
    if (phase1_size > primary_only_size + 8) {
        std::cout << "  WARNING: Phase 1 code larger than expected (padding issue?)" << std::endl;
    }

    // Verify Phase 2 is larger
    constexpr size_t phase2_size = sizeof(ResidualCode<64, 32>);
    std::cout << "  ResidualCode<64, 32> size: " << phase2_size << " bytes" << std::endl;

    if (phase2_size <= phase1_size) {
        std::cout << "  FAIL: Phase 2 should be larger than Phase 1" << std::endl;
        return false;
    }
    std::cout << "  Phase 2 correctly larger than Phase 1: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 6: Type Alias Verification
// ============================================================================

bool test_type_aliases() {
    std::cout << "\n[Test 6] Type Aliases and Policy Traits..." << std::endl;

    // Verify Policy32 is RaBitQ with K=32
    static_assert(Policy32::PRIMARY_BITS == 32, "Policy32 should have K=32");
    static_assert(Policy32::RESIDUAL_BITS == 0, "Policy32 should have R=0");
    static_assert(!Policy32::HAS_RESIDUAL, "Policy32 should not have residual");
    std::cout << "  Policy32: K=32, R=0 OK" << std::endl;

    // Verify Policy64_32 has residual
    static_assert(Policy64_32::PRIMARY_BITS == 64, "Policy64_32 should have K=64");
    static_assert(Policy64_32::RESIDUAL_BITS == 32, "Policy64_32 should have R=32");
    static_assert(Policy64_32::HAS_RESIDUAL, "Policy64_32 should have residual");
    static_assert(Policy64_32::WEIGHT_SHIFT == 2, "Policy64_32 should have Shift=2");
    std::cout << "  Policy64_32: K=64, R=32, Shift=2 OK" << std::endl;

    // Verify MetricPolicyTraits
    static_assert(MetricPolicyTraits<Policy64>::primary_bits == 64, "Traits should expose K");
    static_assert(!MetricPolicyTraits<Policy64>::has_residual, "Traits should expose has_residual");
    std::cout << "  MetricPolicyTraits: OK" << std::endl;

    // Verify is_metric_policy_v
    static_assert(is_metric_policy_v<Policy32>, "Policy32 should be a MetricPolicy");
    static_assert(is_metric_policy_v<Policy64_32>, "Policy64_32 should be a MetricPolicy");
    std::cout << "  is_metric_policy_v: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Unified Distance Computation Tests ===" << std::endl;
    std::cout << "Testing Policy-Based Design for Phase 1 (RaBitQ) and Phase 2 (Residual)" << std::endl;

    int passed = 0;
    int failed = 0;

    auto run_test = [&](const char* name, bool (*test_fn)()) {
        try {
            if (test_fn()) {
                passed++;
            } else {
                failed++;
                std::cout << ">>> TEST FAILED: " << name << std::endl;
            }
        } catch (const std::exception& e) {
            failed++;
            std::cout << ">>> TEST EXCEPTION: " << name << " - " << e.what() << std::endl;
        }
    };

    run_test("ResidualCode operations", test_residual_code_operations);
    run_test("Phase 1 distance", test_phase1_distance);
    run_test("Phase 2 bit-shift weighting", test_phase2_bitshift_weighting);
    run_test("Batch distance", test_batch_distance);
    run_test("Zero-overhead abstraction", test_zero_overhead_abstraction);
    run_test("Type aliases", test_type_aliases);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed == 0) {
        std::cout << "\nAll tests passed! Unified API is working correctly." << std::endl;
        std::cout << "Ready for Phase 3 (Learned Rotations)." << std::endl;
    }

    return failed == 0 ? 0 : 1;
}
