/**
 * RaBitQ Phase 1 Distance Computation Unit Tests
 *
 * Verifies the XOR + PopCount distance optimization:
 * 1. BinaryCode operations (set/get sign bits)
 * 2. RaBitQQuery structure and C1/C2 pre-computation
 * 3. Scalar Hamming distance correctness
 * 4. AVX-512/AVX2 batch kernel correctness
 * 5. Performance: XOR+PopCount vs Gather speedup
 *
 * CRITICAL: Phase 1 replaces expensive SIMD Gather (10-11 cycles)
 * with pure bitwise XOR + PopCount (1-2 cycles).
 *
 * Expected speedup: ~3x on distance computation throughput.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <cmath>

// Include core headers
#include "cphnsw/core/types.hpp"
#include "cphnsw/distance/hamming.hpp"
#include "cphnsw/graph/flat_graph.hpp"
#include "cphnsw/quantizer/cp_encoder.hpp"

using namespace cphnsw;

// Test configuration
constexpr size_t K = 32;       // Code width (32 sign bits = 1 word, 64 = 1 word)
constexpr size_t K64 = 64;     // Also test K=64
constexpr size_t M = 32;       // Neighbors per block
constexpr size_t DIM = 128;    // Vector dimension

// ============================================================================
// Test 1: BinaryCode Operations
// ============================================================================

bool test_binary_code_operations() {
    std::cout << "\n[Test 1] BinaryCode Operations..." << std::endl;

    BinaryCode<32> code32;
    code32.clear();

    // Test clear
    for (size_t w = 0; w < BinaryCode<32>::NUM_WORDS; ++w) {
        if (code32.signs[w] != 0) {
            std::cout << "  FAIL: clear() didn't zero word " << w << std::endl;
            return false;
        }
    }
    std::cout << "  clear(): OK" << std::endl;

    // Test set_sign and get_sign
    code32.set_sign(0, true);   // First bit
    code32.set_sign(7, true);   // 8th bit
    code32.set_sign(31, true);  // Last bit for K=32

    if (!code32.get_sign(0)) {
        std::cout << "  FAIL: sign 0 not set" << std::endl;
        return false;
    }
    if (!code32.get_sign(7)) {
        std::cout << "  FAIL: sign 7 not set" << std::endl;
        return false;
    }
    if (!code32.get_sign(31)) {
        std::cout << "  FAIL: sign 31 not set" << std::endl;
        return false;
    }
    if (code32.get_sign(1)) {
        std::cout << "  FAIL: sign 1 should be unset" << std::endl;
        return false;
    }
    if (code32.get_sign(15)) {
        std::cout << "  FAIL: sign 15 should be unset" << std::endl;
        return false;
    }
    std::cout << "  set_sign/get_sign: OK" << std::endl;

    // Test K=64 (spans 1 word)
    BinaryCode<64> code64;
    code64.clear();
    code64.set_sign(0, true);
    code64.set_sign(32, true);
    code64.set_sign(63, true);

    if (!code64.get_sign(0) || !code64.get_sign(32) || !code64.get_sign(63)) {
        std::cout << "  FAIL: K=64 sign bits not set correctly" << std::endl;
        return false;
    }
    if (code64.get_sign(1) || code64.get_sign(33)) {
        std::cout << "  FAIL: K=64 unexpected sign bits set" << std::endl;
        return false;
    }
    std::cout << "  K=64 operations: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 2: Scalar Hamming Distance
// ============================================================================

bool test_scalar_hamming_distance() {
    std::cout << "\n[Test 2] Scalar Hamming Distance..." << std::endl;

    // Test 1: Identical codes = distance 0
    {
        BinaryCode<32> a, b;
        a.clear();
        b.clear();
        a.set_sign(5, true);
        a.set_sign(10, true);
        b.set_sign(5, true);
        b.set_sign(10, true);

        uint32_t dist = rabitq_hamming_scalar(a, b);
        if (dist != 0) {
            std::cout << "  FAIL: identical codes should have distance 0, got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Identical codes = 0: OK" << std::endl;

    // Test 2: Single bit difference = distance 1
    {
        BinaryCode<32> a, b;
        a.clear();
        b.clear();
        a.set_sign(5, true);  // Only a has bit 5

        uint32_t dist = rabitq_hamming_scalar(a, b);
        if (dist != 1) {
            std::cout << "  FAIL: single bit diff should be 1, got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Single bit diff = 1: OK" << std::endl;

    // Test 3: All bits different (K=32)
    {
        BinaryCode<32> a, b;
        a.signs[0] = 0xFFFFFFFFULL;  // All 32 bits set
        b.signs[0] = 0x00000000ULL;  // No bits set

        uint32_t dist = rabitq_hamming_scalar(a, b);
        if (dist != 32) {
            std::cout << "  FAIL: all 32 bits diff should be 32, got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  All bits diff (K=32) = 32: OK" << std::endl;

    // Test 4: K=64 full difference
    {
        BinaryCode<64> a, b;
        a.signs[0] = 0xFFFFFFFFFFFFFFFFULL;  // All 64 bits set
        b.signs[0] = 0x0000000000000000ULL;

        uint32_t dist = rabitq_hamming_scalar(a, b);
        if (dist != 64) {
            std::cout << "  FAIL: all 64 bits diff should be 64, got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  All bits diff (K=64) = 64: OK" << std::endl;

    // Test 5: Random codes with known popcount
    {
        BinaryCode<32> a, b;
        a.signs[0] = 0x0F0F0F0FULL;  // 16 bits set (4 per byte, 4 bytes)
        b.signs[0] = 0x00000000ULL;

        uint32_t dist = rabitq_hamming_scalar(a, b);
        if (dist != 16) {
            std::cout << "  FAIL: 0x0F0F0F0F has 16 bits, got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Known popcount pattern: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 3: RaBitQ Distance with Pre-computed Scalars
// ============================================================================

bool test_rabitq_distance_formula() {
    std::cout << "\n[Test 3] RaBitQ Distance Formula (C1 + C2 * Hamming)..." << std::endl;

    RaBitQQuery<32> query;
    query.binary.clear();
    query.c1 = 0.0f;      // For normalized vectors, C1 = 0
    query.c2 = 2.0f;      // C2 = 2 * ||q|| * avg_norm / K
    query.query_norm = 1.0f;

    BinaryCode<32> node;
    node.clear();

    // Test 1: Zero Hamming distance
    {
        float dist = rabitq_distance_scalar(query, node);
        float expected = query.c1 + query.c2 * 0.0f;  // 0.0
        if (std::abs(dist - expected) > 1e-6f) {
            std::cout << "  FAIL: zero hamming should give " << expected
                      << ", got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Zero Hamming: OK" << std::endl;

    // Test 2: Hamming = 10
    {
        // Set 10 bits different
        query.binary.signs[0] = 0x3FFULL;  // 10 bits set
        node.clear();

        float dist = rabitq_distance_scalar(query, node);
        float expected = query.c1 + query.c2 * 10.0f;  // 0 + 2*10 = 20
        if (std::abs(dist - expected) > 1e-6f) {
            std::cout << "  FAIL: hamming=10 should give " << expected
                      << ", got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Hamming=10: OK (dist=" << (query.c1 + query.c2 * 10.0f) << ")" << std::endl;

    // Test 3: Different C1, C2 values
    {
        query.binary.clear();
        query.binary.set_sign(0, true);
        query.binary.set_sign(1, true);
        query.binary.set_sign(2, true);  // 3 bits
        node.clear();

        query.c1 = 5.0f;
        query.c2 = 0.5f;

        float dist = rabitq_distance_scalar(query, node);
        float expected = 5.0f + 0.5f * 3.0f;  // 6.5
        if (std::abs(dist - expected) > 1e-6f) {
            std::cout << "  FAIL: c1=5,c2=0.5,h=3 should give " << expected
                      << ", got " << dist << std::endl;
            return false;
        }
    }
    std::cout << "  Custom C1/C2: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 4: NeighborBlock Binary Signs Storage
// ============================================================================

bool test_neighbor_block_signs() {
    std::cout << "\n[Test 4] NeighborBlock Binary Signs Storage..." << std::endl;

    NeighborBlock<uint8_t, 32> block;
    block.count = 8;

    // Set up binary signs for 8 neighbors with known patterns
    for (size_t n = 0; n < 8; ++n) {
        BinaryCode<32> signs;
        signs.clear();
        // Each neighbor has bits [0, n] set
        for (size_t b = 0; b <= n; ++b) {
            signs.set_sign(b, true);
        }
        block.set_neighbor_binary_signs(n, signs);
    }

    // Verify retrieval
    for (size_t n = 0; n < 8; ++n) {
        BinaryCode<32> retrieved = block.get_neighbor_binary_signs(n);

        // Check expected bits are set
        for (size_t b = 0; b <= n; ++b) {
            if (!retrieved.get_sign(b)) {
                std::cout << "  FAIL: neighbor " << n << " bit " << b
                          << " should be set" << std::endl;
                return false;
            }
        }

        // Check unexpected bits are NOT set
        for (size_t b = n + 1; b < 32; ++b) {
            if (retrieved.get_sign(b)) {
                std::cout << "  FAIL: neighbor " << n << " bit " << b
                          << " should NOT be set" << std::endl;
                return false;
            }
        }
    }
    std::cout << "  Set/Get signs: OK" << std::endl;

    // Verify SoA layout for batch access
    const uint64_t (*signs_ptr)[FLASH_MAX_M] = block.get_signs_transposed();
    if (signs_ptr == nullptr) {
        std::cout << "  FAIL: get_signs_transposed returned null" << std::endl;
        return false;
    }

    // For K=32, there's 1 word per neighbor
    // signs_transposed[0][n] should match the pattern we set
    for (size_t n = 0; n < 8; ++n) {
        uint64_t expected = (1ULL << (n + 1)) - 1;  // n+1 bits set
        uint64_t actual = signs_ptr[0][n];
        if (actual != expected) {
            std::cout << "  FAIL: SoA layout mismatch at n=" << n
                      << " expected=0x" << std::hex << expected
                      << " got=0x" << actual << std::dec << std::endl;
            return false;
        }
    }
    std::cout << "  SoA transposed layout: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Test 5: AVX-512/AVX2 Batch Kernel Correctness
// ============================================================================

bool test_batch_kernel_correctness() {
    std::cout << "\n[Test 5] Batch Kernel Correctness (SIMD vs Scalar)..." << std::endl;

    // Set up test data
    std::mt19937 rng(42);

    RaBitQQuery<32> query;
    query.c1 = 0.0f;
    query.c2 = 2.0f / 32.0f;  // Normalized scaling
    query.query_norm = 1.0f;

    // Random query binary code
    query.binary.signs[0] = static_cast<uint64_t>(rng()) |
                            (static_cast<uint64_t>(rng()) << 32);

    NeighborBlock<uint8_t, 32> block;
    block.count = M;

    // Random neighbor binary signs
    for (size_t n = 0; n < M; ++n) {
        BinaryCode<32> signs;
        signs.signs[0] = static_cast<uint64_t>(rng()) |
                         (static_cast<uint64_t>(rng()) << 32);
        block.set_neighbor_binary_signs(n, signs);
        block.ids[n] = static_cast<NodeId>(n);
    }

    // Compute scalar reference
    std::vector<float> scalar_results(M);
    for (size_t n = 0; n < M; ++n) {
        BinaryCode<32> neighbor_signs = block.get_neighbor_binary_signs(n);
        scalar_results[n] = rabitq_distance_scalar(query, neighbor_signs);
    }

    // Compute batch results
    alignas(64) std::vector<float> batch_results(M);

#if CPHNSW_HAS_AVX512
    std::cout << "  Using AVX-512 batch kernel" << std::endl;
    rabitq_hamming_block_avx512<32>(
        query,
        block.get_signs_transposed(),
        M,
        batch_results.data());
#elif CPHNSW_HAS_AVX2
    std::cout << "  Using AVX2 batch kernel" << std::endl;
    // Process with AVX2 batch4 kernel
    alignas(32) uint32_t batch_hamming[4];
    for (size_t n = 0; n + 4 <= M; n += 4) {
        rabitq_hamming_batch4_avx2<32>(
            query.binary.signs,
            block.get_signs_transposed(),
            n,
            batch_hamming);
        for (size_t i = 0; i < 4; ++i) {
            batch_results[n + i] = query.c1 + query.c2 * static_cast<float>(batch_hamming[i]);
        }
    }
    // Scalar remainder
    for (size_t n = (M / 4) * 4; n < M; ++n) {
        BinaryCode<32> signs = block.get_neighbor_binary_signs(n);
        batch_results[n] = rabitq_distance_scalar(query, signs);
    }
#else
    std::cout << "  Using scalar fallback" << std::endl;
    for (size_t n = 0; n < M; ++n) {
        BinaryCode<32> signs = block.get_neighbor_binary_signs(n);
        batch_results[n] = rabitq_distance_scalar(query, signs);
    }
#endif

    // Compare results
    float max_diff = 0.0f;
    bool all_match = true;

    std::cout << "  Neighbor | Scalar      | Batch       | Diff" << std::endl;
    std::cout << "  ---------|-------------|-------------|----------" << std::endl;

    for (size_t n = 0; n < std::min(M, size_t(8)); ++n) {
        float diff = std::abs(scalar_results[n] - batch_results[n]);
        max_diff = std::max(max_diff, diff);

        std::cout << "  " << std::setw(8) << n << " | "
                  << std::setw(11) << std::fixed << std::setprecision(6) << scalar_results[n] << " | "
                  << std::setw(11) << batch_results[n] << " | "
                  << std::setw(10) << diff << std::endl;

        if (diff > 1e-5f) {
            all_match = false;
        }
    }

    // Check remaining silently
    for (size_t n = 8; n < M; ++n) {
        float diff = std::abs(scalar_results[n] - batch_results[n]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-5f) {
            all_match = false;
        }
    }

    if (all_match) {
        std::cout << "  Max diff: " << max_diff << std::endl;
        std::cout << "  PASSED" << std::endl;
        return true;
    } else {
        std::cout << "  FAIL: Batch kernel mismatch! Max diff: " << max_diff << std::endl;
        return false;
    }
}

// ============================================================================
// Test 6: Performance Benchmark - XOR+PopCount vs Gather
// ============================================================================

void benchmark_rabitq_vs_gather() {
    std::cout << "\n[Benchmark] RaBitQ (XOR+PopCount) vs Asymmetric (Gather)..." << std::endl;

    std::mt19937 rng(42);

    // Set up RaBitQ query and data
    RaBitQQuery<K> rabitq_query;
    rabitq_query.c1 = 0.0f;
    rabitq_query.c2 = 2.0f / static_cast<float>(K);
    rabitq_query.query_norm = 1.0f;
    rabitq_query.binary.signs[0] = rng() | (static_cast<uint64_t>(rng()) << 32);

    // Set up asymmetric query
    CPQuery<uint8_t, K> asym_query;
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);
    for (size_t k = 0; k < K; ++k) {
        asym_query.rotated_vecs[k].resize(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            asym_query.rotated_vecs[k][d] = float_dist(rng);
        }
        asym_query.primary_code.components[k] = static_cast<uint8_t>(rng() & 0xFF);
    }

    // Set up neighbor block
    NeighborBlock<uint8_t, K> block;
    block.count = M;
    for (size_t n = 0; n < M; ++n) {
        block.ids[n] = static_cast<NodeId>(n);

        // Set CP codes
        CPCode<uint8_t, K> code;
        for (size_t k = 0; k < K; ++k) {
            code.components[k] = static_cast<uint8_t>(rng() & 0xFF);
            code.magnitudes[k] = static_cast<uint8_t>(rng() & 0xFF);
        }
        block.set_neighbor_code(n, code);

        // Set binary signs
        BinaryCode<K> signs;
        signs.signs[0] = rng() | (static_cast<uint64_t>(rng()) << 32);
        block.set_neighbor_binary_signs(n, signs);
    }

    constexpr size_t iterations = 1000000;
    alignas(64) std::vector<float> results(M);

    // Warm up
    for (int i = 0; i < 100; ++i) {
        for (size_t n = 0; n < M; ++n) {
            BinaryCode<K> signs = block.get_neighbor_binary_signs(n);
            results[n] = rabitq_distance_scalar(rabitq_query, signs);
        }
    }

    // Benchmark RaBitQ scalar
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        asm volatile("" : : "r"(&block) : "memory");
        for (size_t n = 0; n < M; ++n) {
            BinaryCode<K> signs = block.get_neighbor_binary_signs(n);
            results[n] = rabitq_distance_scalar(rabitq_query, signs);
        }
        asm volatile("" : : "r"(results.data()) : "memory");
    }
    auto end = std::chrono::high_resolution_clock::now();
    double rabitq_duration = std::chrono::duration<double>(end - start).count();

    double total_ops = static_cast<double>(iterations) * M;
    double rabitq_mops = (total_ops / rabitq_duration) / 1e6;

    // Benchmark Asymmetric (Gather)
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        asm volatile("" : : "r"(&block) : "memory");
        asymmetric_search_distance_batch_soa<uint8_t, K>(
            asym_query, block.codes_transposed, M, results.data());
        asm volatile("" : : "r"(results.data()) : "memory");
    }
    end = std::chrono::high_resolution_clock::now();
    double gather_duration = std::chrono::duration<double>(end - start).count();

    double gather_mops = (total_ops / gather_duration) / 1e6;

    // Results
    std::cout << "  RaBitQ (XOR+PopCount):" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
              << rabitq_mops << " M comparisons/sec" << std::endl;
    std::cout << "    Latency: " << std::setprecision(2)
              << (rabitq_duration / total_ops) * 1e9 << " ns/neighbor" << std::endl;

    std::cout << "\n  Asymmetric (Gather):" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
              << gather_mops << " M comparisons/sec" << std::endl;
    std::cout << "    Latency: " << std::setprecision(2)
              << (gather_duration / total_ops) * 1e9 << " ns/neighbor" << std::endl;

    double speedup = rabitq_mops / gather_mops;
    std::cout << "\n  Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;

    if (speedup >= 2.5) {
        std::cout << "  Phase 1 target achieved! XOR+PopCount is significantly faster." << std::endl;
    } else if (speedup >= 1.5) {
        std::cout << "  Moderate speedup. Consider enabling AVX-512 VPOPCNTDQ." << std::endl;
    } else {
        std::cout << "  Limited speedup. This is expected for scalar fallback." << std::endl;
        std::cout << "  Note: Full speedup requires AVX-512 batch kernels." << std::endl;
    }
}

// ============================================================================
// Test 7: CPEncoder Binary Encoding
// ============================================================================

bool test_encoder_binary_output() {
    std::cout << "\n[Test 7] CPEncoder Binary Encoding..." << std::endl;

    // Create encoder
    CPEncoder<uint8_t, K> encoder(DIM, 42);

    // Generate test vector
    std::vector<Float> vec(DIM);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t d = 0; d < DIM; ++d) {
        vec[d] = dist(rng);
    }

    // Encode to binary
    std::vector<Float> buffer(encoder.padded_dim());
    BinaryCode<K> binary = encoder.encode_binary_with_buffer(vec.data(), buffer.data());

    // Verify: at least some bits should be set (with high probability for random data)
    uint32_t total_set = 0;
    for (size_t w = 0; w < BinaryCode<K>::NUM_WORDS; ++w) {
#if CPHNSW_HAS_POPCNT
        total_set += __builtin_popcountll(binary.signs[w]);
#else
        uint64_t x = binary.signs[w];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        total_set += static_cast<uint32_t>((x * 0x0101010101010101ULL) >> 56);
#endif
    }

    std::cout << "  Total sign bits set: " << total_set << " / " << K << std::endl;

    // For random data, expect roughly K/2 bits set
    if (total_set == 0 || total_set == K) {
        std::cout << "  WARNING: Unusual bit pattern (all same sign)" << std::endl;
    }

    // Test encode_rabitq_query
    RaBitQQuery<K> query = encoder.encode_rabitq_query_with_buffer(vec.data(), buffer.data());

    // Verify query structure
    if (query.c2 <= 0.0f) {
        std::cout << "  FAIL: c2 should be positive, got " << query.c2 << std::endl;
        return false;
    }

    std::cout << "  RaBitQQuery c1=" << query.c1 << ", c2=" << query.c2
              << ", norm=" << query.query_norm << std::endl;

    // Verify binary codes match
    for (size_t w = 0; w < BinaryCode<K>::NUM_WORDS; ++w) {
        if (binary.signs[w] != query.binary.signs[w]) {
            std::cout << "  FAIL: Binary code mismatch in word " << w << std::endl;
            return false;
        }
    }
    std::cout << "  Binary code consistency: OK" << std::endl;

    std::cout << "  PASSED" << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RaBitQ Phase 1 Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration: K=" << K << ", M=" << M << ", DIM=" << DIM << std::endl;

#if CPHNSW_HAS_AVX512
    std::cout << "SIMD: AVX-512 enabled" << std::endl;
#ifdef __AVX512VPOPCNTDQ__
    std::cout << "PopCount: Native VPOPCNTDQ (Ice Lake+)" << std::endl;
#else
    std::cout << "PopCount: Harley-Seal fallback (Skylake-X)" << std::endl;
#endif
#elif CPHNSW_HAS_AVX2
    std::cout << "SIMD: AVX2 enabled (no AVX-512)" << std::endl;
#else
    std::cout << "SIMD: Scalar fallback" << std::endl;
#endif

    int passed = 0;
    int failed = 0;

    // Core tests
    if (test_binary_code_operations()) passed++; else failed++;
    if (test_scalar_hamming_distance()) passed++; else failed++;
    if (test_rabitq_distance_formula()) passed++; else failed++;
    if (test_neighbor_block_signs()) passed++; else failed++;
    if (test_batch_kernel_correctness()) passed++; else failed++;
    if (test_encoder_binary_output()) passed++; else failed++;

    // Performance benchmark (informational)
    benchmark_rabitq_vs_gather();
    passed++;  // Benchmark always passes

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
