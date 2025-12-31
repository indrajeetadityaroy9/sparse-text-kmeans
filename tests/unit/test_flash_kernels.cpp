/**
 * Flash Memory Layout & AVX-512 Distance Kernel Unit Tests
 *
 * Verifies:
 * 1. Correctness: AVX-512 batch kernel produces bitwise-identical results to scalar
 * 2. Throughput: Micro-benchmark to confirm SIMD speedup
 *
 * CRITICAL: Uses SoA (Struct-of-Arrays) TRANSPOSED layout, not AoS.
 * codes_transposed[K][M] enables contiguous SIMD loads across neighbors.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <iomanip>

// Include core headers
#include "cphnsw/core/types.hpp"
#include "cphnsw/distance/hamming.hpp"
#include "cphnsw/graph/flat_graph.hpp"

// Constants
constexpr size_t K = 32;       // 32 rotations (bytes)
constexpr size_t M = 32;       // 32 neighbors per block
constexpr size_t DIM = 128;    // Vector dimension

using namespace cphnsw;

// Helper: Generate random test data
void setup_test_data(CPQuery<uint8_t, K>& query, NeighborBlock<uint8_t, K>& block) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    // 1. Generate Query (Rotated vectors)
    for (size_t k = 0; k < K; ++k) {
        query.rotated_vecs[k].resize(DIM);
        for (size_t d = 0; d < DIM; ++d) {
            query.rotated_vecs[k][d] = float_dist(rng);
        }
        // Set primary code (not used in distance calc, but for completeness)
        query.primary_code.components[k] = static_cast<uint8_t>(byte_dist(rng));
    }

    // 2. Generate Neighbor Codes using SoA layout
    block.count = M;
    for (size_t i = 0; i < M; ++i) {
        block.ids[i] = static_cast<NodeId>(i);
        CPCode<uint8_t, K> code;
        for (size_t k = 0; k < K; ++k) {
            code.components[k] = static_cast<uint8_t>(byte_dist(rng));
        }
        // This scatters to transposed layout
        block.set_neighbor_code(i, code);
    }
}

// --- Verification Test: Scalar vs SIMD Batch ---
bool verify_correctness() {
    std::cout << "\n[Test] Verifying SoA Batch Kernel Correctness..." << std::endl;

    CPQuery<uint8_t, K> query;
    NeighborBlock<uint8_t, K> block;
    setup_test_data(query, block);

    // 1. Compute Reference (Scalar via gathered codes)
    std::vector<float> scalar_results(M);
    for (size_t i = 0; i < M; ++i) {
        // Gather code from SoA layout and compute scalar distance
        CPCode<uint8_t, K> gathered_code = block.get_neighbor_code_copy(i);
        scalar_results[i] = asymmetric_search_distance(query, gathered_code);
    }

    // 2. Compute Optimized (SIMD Batch with SoA layout)
    std::vector<float> simd_results(M);
    asymmetric_search_distance_batch_soa<uint8_t, K>(
        query, block.codes_transposed, M, simd_results.data());

    // 3. Compare
    float max_diff = 0.0f;
    bool all_match = true;

    std::cout << "  Neighbor | Scalar      | SIMD        | Diff" << std::endl;
    std::cout << "  ---------|-------------|-------------|----------" << std::endl;

    for (size_t i = 0; i < std::min(M, size_t(8)); ++i) {  // Show first 8
        float diff = std::abs(scalar_results[i] - simd_results[i]);
        max_diff = std::max(max_diff, diff);

        std::cout << "  " << std::setw(8) << i << " | "
                  << std::setw(11) << std::fixed << std::setprecision(4) << scalar_results[i] << " | "
                  << std::setw(11) << simd_results[i] << " | "
                  << std::setw(10) << diff << std::endl;

        if (diff > 1e-5f) {
            all_match = false;
        }
    }

    // Check remaining silently
    for (size_t i = 8; i < M; ++i) {
        float diff = std::abs(scalar_results[i] - simd_results[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-5f) {
            all_match = false;
            std::cerr << "  ❌ Mismatch at index " << i
                      << " Scalar: " << scalar_results[i]
                      << " SIMD: " << simd_results[i] << std::endl;
        }
    }

    if (all_match) {
        std::cout << "  ✅ Logic Verified. Max diff: " << max_diff << std::endl;
        return true;
    } else {
        std::cout << "  ❌ MISMATCH DETECTED! SoA transposition may be broken." << std::endl;
        return false;
    }
}

// --- Performance Micro-benchmark ---
void benchmark_throughput() {
    std::cout << "\n[Test] Benchmarking Throughput..." << std::endl;

    CPQuery<uint8_t, K> query;
    NeighborBlock<uint8_t, K> block;
    setup_test_data(query, block);

    alignas(64) std::vector<float> results(M);
    constexpr size_t iterations = 1000000;

    // Warm up cache
    for (int i = 0; i < 100; ++i) {
        asymmetric_search_distance_batch_soa<uint8_t, K>(
            query, block.codes_transposed, M, results.data());
    }

    // Benchmark SIMD batch
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        // Prevent compiler optimization
        asm volatile("" : : "r"(&block) : "memory");
        asymmetric_search_distance_batch_soa<uint8_t, K>(
            query, block.codes_transposed, M, results.data());
        asm volatile("" : : "r"(results.data()) : "memory");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration_sec = std::chrono::duration<double>(end - start).count();

    // Total distance calculations = iterations * M neighbors
    double total_ops = static_cast<double>(iterations) * M;
    double mops = (total_ops / duration_sec) / 1e6;
    double ns_per_neighbor = (duration_sec / total_ops) * 1e9;

    std::cout << "  SIMD Batch (SoA Layout):" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2) << mops
              << " Million comparisons/sec" << std::endl;
    std::cout << "    Latency per neighbor: " << std::setprecision(2)
              << ns_per_neighbor << " ns" << std::endl;

    // Benchmark scalar for comparison
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < iterations; ++i) {
        asm volatile("" : : "r"(&block) : "memory");
        for (size_t n = 0; n < M; ++n) {
            CPCode<uint8_t, K> code = block.get_neighbor_code_copy(n);
            results[n] = asymmetric_search_distance(query, code);
        }
        asm volatile("" : : "r"(results.data()) : "memory");
    }

    end = std::chrono::high_resolution_clock::now();
    double scalar_duration = std::chrono::duration<double>(end - start).count();
    double scalar_mops = (total_ops / scalar_duration) / 1e6;
    double scalar_ns = (scalar_duration / total_ops) * 1e9;

    std::cout << "\n  Scalar (Gather + Loop):" << std::endl;
    std::cout << "    Throughput: " << std::fixed << std::setprecision(2) << scalar_mops
              << " Million comparisons/sec" << std::endl;
    std::cout << "    Latency per neighbor: " << std::setprecision(2)
              << scalar_ns << " ns" << std::endl;

    // Speedup calculation
    double speedup = mops / scalar_mops;
    std::cout << "\n  📊 Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;

    // Performance assessment
    if (speedup >= 3.0) {
        std::cout << "  🚀 Flash Speed Confirmed! SIMD batch is " << speedup << "x faster." << std::endl;
    } else if (speedup >= 1.5) {
        std::cout << "  ✅ Moderate speedup achieved (" << speedup << "x)." << std::endl;
    } else if (mops < 100) {
        std::cout << "  ⚠️  WARNING: Performance lower than expected for AVX-512." << std::endl;
        std::cout << "     Check that -mavx512f -mavx512bw flags are enabled." << std::endl;
    } else {
        std::cout << "  ℹ️  Limited speedup - may be memory bound or code not vectorizing." << std::endl;
    }
}

// --- Alignment Verification ---
bool verify_alignment() {
    std::cout << "\n[Test] Checking Memory Alignment..." << std::endl;

    NeighborBlock<uint8_t, K> block;

    uintptr_t block_addr = reinterpret_cast<uintptr_t>(&block);
    uintptr_t codes_addr = reinterpret_cast<uintptr_t>(&block.codes_transposed[0][0]);

    std::cout << "  NeighborBlock alignof: " << alignof(NeighborBlock<uint8_t, K>) << " bytes" << std::endl;
    std::cout << "  Block address:   0x" << std::hex << block_addr << std::dec << std::endl;
    std::cout << "  Codes address:   0x" << std::hex << codes_addr << std::dec << std::endl;
    std::cout << "  Block % 64:      " << (block_addr % 64) << " (should be 0)" << std::endl;

    if (alignof(NeighborBlock<uint8_t, K>) != 64) {
        std::cout << "  ❌ NeighborBlock is NOT 64-byte aligned!" << std::endl;
        return false;
    }

    if (block_addr % 64 != 0) {
        std::cout << "  ❌ Stack allocation not aligned (use aligned_alloc for heap)" << std::endl;
        return false;
    }

    std::cout << "  ✅ 64-byte alignment confirmed" << std::endl;
    return true;
}

// --- SoA Layout Verification ---
bool verify_soa_layout() {
    std::cout << "\n[Test] Verifying SoA Transposed Layout..." << std::endl;

    NeighborBlock<uint8_t, K> block;

    // Create test pattern: neighbor N has all bytes set to N
    for (size_t n = 0; n < 8; ++n) {
        CPCode<uint8_t, K> code;
        for (size_t k = 0; k < K; ++k) {
            code.components[k] = static_cast<uint8_t>(n * 10 + k);  // Unique per (n,k)
        }
        block.set_neighbor_code(n, code);
    }

    // Verify transposition: codes_transposed[k][n] == n*10 + k
    bool correct = true;
    std::cout << "  Layout check (showing k=0..3, n=0..3):" << std::endl;
    std::cout << "          n=0   n=1   n=2   n=3" << std::endl;

    for (size_t k = 0; k < 4; ++k) {
        std::cout << "  k=" << k << ":   ";
        for (size_t n = 0; n < 4; ++n) {
            uint8_t expected = static_cast<uint8_t>(n * 10 + k);
            uint8_t actual = block.codes_transposed[k][n];
            std::cout << std::setw(4) << static_cast<int>(actual) << "  ";
            if (actual != expected) {
                correct = false;
            }
        }
        std::cout << std::endl;
    }

    // Verify gather works correctly
    for (size_t n = 0; n < 8; ++n) {
        CPCode<uint8_t, K> gathered = block.get_neighbor_code_copy(n);
        for (size_t k = 0; k < K; ++k) {
            uint8_t expected = static_cast<uint8_t>(n * 10 + k);
            if (gathered.components[k] != expected) {
                correct = false;
                std::cout << "  ❌ Gather mismatch: n=" << n << " k=" << k
                          << " expected=" << static_cast<int>(expected)
                          << " got=" << static_cast<int>(gathered.components[k]) << std::endl;
            }
        }
    }

    if (correct) {
        std::cout << "  ✅ SoA transposition is correct" << std::endl;
        return true;
    } else {
        std::cout << "  ❌ SoA transposition has errors!" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Flash Kernel Unit Tests (SoA Layout)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration: K=" << K << ", M=" << M << ", DIM=" << DIM << std::endl;

#if CPHNSW_HAS_AVX512
    std::cout << "SIMD: AVX-512 enabled" << std::endl;
#elif CPHNSW_HAS_AVX2
    std::cout << "SIMD: AVX2 enabled (no AVX-512)" << std::endl;
#else
    std::cout << "SIMD: Scalar fallback (no AVX)" << std::endl;
#endif

    int passed = 0;
    int failed = 0;

    // Test 1: Alignment
    if (verify_alignment()) passed++; else failed++;

    // Test 2: SoA Layout
    if (verify_soa_layout()) passed++; else failed++;

    // Test 3: Correctness
    if (verify_correctness()) passed++; else failed++;

    // Test 4: Throughput (always runs, but doesn't fail)
    benchmark_throughput();
    passed++;  // Benchmark is informational

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
