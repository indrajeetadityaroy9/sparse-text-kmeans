#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <set>

using namespace cphnsw;

// Generate random unit vectors on sphere
void generate_random_sphere(std::vector<Float>& data, size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<Float> normal(0.0f, 1.0f);

    data.resize(n * dim);
    for (size_t i = 0; i < n; ++i) {
        Float norm = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            Float val = normal(rng);
            data[i * dim + d] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < dim; ++d) {
            data[i * dim + d] /= norm;
        }
    }
}

// Test tiered construction with different backbone sizes
void test_tiered_construction(size_t N, size_t dim, size_t backbone_size) {
    std::cout << "\n=== Testing backbone_size = " << backbone_size << " ===" << std::endl;

    // Generate data
    std::vector<Float> data;
    generate_random_sphere(data, N, dim, 42);

    // Create index with specified backbone size
    CPHNSWIndex8 index(dim, 16, 100);
    index.set_backbone_size(backbone_size);

    // Build index and time it
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i) {
        index.add(&data[i * dim]);

        // Progress every 2000
        if ((i + 1) % 2000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            std::cout << "  Inserted " << (i + 1) << "/" << N
                      << " (" << std::fixed << std::setprecision(1)
                      << (i + 1) / elapsed << " vec/s)" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end - start).count();

    // Check connectivity
    size_t connected = index.verify_connectivity();

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Build time: " << std::fixed << std::setprecision(2)
              << build_time << " s" << std::endl;
    std::cout << "  Build rate: " << std::setprecision(0)
              << N / build_time << " vectors/s" << std::endl;
    std::cout << "  Connectivity: " << connected << "/" << N;
    if (connected == N) {
        std::cout << " (100% CONNECTED)" << std::endl;
    } else {
        std::cout << " (" << std::setprecision(2)
                  << (100.0 * connected / N) << "% connected)" << std::endl;
    }

    // Test search quality using brute force as ground truth
    std::cout << "\nSearch quality test (10 random queries, k=10):" << std::endl;

    std::vector<Float> queries;
    generate_random_sphere(queries, 10, dim, 123);

    size_t total_correct = 0;
    for (size_t q = 0; q < 10; ++q) {
        // Brute force ground truth
        auto bf_results = index.brute_force_search(&queries[q * dim], 10);

        // HNSW search
        auto hnsw_results = index.search(&queries[q * dim], 10, 100);

        // Count overlap
        std::set<NodeId> gt_set;
        for (const auto& r : bf_results) gt_set.insert(r.id);

        size_t correct = 0;
        for (const auto& r : hnsw_results) {
            if (gt_set.count(r.id)) ++correct;
        }
        total_correct += correct;
    }

    std::cout << "  Recall@10 (vs brute force CP): "
              << std::setprecision(1) << (100.0 * total_correct / 100) << "%" << std::endl;
}

int main() {
    std::cout << "CP-HNSW Tiered Construction Test" << std::endl;
    std::cout << "=================================" << std::endl;

    const size_t N = 10000;
    const size_t dim = 128;

    // Test different backbone sizes
    test_tiered_construction(N, dim, 0);       // No backbone (pure CP)
    test_tiered_construction(N, dim, 1000);    // Small backbone
    test_tiered_construction(N, dim, 5000);    // Medium backbone
    test_tiered_construction(N, dim, 10000);   // Full backbone (all float)

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "backbone=0: Fastest but may have connectivity issues" << std::endl;
    std::cout << "backbone=1000-5000: Good tradeoff" << std::endl;
    std::cout << "backbone=N: 100% connected but O(n²) build" << std::endl;

    return 0;
}
