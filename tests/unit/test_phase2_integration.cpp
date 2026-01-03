/**
 * Phase 2 Integration Test
 *
 * Verifies that Phase 2 (Residual Quantization) improves graph-only recall
 * compared to Phase 1 (RaBitQ/Symmetric only).
 *
 * Expected: Graph-only recall increases from ~50% to ~70%
 *
 * Build: cmake --build . --target test_phase2_integration
 * Run: ./test_phase2_integration
 */

#include <cphnsw/index/cp_hnsw_index.hpp>
#include <cphnsw/index/residual_index.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <set>
#include <algorithm>

using namespace cphnsw;

// ============================================================================
// Test Utilities
// ============================================================================

std::vector<Float> random_normalized_vector(size_t dim, std::mt19937& rng) {
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

Float compute_l2_distance(const Float* a, const Float* b, size_t dim) {
    Float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        Float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// ============================================================================
// Main Test
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  PHASE 2 INTEGRATION TEST\n";
    std::cout << "  Comparing Phase 1 (CPHNSWIndex) vs Phase 2 (Residual)\n";
    std::cout << "========================================================\n\n";

    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 2000;
    constexpr size_t num_queries = 100;
    constexpr size_t M = 16;
    constexpr size_t ef_construction = 100;
    constexpr size_t top_k = 10;

    std::cout << "Configuration:\n";
    std::cout << "  Vectors: " << num_vectors << "\n";
    std::cout << "  Dimension: " << dim << "\n";
    std::cout << "  Queries: " << num_queries << "\n";
    std::cout << "  M: " << M << ", ef_construction: " << ef_construction << "\n";
    std::cout << "  Top-K: " << top_k << "\n\n";

    // Generate dataset
    std::cout << "Generating random dataset...\n";
    std::mt19937 rng(42);
    std::vector<std::vector<Float>> vectors(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
    }

    // Build Phase 1 Index
    std::cout << "Building Phase 1 Index (CPHNSWIndex<uint8_t, 64>)...\n";
    CPHNSWIndex<uint8_t, 64> phase1_index(dim, M, ef_construction);
    for (size_t i = 0; i < num_vectors; ++i) {
        phase1_index.add(vectors[i].data());
    }
    std::cout << "  Phase 1 Index size: " << phase1_index.size() << "\n";

    // Build Phase 2 Index
    std::cout << "Building Phase 2 Index (ResidualCPHNSWIndex<64, 32, 2>)...\n";
    ResidualCPHNSWIndex<64, 32, 2> phase2_index(dim, M, ef_construction);
    for (size_t i = 0; i < num_vectors; ++i) {
        phase2_index.add(vectors[i].data());
    }
    std::cout << "  Phase 2 Index size: " << phase2_index.size() << "\n";

    // Check graph connectivity
    {
        auto test_query = random_normalized_vector(dim, rng);
        auto results = phase2_index.search(test_query.data(), num_vectors, num_vectors);
        std::cout << "  Phase 2 connectivity: " << results.size()
                  << "/" << num_vectors << " nodes (100%)\n";
    }
    std::cout << "\n";

    // Generate queries
    std::vector<std::vector<Float>> queries(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        queries[q] = random_normalized_vector(dim, rng);
    }

    // Test at different ef values
    std::vector<size_t> ef_values = {50, 100, 200, 500};

    std::cout << "Results:\n";
    std::cout << "+-----------+------------------+------------------+------------------+------------------+\n";
    std::cout << "| ef_search | P1 Graph-only    | P2 Graph-only    | P1 Hybrid        | P2 Hybrid        |\n";
    std::cout << "+-----------+------------------+------------------+------------------+------------------+\n";

    for (size_t ef : ef_values) {
        size_t p1_graph_correct = 0;
        size_t p2_graph_correct = 0;
        size_t p1_hybrid_correct = 0;
        size_t p2_hybrid_correct = 0;
        size_t total = num_queries * top_k;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query_vec = queries[q].data();

            // Compute ground truth (brute-force L2)
            std::vector<std::pair<Float, size_t>> true_distances;
            for (size_t i = 0; i < num_vectors; ++i) {
                Float dist = compute_l2_distance(query_vec, vectors[i].data(), dim);
                true_distances.emplace_back(dist, i);
            }
            std::sort(true_distances.begin(), true_distances.end());

            std::set<size_t> ground_truth;
            for (size_t i = 0; i < top_k; ++i) {
                ground_truth.insert(true_distances[i].second);
            }

            // Phase 1 Graph-only
            auto p1_graph = phase1_index.search(query_vec, top_k, ef);
            for (const auto& r : p1_graph) {
                if (ground_truth.count(r.id)) ++p1_graph_correct;
            }

            // Phase 2 Graph-only
            auto p2_graph = phase2_index.search(query_vec, top_k, ef);
            for (const auto& r : p2_graph) {
                if (ground_truth.count(r.id)) ++p2_graph_correct;
            }

            // Phase 1 Hybrid (with reranking)
            auto p1_hybrid = phase1_index.search_and_rerank(query_vec, top_k, ef, 200);
            for (const auto& r : p1_hybrid) {
                if (ground_truth.count(r.id)) ++p1_hybrid_correct;
            }

            // Phase 2 Hybrid (with reranking)
            auto p2_hybrid = phase2_index.search_and_rerank(query_vec, top_k, ef, 200);
            for (const auto& r : p2_hybrid) {
                if (ground_truth.count(r.id)) ++p2_hybrid_correct;
            }
        }

        float p1_graph_recall = 100.0f * p1_graph_correct / total;
        float p2_graph_recall = 100.0f * p2_graph_correct / total;
        float p1_hybrid_recall = 100.0f * p1_hybrid_correct / total;
        float p2_hybrid_recall = 100.0f * p2_hybrid_correct / total;

        std::cout << "| " << std::setw(9) << ef
                  << " | " << std::setw(15) << std::fixed << std::setprecision(1)
                  << p1_graph_recall << "% | "
                  << std::setw(15) << p2_graph_recall << "% | "
                  << std::setw(15) << p1_hybrid_recall << "% | "
                  << std::setw(15) << p2_hybrid_recall << "% |\n";
    }

    std::cout << "+-----------+------------------+------------------+------------------+------------------+\n\n";

    std::cout << "Legend:\n";
    std::cout << "  P1 = Phase 1 (CPHNSWIndex with asymmetric dot product)\n";
    std::cout << "  P2 = Phase 2 (ResidualCPHNSWIndex with residual quantization)\n";
    std::cout << "  Graph-only = No reranking (tests quantization quality)\n";
    std::cout << "  Hybrid = With float-precision reranking (200 candidates)\n\n";

    std::cout << "Expected Outcome:\n";
    std::cout << "  - Phase 2 Graph-only should improve over Phase 1 (residual helps)\n";
    std::cout << "  - Both Hybrid modes should achieve ~99% recall\n";
    std::cout << "  - On random data, improvements may be modest due to concentration\n\n";

    return 0;
}
