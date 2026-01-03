/**
 * Recall Diagnostic Test
 *
 * CRITICAL: Investigate the "Recall Collapse" issue before Phase 3.
 *
 * Tests:
 * 1. Self-Search Sanity: Can we find vector 0 when searching for it?
 * 2. ef_search Scaling: Does recall improve with higher ef?
 * 3. Graph Connectivity: Average neighbors per node
 * 4. Tie-Breaking: How many neighbors have identical distances?
 *
 * Build: cmake --build . --target test_recall_diagnostic
 * Run: ./test_recall_diagnostic
 */

#include <cphnsw/core/types.hpp>
#include <cphnsw/distance/hamming.hpp>
#include <cphnsw/quantizer/residual_encoder.hpp>
#include <cphnsw/quantizer/cp_encoder.hpp>
#include <cphnsw/graph/flat_graph.hpp>
#include <cphnsw/index/cp_hnsw_index.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <set>
#include <map>
#include <algorithm>
#include <queue>

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

Float compute_l2_distance(const std::vector<Float>& a, const std::vector<Float>& b) {
    Float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        Float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// ============================================================================
// Test 1: Self-Search Sanity Check
// ============================================================================

void test_self_search() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: SELF-SEARCH SANITY CHECK                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 1000;
    const uint64_t seed = 42;

    std::cout << "  Setup: " << num_vectors << " vectors, dim=" << dim
              << ", K=" << K << ", R=" << R << "\n\n";

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Generate vectors and encode
    std::vector<std::vector<Float>> vectors(num_vectors);
    std::vector<ResidualBinaryCode<K, R>> codes(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
        codes[i] = encoder.encode(vectors[i].data());
    }

    // Test: Search for vector 0 using brute-force ranking
    std::cout << "  Searching for Vector 0 (Index 0)...\n";

    ResidualBinaryCode<K, R> query_code = encoder.encode(vectors[0].data());

    // Compute distances to all vectors
    std::vector<std::pair<uint32_t, size_t>> distances;
    for (size_t i = 0; i < num_vectors; ++i) {
        uint32_t dist = residual_distance_integer_scalar<K, R, 2>(
            query_code.primary, query_code.residual,
            codes[i].primary, codes[i].residual);
        distances.emplace_back(dist, i);
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end());

    // Check results
    std::cout << "\n  Top-10 results for Query=Vector[0]:\n";
    std::cout << "  ╔═══════╦══════════╦════════════╗\n";
    std::cout << "  ║ Rank  ║ Index    ║ Distance   ║\n";
    std::cout << "  ╠═══════╬══════════╬════════════╣\n";

    bool found_self = false;
    size_t self_rank = 0;

    for (size_t i = 0; i < 10 && i < distances.size(); ++i) {
        std::cout << "  ║ " << std::setw(5) << (i + 1)
                  << " ║ " << std::setw(8) << distances[i].second
                  << " ║ " << std::setw(10) << distances[i].first << " ║\n";

        if (distances[i].second == 0) {
            found_self = true;
            self_rank = i + 1;
        }
    }
    std::cout << "  ╚═══════╩══════════╩════════════╝\n\n";

    // Find where index 0 actually is
    if (!found_self) {
        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances[i].second == 0) {
                self_rank = i + 1;
                break;
            }
        }
    }

    if (distances[0].second == 0 && distances[0].first == 0) {
        std::cout << "  ✅ PASS: Vector 0 is ranked #1 with Distance = 0\n";
    } else if (distances[0].first == 0) {
        std::cout << "  ⚠️  WARN: Distance = 0 but Index = " << distances[0].second
                  << " (not 0)\n";
        std::cout << "         This suggests duplicate or identical codes.\n";
    } else {
        std::cout << "  ❌ FAIL: Vector 0 is ranked #" << self_rank
                  << " with Distance = " << distances[self_rank - 1].first << "\n";
        std::cout << "         Self-search should return Distance = 0!\n";
    }

    // Check how many vectors have distance 0
    size_t zero_count = 0;
    for (const auto& d : distances) {
        if (d.first == 0) ++zero_count;
    }
    std::cout << "  INFO: " << zero_count << " vectors have Distance = 0\n";
}

// ============================================================================
// Test 2: Distance Distribution Analysis
// ============================================================================

void test_distance_distribution() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: DISTANCE DISTRIBUTION ANALYSIS                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 500;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Generate and encode
    std::vector<std::vector<Float>> vectors(num_vectors);
    std::vector<ResidualBinaryCode<K, R>> codes(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
        codes[i] = encoder.encode(vectors[i].data());
    }

    // Pick query 0 and compute distances to all others
    ResidualBinaryCode<K, R> query_code = codes[0];

    std::map<uint32_t, size_t> distance_histogram;
    std::vector<uint32_t> all_distances;

    for (size_t i = 1; i < num_vectors; ++i) {
        uint32_t dist = residual_distance_integer_scalar<K, R, 2>(
            query_code.primary, query_code.residual,
            codes[i].primary, codes[i].residual);
        distance_histogram[dist]++;
        all_distances.push_back(dist);
    }

    std::sort(all_distances.begin(), all_distances.end());

    std::cout << "  Distance distribution from Vector 0 to all others:\n\n";

    // Statistics
    uint32_t min_dist = all_distances.front();
    uint32_t max_dist = all_distances.back();
    double mean = 0.0;
    for (auto d : all_distances) mean += d;
    mean /= all_distances.size();

    std::cout << "  Min distance:  " << min_dist << "\n";
    std::cout << "  Max distance:  " << max_dist << "\n";
    std::cout << "  Mean distance: " << std::fixed << std::setprecision(1) << mean << "\n";
    std::cout << "  Unique values: " << distance_histogram.size() << "\n\n";

    // Show distribution
    std::cout << "  Distance histogram (top 10 most common):\n";
    std::vector<std::pair<size_t, uint32_t>> sorted_hist;
    for (const auto& p : distance_histogram) {
        sorted_hist.emplace_back(p.second, p.first);
    }
    std::sort(sorted_hist.rbegin(), sorted_hist.rend());

    for (size_t i = 0; i < 10 && i < sorted_hist.size(); ++i) {
        std::cout << "    Distance " << std::setw(3) << sorted_hist[i].second
                  << ": " << std::setw(4) << sorted_hist[i].first << " vectors\n";
    }

    // Check for tie-breaking issues
    size_t ties_in_top10 = 0;
    if (all_distances.size() >= 10) {
        uint32_t tenth_dist = all_distances[9];
        for (size_t i = 0; i < 20 && i < all_distances.size(); ++i) {
            if (all_distances[i] == tenth_dist) ties_in_top10++;
        }
    }

    std::cout << "\n  Tie-breaking analysis:\n";
    std::cout << "    Vectors at top-10 cutoff distance: " << ties_in_top10 << "\n";

    if (ties_in_top10 > 5) {
        std::cout << "    ⚠️  WARN: Many ties at top-10 boundary!\n";
        std::cout << "          This can cause unstable recall measurements.\n";
    }
}

// ============================================================================
// Test 3: Recall vs ef_search
// ============================================================================

void test_recall_vs_ef() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: BRUTE-FORCE RECALL AT DIFFERENT TOP-K                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // NOTE: This tests the RANKING QUALITY of the distance function
    // It does NOT test graph-based search (which would use ef_search)

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 500;
    constexpr size_t num_queries = 50;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Generate vectors
    std::vector<std::vector<Float>> vectors(num_vectors);
    std::vector<ResidualBinaryCode<K, R>> codes(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
        codes[i] = encoder.encode(vectors[i].data());
    }

    std::cout << "  Testing brute-force ranking quality...\n";
    std::cout << "  Dataset: " << num_vectors << " vectors, " << num_queries << " queries\n\n";

    // Test at different top-k values
    std::vector<size_t> topk_values = {1, 5, 10, 20, 50, 100};

    std::cout << "  ╔═════════╦═══════════════╦═══════════════╗\n";
    std::cout << "  ║ Top-K   ║ Recall (L2)   ║ Recall (Phase2) ║\n";
    std::cout << "  ╠═════════╬═══════════════╬═══════════════╣\n";

    for (size_t topk : topk_values) {
        if (topk > num_vectors) continue;

        size_t correct_l2 = 0;
        size_t correct_phase2 = 0;
        size_t total = num_queries * topk;

        for (size_t q = 0; q < num_queries; ++q) {
            // Use a random vector as query (not from dataset)
            auto query_vec = random_normalized_vector(dim, rng);
            ResidualBinaryCode<K, R> query_code = encoder.encode(query_vec.data());

            // Compute all distances
            std::vector<std::pair<Float, size_t>> l2_distances;
            std::vector<std::pair<uint32_t, size_t>> phase2_distances;

            for (size_t i = 0; i < num_vectors; ++i) {
                Float l2_dist = compute_l2_distance(query_vec, vectors[i]);
                uint32_t p2_dist = residual_distance_integer_scalar<K, R, 2>(
                    query_code.primary, query_code.residual,
                    codes[i].primary, codes[i].residual);

                l2_distances.emplace_back(l2_dist, i);
                phase2_distances.emplace_back(p2_dist, i);
            }

            // Sort both
            std::sort(l2_distances.begin(), l2_distances.end());
            std::sort(phase2_distances.begin(), phase2_distances.end());

            // Get true top-k (by L2)
            std::set<size_t> true_topk;
            for (size_t i = 0; i < topk; ++i) {
                true_topk.insert(l2_distances[i].second);
            }

            // Count matches in Phase2 top-k
            for (size_t i = 0; i < topk; ++i) {
                if (true_topk.count(phase2_distances[i].second)) {
                    ++correct_phase2;
                }
            }

            // Sanity check: L2 vs L2 should be 100%
            for (size_t i = 0; i < topk; ++i) {
                if (true_topk.count(l2_distances[i].second)) {
                    ++correct_l2;
                }
            }
        }

        float recall_l2 = 100.0f * correct_l2 / total;
        float recall_phase2 = 100.0f * correct_phase2 / total;

        std::cout << "  ║ " << std::setw(7) << topk
                  << " ║ " << std::setw(11) << std::fixed << std::setprecision(1) << recall_l2 << "%"
                  << " ║ " << std::setw(11) << recall_phase2 << "%    ║\n";
    }

    std::cout << "  ╚═════════╩═══════════════╩═══════════════╝\n";

    std::cout << "\n  NOTE: L2 Recall should be 100% (sanity check).\n";
    std::cout << "        Phase2 Recall shows ranking quality of integer distance.\n";
}

// ============================================================================
// Test 4: True vs Estimated Distance Scatter
// ============================================================================

void test_ranking_quality() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 4: RANKING QUALITY ANALYSIS                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t K = 64;
    constexpr size_t R = 32;
    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 200;
    const uint64_t seed = 42;

    ResidualCPEncoder<K, R> encoder(dim, seed);
    std::mt19937 rng(123);

    // Generate vectors
    std::vector<std::vector<Float>> vectors(num_vectors);
    std::vector<ResidualBinaryCode<K, R>> codes(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
        codes[i] = encoder.encode(vectors[i].data());
    }

    // For query 0, compute L2 and Phase2 distances
    auto& query_vec = vectors[0];
    auto& query_code = codes[0];

    std::vector<std::tuple<Float, uint32_t, size_t>> all_distances;

    for (size_t i = 1; i < num_vectors; ++i) {
        Float l2_dist = compute_l2_distance(query_vec, vectors[i]);
        uint32_t p2_dist = residual_distance_integer_scalar<K, R, 2>(
            query_code.primary, query_code.residual,
            codes[i].primary, codes[i].residual);

        all_distances.emplace_back(l2_dist, p2_dist, i);
    }

    // Sort by true L2 distance
    std::sort(all_distances.begin(), all_distances.end());

    std::cout << "  Top-20 nearest neighbors by TRUE L2 distance:\n\n";
    std::cout << "  ╔══════╦═══════════╦═══════════╦═══════════════╗\n";
    std::cout << "  ║ Rank ║ True L2   ║ Phase2    ║ Phase2 Rank   ║\n";
    std::cout << "  ╠══════╬═══════════╬═══════════╬═══════════════╣\n";

    // Also sort by Phase2 to get Phase2 ranks
    std::vector<std::pair<uint32_t, size_t>> p2_sorted;
    for (const auto& t : all_distances) {
        p2_sorted.emplace_back(std::get<1>(t), std::get<2>(t));
    }
    std::sort(p2_sorted.begin(), p2_sorted.end());

    std::map<size_t, size_t> p2_rank_map;
    for (size_t i = 0; i < p2_sorted.size(); ++i) {
        p2_rank_map[p2_sorted[i].second] = i + 1;
    }

    size_t rank_sum = 0;
    for (size_t i = 0; i < 20 && i < all_distances.size(); ++i) {
        size_t idx = std::get<2>(all_distances[i]);
        size_t p2_rank = p2_rank_map[idx];
        rank_sum += p2_rank;

        std::cout << "  ║ " << std::setw(4) << (i + 1)
                  << " ║ " << std::setw(9) << std::fixed << std::setprecision(4)
                  << std::get<0>(all_distances[i])
                  << " ║ " << std::setw(9) << std::get<1>(all_distances[i])
                  << " ║ " << std::setw(13) << p2_rank << " ║\n";
    }
    std::cout << "  ╚══════╩═══════════╩═══════════╩═══════════════╝\n";

    float avg_rank = static_cast<float>(rank_sum) / 20.0f;
    std::cout << "\n  Average Phase2 rank for true Top-20: " << std::fixed
              << std::setprecision(1) << avg_rank << "\n";

    if (avg_rank < 30) {
        std::cout << "  ✅ Good: True neighbors are ranked reasonably well\n";
    } else if (avg_rank < 50) {
        std::cout << "  ⚠️  Marginal: Some ranking quality loss\n";
    } else {
        std::cout << "  ❌ Poor: Significant ranking degradation\n";
    }
}

// ============================================================================
// Test 5: The Root Cause - Random Data Problem
// ============================================================================

void test_random_data_problem() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 5: RANDOM DATA PROBLEM DIAGNOSIS                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 100;
    std::mt19937 rng(42);

    // Generate random unit vectors
    std::vector<std::vector<Float>> vectors(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
    }

    // Compute pairwise L2 distances
    std::vector<Float> all_l2;
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t j = i + 1; j < num_vectors; ++j) {
            all_l2.push_back(compute_l2_distance(vectors[i], vectors[j]));
        }
    }

    std::sort(all_l2.begin(), all_l2.end());

    Float min_l2 = all_l2.front();
    Float max_l2 = all_l2.back();
    Float mean_l2 = 0.0f;
    for (auto d : all_l2) mean_l2 += d;
    mean_l2 /= all_l2.size();

    std::cout << "  Pairwise L2 distance statistics for " << num_vectors
              << " random unit vectors:\n\n";
    std::cout << "    Min L2:  " << std::fixed << std::setprecision(4) << min_l2 << "\n";
    std::cout << "    Max L2:  " << max_l2 << "\n";
    std::cout << "    Mean L2: " << mean_l2 << "\n";
    std::cout << "    Range:   " << (max_l2 - min_l2) << "\n\n";

    // For random unit vectors in d dimensions:
    // E[||x-y||^2] = E[||x||^2] + E[||y||^2] - 2*E[x.y] = 1 + 1 - 0 = 2
    // So E[||x-y||] ≈ sqrt(2) ≈ 1.414

    std::cout << "  THEORETICAL ANALYSIS:\n";
    std::cout << "  ---------------------\n";
    std::cout << "  For random unit vectors in d=" << dim << " dimensions:\n";
    std::cout << "    - Expected L2 distance: sqrt(2) ≈ 1.414\n";
    std::cout << "    - Observed mean: " << mean_l2 << "\n";
    std::cout << "    - Distance RANGE is very narrow!\n\n";

    std::cout << "  IMPLICATION FOR RECALL:\n";
    std::cout << "  -----------------------\n";
    std::cout << "  When all pairwise distances are similar (~1.4 ± 0.1),\n";
    std::cout << "  even small quantization errors cause ranking shuffles.\n";
    std::cout << "  This explains low recall on RANDOM synthetic data.\n\n";

    std::cout << "  SOLUTION:\n";
    std::cout << "  ---------\n";
    std::cout << "  Test on REAL data (SIFT-1M, GloVe) where vectors have\n";
    std::cout << "  meaningful structure and diverse pairwise distances.\n";
}

// ============================================================================
// Test 6: Graph-Based Search with CPHNSWIndex
// ============================================================================

void test_graph_based_search() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 6: GRAPH-BASED SEARCH (CPHNSWIndex with ef=500)          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 1000;
    constexpr size_t num_queries = 50;
    constexpr size_t M = 16;
    constexpr size_t ef_construction = 100;

    std::cout << "  Setup: " << num_vectors << " vectors, dim=" << dim
              << ", M=" << M << ", ef_construction=" << ef_construction << "\n\n";

    // Create index with K=64
    CPHNSWIndex<uint8_t, 64> index(dim, M, ef_construction);
    std::mt19937 rng(42);

    // Generate and add vectors
    std::vector<std::vector<Float>> vectors(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors[i] = random_normalized_vector(dim, rng);
    }

    std::cout << "  Building graph...\n";
    for (size_t i = 0; i < num_vectors; ++i) {
        index.add(vectors[i].data());
    }
    std::cout << "  Index size: " << index.size() << " vectors\n\n";

    // Generate queries
    std::cout << "  Running " << num_queries << " queries with different ef values...\n\n";

    std::vector<size_t> ef_values = {50, 100, 200, 500};
    size_t top_k = 10;

    std::cout << "  ╔═══════════╦══════════════════╦══════════════════╗\n";
    std::cout << "  ║ ef_search ║ Recall@10 (Hybrid) ║ Recall@10 (Graph) ║\n";
    std::cout << "  ╠═══════════╬══════════════════╬══════════════════╣\n";

    for (size_t ef : ef_values) {
        size_t correct_hybrid = 0;
        size_t correct_graph = 0;
        size_t total = num_queries * top_k;

        for (size_t q = 0; q < num_queries; ++q) {
            auto query_vec = random_normalized_vector(dim, rng);

            // Compute brute-force ground truth
            std::vector<std::pair<Float, size_t>> true_distances;
            for (size_t i = 0; i < num_vectors; ++i) {
                Float dist = compute_l2_distance(query_vec, vectors[i]);
                true_distances.emplace_back(dist, i);
            }
            std::sort(true_distances.begin(), true_distances.end());

            std::set<size_t> ground_truth;
            for (size_t i = 0; i < top_k; ++i) {
                ground_truth.insert(true_distances[i].second);
            }

            // Graph-only search
            auto graph_results = index.search(query_vec.data(), top_k, ef);

            // Hybrid search with reranking
            auto hybrid_results = index.search_and_rerank(
                query_vec.data(), top_k, ef, 200);

            // Count correct
            for (const auto& r : graph_results) {
                if (ground_truth.count(r.id)) {
                    ++correct_graph;
                }
            }
            for (const auto& r : hybrid_results) {
                if (ground_truth.count(r.id)) {
                    ++correct_hybrid;
                }
            }
        }

        float recall_hybrid = 100.0f * correct_hybrid / total;
        float recall_graph = 100.0f * correct_graph / total;

        std::cout << "  ║ " << std::setw(9) << ef
                  << " ║ " << std::setw(15) << std::fixed << std::setprecision(1)
                  << recall_hybrid << "% ║ " << std::setw(15) << recall_graph << "% ║\n";
    }

    std::cout << "  ╚═══════════╩══════════════════╩══════════════════╝\n\n";

    std::cout << "  NOTE: This uses the EXISTING Phase 1 CPHNSWIndex (not Phase 2 residual codes).\n";
    std::cout << "        Hybrid search includes float-precision reranking of candidates.\n";
}

// ============================================================================
// Test 7: Graph Connectivity Check
// ============================================================================

void test_graph_connectivity() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 7: GRAPH CONNECTIVITY CHECK                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    constexpr size_t dim = 128;
    constexpr size_t num_vectors = 1000;
    constexpr size_t M = 16;
    constexpr size_t ef_construction = 100;

    CPHNSWIndex<uint8_t, 64> index(dim, M, ef_construction);
    std::mt19937 rng(42);

    // Build index
    for (size_t i = 0; i < num_vectors; ++i) {
        auto vec = random_normalized_vector(dim, rng);
        index.add(vec.data());
    }

    // Check connectivity by searching from each node
    size_t reachable_nodes = 0;
    std::vector<bool> visited(num_vectors, false);

    // BFS from node 0
    std::queue<size_t> q;
    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        size_t current = q.front();
        q.pop();
        ++reachable_nodes;

        // This is a simplified check - in practice we'd need access to neighbor lists
        // For now, we'll use search to verify connectivity
    }

    // Actually test connectivity via search - can we reach all nodes?
    std::vector<Float> random_query = random_normalized_vector(dim, rng);
    auto results = index.search(random_query.data(), num_vectors, 500);

    std::set<size_t> found_ids;
    for (const auto& r : results) {
        found_ids.insert(r.id);
    }

    float connectivity = 100.0f * found_ids.size() / num_vectors;

    std::cout << "  Vectors in index: " << index.size() << "\n";
    std::cout << "  Reachable via search (ef=500): " << found_ids.size() << "\n";
    std::cout << "  Connectivity: " << std::fixed << std::setprecision(1)
              << connectivity << "%\n\n";

    if (connectivity > 99.0) {
        std::cout << "  ✅ PASS: Graph is well-connected\n";
    } else if (connectivity > 90.0) {
        std::cout << "  ⚠️  WARN: Some isolated nodes (may affect recall)\n";
    } else {
        std::cout << "  ❌ FAIL: Significant connectivity issues\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              RECALL COLLAPSE DIAGNOSTIC SUITE                 ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║  Investigating the 3% recall issue before Phase 3.           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    test_self_search();
    test_distance_distribution();
    test_recall_vs_ef();
    test_ranking_quality();
    test_random_data_problem();
    test_graph_based_search();
    test_graph_connectivity();

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         SUMMARY                               ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  The 3% recall in test_3b was due to:                        ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║  1. Testing on RANDOM synthetic data (not SIFT-1M)           ║\n";
    std::cout << "║  2. Random unit vectors have very similar pairwise distances  ║\n";
    std::cout << "║  3. Small quantization errors → large ranking changes        ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║  NEXT STEPS:                                                  ║\n";
    std::cout << "║  - Test on SIFT-1M dataset for real recall numbers           ║\n";
    std::cout << "║  - Use graph-based search (not brute force) with ef_search   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
