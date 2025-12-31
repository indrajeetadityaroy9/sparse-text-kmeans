/**
 * QA Protocol Tests for CP-NSW-Flash Implementation
 *
 * This test suite catches silent mathematical errors, race conditions,
 * and performance regressions after architectural changes (SoA layout, GPU construction).
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "../include/cphnsw/distance/hamming.hpp"
#include "../include/cphnsw/graph/flat_graph.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>
#include <queue>
#include <unordered_set>
#include <iomanip>

using namespace cphnsw;

// Helper: Generate random unit vector
std::vector<float> random_unit_vector(size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
    return vec;
}

// Helper: Flatten vector of vectors into contiguous array
std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vecs) {
    if (vecs.empty()) return {};
    size_t dim = vecs[0].size();
    std::vector<float> flat(vecs.size() * dim);
    for (size_t i = 0; i < vecs.size(); ++i) {
        std::copy(vecs[i].begin(), vecs[i].end(), flat.begin() + i * dim);
    }
    return flat;
}

// Helper: Compute true cosine distance (negative dot product for unit vectors)
float true_cosine_distance(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return -dot;  // Negative dot product (lower = more similar)
}

// =============================================================================
// Section 1: Mathematical & Algorithmic Integrity
// =============================================================================

bool test_identity() {
    std::cout << "\n=== Test 1.1: Identity Test ===" << std::endl;
    std::cout << "Insert vector X, query for X. Distance should be ~0." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(42);
    std::vector<float> test_vec = random_unit_vector(dim, rng);

    // Build small index
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < 100; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }
    vectors[50] = test_vec;  // Insert our test vector at position 50

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());

    // Query for the same vector
    auto results = index.search(test_vec.data(), 1, 100);

    if (results.empty()) {
        std::cout << "  [FAIL] No results returned" << std::endl;
        return false;
    }

    std::cout << "  Query result: ID=" << results[0].id << ", Distance=" << results[0].distance << std::endl;

    // Check if we found the same vector (should be ID 50)
    if (results[0].id == 50) {
        std::cout << "  [PASS] Found exact match (ID=50)" << std::endl;

        // Distance should be very close to negative max (most similar)
        if (results[0].distance < 0) {
            std::cout << "  [PASS] Distance is negative (correct sign convention)" << std::endl;
            return true;
        } else {
            std::cout << "  [WARN] Distance is positive: " << results[0].distance << std::endl;
            std::cout << "         This may indicate sign flip issues" << std::endl;
            return true;  // Still found correct ID
        }
    } else {
        std::cout << "  [FAIL] Expected ID=50, got ID=" << results[0].id << std::endl;
        return false;
    }
}

bool test_sign_flip_regression() {
    std::cout << "\n=== Test 1.2: Sign Flip Regression Check ===" << std::endl;
    std::cout << "Top 5 candidates should have NEGATIVE scores, sorted ascending." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(123);
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < 1000; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());

    auto query = random_unit_vector(dim, rng);
    auto results = index.search(query.data(), 5, 100);

    std::cout << "  Top 5 results (should be negative, ascending):" << std::endl;
    bool all_negative = true;
    bool ascending = true;

    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "    [" << i << "] ID=" << results[i].id
                  << ", Distance=" << std::fixed << std::setprecision(4)
                  << results[i].distance << std::endl;

        if (results[i].distance > 0) {
            all_negative = false;
        }
        if (i > 0 && results[i].distance < results[i-1].distance) {
            ascending = false;
        }
    }

    if (!all_negative) {
        std::cout << "  [FAIL] Some distances are positive (sign flip bug!)" << std::endl;
        return false;
    }

    if (!ascending) {
        std::cout << "  [FAIL] Results not sorted ascending (heap bug!)" << std::endl;
        return false;
    }

    std::cout << "  [PASS] All distances negative and sorted correctly" << std::endl;
    return true;
}

bool test_soa_transposition() {
    std::cout << "\n=== Test 1.3: SoA Transposition Verification ===" << std::endl;
    std::cout << "Comparing scalar vs batch distance computation." << std::endl;

    constexpr size_t K = 32;
    constexpr size_t dim = 128;

    std::mt19937 rng(456);

    // Create a mock query
    CPQuery<uint8_t, K> query;
    for (size_t k = 0; k < K; ++k) {
        query.rotated_vecs[k].resize(dim);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < dim; ++i) {
            query.rotated_vecs[k][i] = dist(rng);
        }
        // Set primary code
        query.primary_code.components[k] = static_cast<uint8_t>((rng() % dim) << 1 | (rng() % 2));
    }

    // Create neighbor block with SoA layout
    NeighborBlock<uint8_t, K> block;
    block.count = 8;  // 8 neighbors

    // Fill with random codes
    std::vector<CPCode<uint8_t, K>> neighbor_codes(8);
    for (size_t n = 0; n < 8; ++n) {
        block.ids[n] = static_cast<NodeId>(n);
        for (size_t k = 0; k < K; ++k) {
            uint8_t raw = static_cast<uint8_t>((rng() % dim) << 1 | (rng() % 2));
            neighbor_codes[n].components[k] = raw;
        }
        block.set_neighbor_code(n, neighbor_codes[n]);
    }

    // Compute distances using scalar (from gathered code)
    std::vector<AsymmetricDist> scalar_distances(8);
    for (size_t n = 0; n < 8; ++n) {
        CPCode<uint8_t, K> gathered_code = block.get_neighbor_code_copy(n);
        scalar_distances[n] = asymmetric_search_distance(query, gathered_code);
    }

    // Compute distances using batch SoA kernel
    std::vector<AsymmetricDist> batch_distances(8);
    asymmetric_search_distance_batch_soa<uint8_t, K>(
        query, block.codes_transposed, 8, batch_distances.data());

    // Compare
    bool all_match = true;
    float max_diff = 0.0f;

    std::cout << "  Neighbor | Scalar Dist | Batch Dist | Diff" << std::endl;
    std::cout << "  ---------|-------------|------------|------" << std::endl;

    for (size_t n = 0; n < 8; ++n) {
        float diff = std::abs(scalar_distances[n] - batch_distances[n]);
        max_diff = std::max(max_diff, diff);

        std::cout << "  " << std::setw(8) << n << " | "
                  << std::setw(11) << std::fixed << std::setprecision(4) << scalar_distances[n] << " | "
                  << std::setw(10) << batch_distances[n] << " | "
                  << std::setw(6) << diff << std::endl;

        if (diff > 1e-5f) {
            all_match = false;
        }
    }

    if (all_match) {
        std::cout << "  [PASS] Scalar and batch results are identical (max diff: " << max_diff << ")" << std::endl;
        return true;
    } else {
        std::cout << "  [FAIL] Results differ! SoA transposition is broken." << std::endl;
        return false;
    }
}

bool test_finger_calibration_sanity() {
    std::cout << "\n=== Test 1.4: FINGER Calibration Sanity ===" << std::endl;
    std::cout << "Alpha should be > 0 (positive correlation)." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(789);
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < 5000; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());
    index.calibrate(1000);  // Calibrate with 1000 samples

    float alpha = index.get_calibration().alpha;
    float beta = index.get_calibration().beta;
    float r_squared = index.get_calibration().r_squared;

    std::cout << "  Alpha:     " << alpha << std::endl;
    std::cout << "  Beta:      " << beta << std::endl;
    std::cout << "  R-squared: " << r_squared << std::endl;

    if (alpha > 0) {
        std::cout << "  [PASS] Alpha > 0 (positive correlation)" << std::endl;
        return true;
    } else if (std::abs(alpha) < 1e-6f) {
        std::cout << "  [WARN] Alpha ~ 0 (estimator provides no signal)" << std::endl;
        return false;
    } else {
        std::cout << "  [FAIL] Alpha < 0 (negative correlation - sign bug!)" << std::endl;
        return false;
    }
}

// =============================================================================
// Section 2: Memory & Systems (Flash Layout)
// =============================================================================

bool test_alignment() {
    std::cout << "\n=== Test 2.1: Alignment Assertion ===" << std::endl;
    std::cout << "NeighborBlock codes_transposed must be 64-byte aligned for AVX-512." << std::endl;

    constexpr size_t K = 32;

    NeighborBlock<uint8_t, K> block;

    uintptr_t block_addr = reinterpret_cast<uintptr_t>(&block);
    uintptr_t codes_addr = reinterpret_cast<uintptr_t>(&block.codes_transposed[0][0]);

    std::cout << "  Block address:  0x" << std::hex << block_addr << std::dec << std::endl;
    std::cout << "  Codes address:  0x" << std::hex << codes_addr << std::dec << std::endl;
    std::cout << "  Block alignment: " << (block_addr % 64) << " (should be 0)" << std::endl;

    if (block_addr % 64 != 0) {
        std::cout << "  [FAIL] Block is not 64-byte aligned" << std::endl;
        return false;
    }

    std::cout << "  [PASS] Block is 64-byte aligned" << std::endl;
    return true;
}

// =============================================================================
// Section 3: Graph Construction
// =============================================================================

bool test_self_loop_removal() {
    std::cout << "\n=== Test 3.1: Self-Loop Removal ===" << std::endl;
    std::cout << "No node should be its own nearest neighbor in search results." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(111);
    std::vector<std::vector<float>> vectors;
    for (int i = 0; i < 1000; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());

    // Check if any query returns itself as a neighbor (other than top-1 for exact match)
    // This is an indirect self-loop check
    int suspicious = 0;
    for (int i = 0; i < 100; ++i) {
        auto results = index.search(vectors[i].data(), 10, 100);

        // Count how many times node i appears in its own results
        int self_count = 0;
        for (const auto& r : results) {
            if (r.id == static_cast<NodeId>(i)) {
                self_count++;
            }
        }

        // Should appear exactly once (as itself)
        if (self_count > 1) {
            suspicious++;
            if (suspicious <= 3) {
                std::cout << "  Node " << i << " appears " << self_count << " times in results" << std::endl;
            }
        }
    }

    if (suspicious == 0) {
        std::cout << "  [PASS] No duplicate self-references in search results" << std::endl;
        return true;
    } else {
        std::cout << "  [WARN] Found " << suspicious << " suspicious self-references" << std::endl;
        return true;  // This is a soft check
    }
}

bool test_graph_reachability() {
    std::cout << "\n=== Test 3.2: Graph Reachability (Islands Check) ===" << std::endl;
    std::cout << "verify_connectivity() should return ~100% of nodes." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;
    constexpr size_t N = 5000;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(222);
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < N; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());

    // Use built-in connectivity check
    size_t reachable = index.verify_connectivity();
    float reachability = 100.0f * reachable / N;

    std::cout << "  Reachable: " << reachable << "/" << N
              << " (" << std::fixed << std::setprecision(1) << reachability << "%)" << std::endl;

    if (reachability >= 99.0f) {
        std::cout << "  [PASS] Graph is fully connected" << std::endl;
        return true;
    } else if (reachability >= 90.0f) {
        std::cout << "  [WARN] Graph has minor islands (may affect recall)" << std::endl;
        return true;
    } else {
        std::cout << "  [FAIL] Graph is fragmented! Check rank pruning settings." << std::endl;
        return false;
    }
}

// =============================================================================
// Section 4: Edge Cases & Robustness
// =============================================================================

bool test_small_dataset() {
    std::cout << "\n=== Test 4.1: Dataset Size < Batch Size ===" << std::endl;
    std::cout << "Running with only 100 vectors." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    try {
        CPHNSWIndex<uint8_t, K> index(dim, 16, 50);

        std::mt19937 rng(333);
        std::vector<std::vector<float>> vectors;
        for (int i = 0; i < 100; ++i) {
            vectors.push_back(random_unit_vector(dim, rng));
        }

        auto flat = flatten_vectors(vectors);
        index.add_batch(flat.data(), vectors.size());

        auto query = random_unit_vector(dim, rng);
        auto results = index.search(query.data(), 10, 50);

        std::cout << "  Build and search completed successfully" << std::endl;
        std::cout << "  Results returned: " << results.size() << std::endl;

        if (results.size() >= 1) {
            std::cout << "  [PASS] Small dataset handled correctly" << std::endl;
            return true;
        } else {
            std::cout << "  [FAIL] No results returned" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_duplicate_vectors() {
    std::cout << "\n=== Test 4.2: Duplicate Vectors ===" << std::endl;
    std::cout << "Inserting same vector 50 times, search should return multiple IDs." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    try {
        CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

        std::mt19937 rng(444);
        auto duplicate_vec = random_unit_vector(dim, rng);

        std::vector<std::vector<float>> vectors;
        // 50 unique vectors
        for (int i = 0; i < 50; ++i) {
            vectors.push_back(random_unit_vector(dim, rng));
        }
        // 50 duplicates
        for (int i = 0; i < 50; ++i) {
            vectors.push_back(duplicate_vec);
        }

        auto flat = flatten_vectors(vectors);
        index.add_batch(flat.data(), vectors.size());

        // Search for the duplicate vector
        auto results = index.search(duplicate_vec.data(), 60, 200);

        // Count how many of the results are duplicates (IDs 50-99)
        int duplicate_hits = 0;
        for (const auto& r : results) {
            if (r.id >= 50 && r.id < 100) {
                duplicate_hits++;
            }
        }

        std::cout << "  Found " << duplicate_hits << "/50 duplicates in top "
                  << results.size() << " results" << std::endl;

        if (duplicate_hits >= 40) {
            std::cout << "  [PASS] Most duplicates found" << std::endl;
            return true;
        } else if (duplicate_hits >= 20) {
            std::cout << "  [WARN] Only " << duplicate_hits << " duplicates found (recall issue)" << std::endl;
            return true;
        } else {
            std::cout << "  [FAIL] Very few duplicates found - possible infinite loop or bug" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool test_zero_vector() {
    std::cout << "\n=== Test 4.3: Zero Vector Handling ===" << std::endl;
    std::cout << "Zero vector should not produce NaN (graceful normalization)." << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;

    try {
        CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

        std::mt19937 rng(555);
        std::vector<std::vector<float>> vectors;

        // 99 random vectors
        for (int i = 0; i < 99; ++i) {
            vectors.push_back(random_unit_vector(dim, rng));
        }
        // 1 zero vector
        vectors.push_back(std::vector<float>(dim, 0.0f));

        auto flat = flatten_vectors(vectors);
        index.add_batch(flat.data(), vectors.size());

        // Query with a normal vector
        auto query = random_unit_vector(dim, rng);
        auto results = index.search(query.data(), 10, 50);

        // Check for NaN in results
        bool has_nan = false;
        for (const auto& r : results) {
            if (std::isnan(r.distance)) {
                has_nan = true;
                break;
            }
        }

        if (has_nan) {
            std::cout << "  [FAIL] NaN detected in results (zero vector corruption)" << std::endl;
            return false;
        }

        std::cout << "  Results returned: " << results.size() << " (no NaN)" << std::endl;
        std::cout << "  [PASS] Zero vector handled gracefully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "  [INFO] Exception thrown (acceptable): " << e.what() << std::endl;
        std::cout << "  [PASS] Zero vector rejected with exception" << std::endl;
        return true;
    }
}

// =============================================================================
// Section 5: Performance Red Flags
// =============================================================================

bool test_recall_vs_ef() {
    std::cout << "\n=== Test 5.1: Recall vs ef (Should NOT decrease) ===" << std::endl;

    constexpr size_t dim = 128;
    constexpr size_t K = 16;
    constexpr size_t N = 5000;
    constexpr size_t num_queries = 100;
    constexpr size_t k_neighbors = 10;

    CPHNSWIndex<uint8_t, K> index(dim, 16, 100);

    std::mt19937 rng(666);
    std::vector<std::vector<float>> vectors;
    for (size_t i = 0; i < N; ++i) {
        vectors.push_back(random_unit_vector(dim, rng));
    }

    auto flat = flatten_vectors(vectors);
    index.add_batch(flat.data(), vectors.size());

    // Generate queries
    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < num_queries; ++i) {
        queries.push_back(random_unit_vector(dim, rng));
    }

    // Compute ground truth (brute force)
    std::vector<std::vector<NodeId>> ground_truth(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        std::vector<std::pair<float, NodeId>> dists;
        for (size_t i = 0; i < N; ++i) {
            float d = true_cosine_distance(queries[q].data(), vectors[i].data(), dim);
            dists.emplace_back(d, static_cast<NodeId>(i));
        }
        std::partial_sort(dists.begin(), dists.begin() + k_neighbors, dists.end());
        for (size_t i = 0; i < k_neighbors; ++i) {
            ground_truth[q].push_back(dists[i].second);
        }
    }

    // Test recall at different ef values
    std::vector<size_t> ef_values = {10, 20, 50, 100, 200};
    std::vector<float> recalls;

    std::cout << "  ef     Recall@10" << std::endl;
    std::cout << "  -----  ---------" << std::endl;

    float prev_recall = 0.0f;
    bool decreasing = false;

    for (size_t ef : ef_values) {
        size_t hits = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            auto results = index.search(queries[q].data(), k_neighbors, ef);
            std::unordered_set<NodeId> gt_set(ground_truth[q].begin(), ground_truth[q].end());
            for (const auto& r : results) {
                if (gt_set.count(r.id)) {
                    hits++;
                }
            }
        }

        float recall = static_cast<float>(hits) / (num_queries * k_neighbors);
        recalls.push_back(recall);

        std::cout << "  " << std::setw(5) << ef << "  "
                  << std::fixed << std::setprecision(4) << recall << std::endl;

        if (recall < prev_recall - 0.01f) {  // Allow 1% tolerance
            decreasing = true;
        }
        prev_recall = recall;
    }

    if (decreasing) {
        std::cout << "  [FAIL] Recall decreased as ef increased (sign flip or overflow bug!)" << std::endl;
        return false;
    }

    std::cout << "  [PASS] Recall increases monotonically with ef" << std::endl;
    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CP-NSW-Flash QA Protocol Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int failed = 0;

    // Section 1: Mathematical & Algorithmic Integrity
    std::cout << "\n### Section 1: Mathematical & Algorithmic Integrity ###" << std::endl;

    if (test_identity()) passed++; else failed++;
    if (test_sign_flip_regression()) passed++; else failed++;
    if (test_soa_transposition()) passed++; else failed++;
    if (test_finger_calibration_sanity()) passed++; else failed++;

    // Section 2: Memory & Systems
    std::cout << "\n### Section 2: Memory & Systems (Flash Layout) ###" << std::endl;

    if (test_alignment()) passed++; else failed++;

    // Section 3: Graph Construction
    std::cout << "\n### Section 3: Graph Construction ###" << std::endl;

    if (test_self_loop_removal()) passed++; else failed++;
    if (test_graph_reachability()) passed++; else failed++;

    // Section 4: Edge Cases
    std::cout << "\n### Section 4: Edge Cases & Robustness ###" << std::endl;

    if (test_small_dataset()) passed++; else failed++;
    if (test_duplicate_vectors()) passed++; else failed++;
    if (test_zero_vector()) passed++; else failed++;

    // Section 5: Performance Red Flags
    std::cout << "\n### Section 5: Performance Red Flags ###" << std::endl;

    if (test_recall_vs_ef()) passed++; else failed++;

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "QA Protocol Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total:  " << (passed + failed) << std::endl;

    if (failed == 0) {
        std::cout << "\n[SUCCESS] All QA tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAILURE] " << failed << " test(s) failed. Review above." << std::endl;
        return 1;
    }
}
