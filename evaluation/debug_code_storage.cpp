/**
 * Debug Code Storage
 *
 * Directly verifies that stored codes match freshly encoded codes.
 * This isolates the exact point of data corruption.
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "../include/cphnsw/distance/hamming.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace cphnsw;
using namespace cphnsw::eval;

constexpr size_t K = 32;
using Index = CPHNSWIndex<uint8_t, K>;
using Encoder = CPEncoder<uint8_t, K>;
using Code = CPCode<uint8_t, K>;
using Query = CPQuery<uint8_t, K>;

float true_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Debug: Code Storage Verification\n";
    std::cout << "========================================\n\n";

    // Load SIFT
    std::cout << "Loading SIFT-1M...\n";
    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");

    size_t N = 10000;  // Use first 10K for quick test
    size_t dim = dataset.dim;

    std::cout << "Using N=" << N << ", dim=" << dim << "\n\n";

    // Build index with GPU
    CPHNSWParams params;
    params.dim = dim;
    params.k = K;
    params.M = 32;
    params.ef_construction = 200;
    params.seed = 42;
    params.finalize();

    std::cout << "Building index with GPU k-NN...\n";
    Index index(params);
    index.build_with_gpu_knn(dataset.base_vectors.data(), N, 64, true, 1.1f);

    std::cout << "\n========================================\n";
    std::cout << "Test 1: Stored Code vs Fresh Code\n";
    std::cout << "========================================\n\n";

    // Create a fresh encoder with SAME seed
    Encoder fresh_encoder(dim, params.seed);

    // Compare first 100 codes
    size_t exact_matches = 0;
    size_t close_matches = 0;
    size_t total = 0;

    for (size_t i = 0; i < 100; ++i) {
        const Float* vec = dataset.base_vectors.data() + i * dim;

        // Get stored code
        const Code& stored = index.get_code(static_cast<NodeId>(i));

        // Encode fresh
        Code fresh = fresh_encoder.encode(vec);

        // Compare
        int ham = hamming_distance(stored, fresh);

        if (ham == 0) exact_matches++;
        else if (ham <= 2) close_matches++;
        total++;

        if (i < 10 || ham > 0) {
            std::cout << "  Node " << std::setw(4) << i << ": Hamming = " << ham;
            if (ham > 0 && i < 10) {
                std::cout << " (MISMATCH)";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\nSummary:\n";
    std::cout << "  Exact matches:  " << exact_matches << "/" << total << "\n";
    std::cout << "  Close (ham<=2): " << close_matches << "/" << total << "\n";
    std::cout << "  Mismatches:     " << (total - exact_matches) << "/" << total << "\n";

    if (exact_matches == total) {
        std::cout << "  [PASS] All codes match!\n";
    } else {
        std::cout << "  [FAIL] Codes do NOT match - storage is corrupted!\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Test 2: R² with Stored Codes vs Fresh\n";
    std::cout << "========================================\n\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    std::vector<float> X_stored, Y_stored;
    std::vector<float> X_fresh, Y_fresh;

    for (int i = 0; i < 1000; ++i) {
        size_t u = dist(rng);
        size_t v = dist(rng);
        if (u == v) continue;

        const Float* vec_u = dataset.base_vectors.data() + u * dim;
        const Float* vec_v = dataset.base_vectors.data() + v * dim;

        float true_dist = -true_similarity(vec_u, vec_v, dim);

        // Fresh encoding
        Query query_u = fresh_encoder.encode_query(vec_u);
        Code code_v_fresh = fresh_encoder.encode(vec_v);
        float asymm_fresh = asymmetric_search_distance(query_u, code_v_fresh);
        X_fresh.push_back(asymm_fresh);
        Y_fresh.push_back(true_dist);

        // Stored code
        const Code& code_v_stored = index.get_code(static_cast<NodeId>(v));
        float asymm_stored = asymmetric_search_distance(query_u, code_v_stored);
        X_stored.push_back(asymm_stored);
        Y_stored.push_back(true_dist);
    }

    // Compute R² for both
    auto compute_r2 = [](const std::vector<float>& X, const std::vector<float>& Y) {
        float mean_x = 0, mean_y = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            mean_x += X[i];
            mean_y += Y[i];
        }
        mean_x /= X.size();
        mean_y /= Y.size();

        float var_x = 0, var_y = 0, cov_xy = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            float dx = X[i] - mean_x;
            float dy = Y[i] - mean_y;
            var_x += dx * dx;
            var_y += dy * dy;
            cov_xy += dx * dy;
        }

        float r = cov_xy / std::sqrt(var_x * var_y);
        return r * r;
    };

    float r2_fresh = compute_r2(X_fresh, Y_fresh);
    float r2_stored = compute_r2(X_stored, Y_stored);

    std::cout << "R² with fresh encoding:  " << std::fixed << std::setprecision(4) << r2_fresh << "\n";
    std::cout << "R² with stored codes:    " << std::fixed << std::setprecision(4) << r2_stored << "\n";

    if (r2_stored > 0.5) {
        std::cout << "\n[PASS] Stored codes are working correctly!\n";
    } else if (r2_fresh > 0.5 && r2_stored < 0.2) {
        std::cout << "\n[FAIL] Code storage is corrupted!\n";
        std::cout << "       Fresh encoding works (R²=" << r2_fresh << ")\n";
        std::cout << "       But stored codes don't (R²=" << r2_stored << ")\n";
    } else {
        std::cout << "\n[WARN] Both R² values are low - possible encoder issue\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Test 3: First 3 Codes Byte-by-Byte\n";
    std::cout << "========================================\n\n";

    for (size_t i = 0; i < 3; ++i) {
        const Float* vec = dataset.base_vectors.data() + i * dim;
        const Code& stored = index.get_code(static_cast<NodeId>(i));
        Code fresh = fresh_encoder.encode(vec);

        std::cout << "Node " << i << ":\n";
        std::cout << "  Stored: [";
        for (size_t k = 0; k < std::min(K, size_t(16)); ++k) {
            if (k > 0) std::cout << " ";
            std::cout << std::setw(3) << (int)stored.components[k];
        }
        std::cout << " ...]\n";

        std::cout << "  Fresh:  [";
        for (size_t k = 0; k < std::min(K, size_t(16)); ++k) {
            if (k > 0) std::cout << " ";
            std::cout << std::setw(3) << (int)fresh.components[k];
        }
        std::cout << " ...]\n";

        // Show differences
        std::cout << "  Diff:   [";
        for (size_t k = 0; k < std::min(K, size_t(16)); ++k) {
            if (k > 0) std::cout << " ";
            if (stored.components[k] != fresh.components[k]) {
                std::cout << std::setw(3) << "***";
            } else {
                std::cout << std::setw(3) << ".";
            }
        }
        std::cout << " ...]\n\n";
    }

    return 0;
}
