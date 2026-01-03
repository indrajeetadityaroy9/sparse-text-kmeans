/**
 * Debug Recall Issues
 *
 * Isolates graph navigation from encoding quality.
 */

#include "../include/cphnsw/legacy/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unordered_set>

using namespace cphnsw;
using namespace cphnsw::eval;

constexpr size_t K = 32;
using Index = CPHNSWIndex<uint8_t, K>;

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_s() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
};

std::vector<std::vector<NodeId>> compute_ground_truth(
    const Float* base, size_t n_base,
    const Float* queries, size_t n_queries,
    size_t dim, size_t k) {

    std::vector<std::vector<NodeId>> gt(n_queries);

#pragma omp parallel for schedule(dynamic, 10)
    for (size_t q = 0; q < n_queries; ++q) {
        const Float* qv = queries + q * dim;
        std::vector<std::pair<Float, NodeId>> dists(n_base);

        for (size_t i = 0; i < n_base; ++i) {
            const Float* bv = base + i * dim;
            Float dot = 0;
            for (size_t j = 0; j < dim; ++j) {
                dot += qv[j] * bv[j];
            }
            dists[i] = {-dot, static_cast<NodeId>(i)};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].resize(k);
        for (size_t i = 0; i < k; ++i) {
            gt[q][i] = dists[i].second;
        }
    }
    return gt;
}

float compute_recall(const std::vector<SearchResult>& results,
                     const std::vector<NodeId>& gt, size_t k) {
    std::unordered_set<NodeId> gt_set(gt.begin(), gt.begin() + k);
    size_t hits = 0;
    for (size_t i = 0; i < std::min(results.size(), k); ++i) {
        if (gt_set.count(results[i].id)) hits++;
    }
    return static_cast<float>(hits) / k;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Debug Recall Issues\n";
    std::cout << "========================================\n\n";

    // Use smaller subset for faster testing
    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");

    size_t N = 100000;  // 100K for reasonable test time
    size_t n_queries = std::min(dataset.num_queries, size_t(1000));
    size_t dim = dataset.dim;
    size_t k = 10;

    std::cout << "N=" << N << ", queries=" << n_queries << ", k=" << k << "\n\n";

    // Build index
    CPHNSWParams params;
    params.dim = dim;
    params.k = K;
    params.M = 32;
    params.ef_construction = 200;
    params.seed = 42;
    params.finalize();

    std::cout << "Building index with GPU k-NN...\n";
    Timer timer;
    timer.start();

    Index index(params);
    index.build_with_gpu_knn(dataset.base_vectors.data(), N, 64, true, 1.1f);
    std::cout << "Build time: " << std::fixed << std::setprecision(1)
              << timer.elapsed_s() << "s\n";
    std::cout << "Connectivity: " << index.verify_connectivity() << "/" << N << "\n\n";

    // Ground truth
    std::cout << "Computing ground truth...\n";
    timer.start();
    auto gt = compute_ground_truth(
        dataset.base_vectors.data(), N,
        dataset.query_vectors.data(), n_queries,
        dim, k);
    std::cout << "  Done in " << timer.elapsed_s() << "s\n\n";

    // ========================================
    // Test 1: Standard Search (Graph + CP distance)
    // ========================================
    std::cout << "========================================\n";
    std::cout << "Test 1: Standard Search (graph + CP distance)\n";
    std::cout << "========================================\n\n";

    for (size_t ef : {50, 100, 200, 500}) {
        float total_recall = 0;
        timer.start();
        for (size_t q = 0; q < n_queries; ++q) {
            auto results = index.search(
                dataset.query_vectors.data() + q * dim, k, ef);
            total_recall += compute_recall(results, gt[q], k);
        }
        float avg_recall = total_recall / n_queries;
        double qps = n_queries / timer.elapsed_s();

        std::cout << "  ef=" << std::setw(4) << ef
                  << "  recall=" << std::fixed << std::setprecision(4) << avg_recall
                  << "  QPS=" << std::setw(6) << static_cast<int>(qps) << "\n";
    }

    // ========================================
    // Test 2: Search + Rerank (graph + exact rerank)
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Test 2: Search + Rerank (graph + exact rerank)\n";
    std::cout << "========================================\n\n";

    for (size_t ef : {50, 100, 200, 500}) {
        float total_recall = 0;
        size_t rerank_k = std::max(k * 10, size_t(100));
        timer.start();
        for (size_t q = 0; q < n_queries; ++q) {
            auto results = index.search_and_rerank(
                dataset.query_vectors.data() + q * dim, k, ef, rerank_k);
            total_recall += compute_recall(results, gt[q], k);
        }
        float avg_recall = total_recall / n_queries;
        double qps = n_queries / timer.elapsed_s();

        std::cout << "  ef=" << std::setw(4) << ef
                  << "  recall=" << std::fixed << std::setprecision(4) << avg_recall
                  << "  QPS=" << std::setw(6) << static_cast<int>(qps) << "\n";
    }

    // ========================================
    // Test 3: Brute Force with CP Distance (no graph)
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Test 3: Brute Force CP (no graph navigation)\n";
    std::cout << "========================================\n\n";

    {
        float total_recall = 0;
        timer.start();
        for (size_t q = 0; q < std::min(n_queries, size_t(100)); ++q) {
            auto results = index.brute_force_search(
                dataset.query_vectors.data() + q * dim, k);
            total_recall += compute_recall(results, gt[q], k);
        }
        float avg_recall = total_recall / std::min(n_queries, size_t(100));

        std::cout << "  (100 queries) recall=" << std::fixed << std::setprecision(4) << avg_recall << "\n";
        
        std::cout << "\n  This measures encoding quality without graph navigation.\n";
        std::cout << "  If this is low, the problem is CP encoding/distance.\n";
        std::cout << "  If this is high but Test 1 is low, the problem is graph navigation.\n";
    }

    // ========================================
    // Test 4: Calibration Check
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Test 4: Calibration Diagnostics\n";
    std::cout << "========================================\n\n";

    auto calib = index.calibrate(1000);
    std::cout << "  Calibration: alpha=" << calib.alpha
              << ", beta=" << calib.beta
              << ", R^2=" << calib.r_squared << "\n";

    // Test calibrated search
    std::cout << "\n  Calibrated Search:\n";
    for (size_t ef : {100, 200, 500}) {
        float total_recall = 0;
        for (size_t q = 0; q < n_queries; ++q) {
            auto results = index.search_calibrated(
                dataset.query_vectors.data() + q * dim, k, ef);
            total_recall += compute_recall(results, gt[q], k);
        }
        float avg_recall = total_recall / n_queries;
        std::cout << "  ef=" << std::setw(4) << ef
                  << "  recall=" << std::fixed << std::setprecision(4) << avg_recall << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "Using " << params.k_entry << " random entry points per search\n";

    return 0;
}
