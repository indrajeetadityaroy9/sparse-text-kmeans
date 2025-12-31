/**
 * Test different K (code width) values for CP-LSH.
 *
 * Measures brute-force recall to isolate encoding quality from graph quality.
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <unordered_set>

using namespace cphnsw;
using namespace cphnsw::eval;

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

template<size_t K>
void test_k_value(const Dataset& dataset, size_t N, size_t n_queries,
                  const std::vector<std::vector<NodeId>>& gt, size_t k) {

    using Index = CPHNSWIndex<uint8_t, K>;

    CPHNSWParams params;
    params.dim = dataset.dim;
    params.k = K;
    params.M = 32;
    params.ef_construction = 200;
    params.seed = 42;
    params.finalize();

    std::cout << "\nK=" << K << " (code size: " << K << " bytes):\n";

    Index index(params);
    index.build_with_gpu_knn(dataset.base_vectors.data(), N, 64, true, 1.1f);
    std::cout << "  Connectivity: " << index.verify_connectivity() << "/" << N << "\n";

    // Brute force recall (encoding quality only)
    float total_recall = 0;
    for (size_t q = 0; q < std::min(n_queries, size_t(100)); ++q) {
        auto results = index.brute_force_search(
            dataset.query_vectors.data() + q * dataset.dim, k);
        total_recall += compute_recall(results, gt[q], k);
    }
    float bf_recall = total_recall / std::min(n_queries, size_t(100));
    std::cout << "  Brute Force CP Recall: " << std::fixed << std::setprecision(4)
              << bf_recall << "\n";

    // Standard search recall
    total_recall = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        auto results = index.search(
            dataset.query_vectors.data() + q * dataset.dim, k, 200);
        total_recall += compute_recall(results, gt[q], k);
    }
    float search_recall = total_recall / n_queries;
    std::cout << "  Standard Search (ef=200): " << std::fixed << std::setprecision(4)
              << search_recall << "\n";

    // Rerank recall
    total_recall = 0;
    for (size_t q = 0; q < n_queries; ++q) {
        auto results = index.search_and_rerank(
            dataset.query_vectors.data() + q * dataset.dim, k, 200, 100);
        total_recall += compute_recall(results, gt[q], k);
    }
    float rerank_recall = total_recall / n_queries;
    std::cout << "  Search + Rerank (ef=200): " << std::fixed << std::setprecision(4)
              << rerank_recall << "\n";

    // Calibration R²
    auto calib = index.calibrate(1000);
    std::cout << "  Calibration R²: " << std::fixed << std::setprecision(4)
              << calib.r_squared << "\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "CP-LSH K Value Comparison\n";
    std::cout << "========================================\n";

    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");

    size_t N = 100000;
    size_t n_queries = std::min(dataset.num_queries, size_t(1000));
    size_t k = 10;

    std::cout << "N=" << N << ", queries=" << n_queries << ", k=" << k << "\n";

    // Compute ground truth once
    std::cout << "\nComputing ground truth...\n";
    auto gt = compute_ground_truth(
        dataset.base_vectors.data(), N,
        dataset.query_vectors.data(), n_queries,
        dataset.dim, k);
    std::cout << "Done.\n";

    // Test different K values
    test_k_value<16>(dataset, N, n_queries, gt, k);
    test_k_value<32>(dataset, N, n_queries, gt, k);
    test_k_value<64>(dataset, N, n_queries, gt, k);

    return 0;
}
