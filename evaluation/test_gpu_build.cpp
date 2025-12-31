/**
 * GPU k-NN Build Pipeline Test
 *
 * Tests the full GPU→Flash pipeline:
 * 1. GPU: Tiled brute-force k-NN (CAGRA-style)
 * 2. CPU: Parallel encoding
 * 3. CPU: Parallel Flash ingestion (SoA layout)
 * 4. Search and recall verification
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cphnsw;
using namespace cphnsw::eval;

constexpr size_t K_ROTATIONS = 32;
using Index = CPHNSWIndex<uint8_t, K_ROTATIONS>;

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    double elapsed_s() const { return elapsed_ms() / 1000.0; }
};

void normalize_vectors(std::vector<Float>& vecs, size_t dim) {
    size_t n = vecs.size() / dim;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < n; ++i) {
        Float* v = &vecs[i * dim];
        Float norm_sq = 0;
        for (size_t j = 0; j < dim; ++j) {
            norm_sq += v[j] * v[j];
        }
        Float norm = std::sqrt(norm_sq);
        if (norm > 1e-10f) {
            for (size_t j = 0; j < dim; ++j) {
                v[j] /= norm;
            }
        }
    }
}

std::vector<std::vector<NodeId>> compute_ground_truth(
    const Float* base, size_t n_base,
    const Float* queries, size_t n_queries,
    size_t dim, size_t k) {

    std::vector<std::vector<NodeId>> gt(n_queries);

    std::cout << "Computing ground truth..." << std::flush;
    Timer timer;
    timer.start();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
#endif
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

    std::cout << " done in " << std::fixed << std::setprecision(1)
              << timer.elapsed_s() << "s\n";
    return gt;
}

float compute_recall(const std::vector<SearchResult>& results,
                     const std::vector<NodeId>& gt, size_t k) {
    std::unordered_set<NodeId> gt_set(gt.begin(), gt.begin() + k);
    size_t hits = 0;
    for (size_t i = 0; i < std::min(results.size(), k); ++i) {
        if (gt_set.count(results[i].id)) {
            hits++;
        }
    }
    return static_cast<float>(hits) / k;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "GPU k-NN Build Pipeline Test\n";
    std::cout << "========================================\n\n";

    // System info
    std::cout << "System:\n";
    std::cout << "  CPU threads: " << std::thread::hardware_concurrency() << "\n";
#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
#endif

#ifdef CPHNSW_HAS_CUDA
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "  GPU: " << prop.name << " (" << prop.totalGlobalMem / (1024*1024*1024) << " GB)\n";
#endif
    std::cout << "\n";

    // Load SIFT-1M (already normalized)
    std::cout << "Loading SIFT-1M...\n";
    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");
    std::cout << "  Base: " << dataset.num_base << " x " << dataset.dim << "\n";
    std::cout << "  Queries: " << dataset.num_queries << " x " << dataset.dim << "\n\n";

    // Ground truth
    const size_t k = 10;
    auto gt = compute_ground_truth(
        dataset.base_vectors.data(), dataset.num_base,
        dataset.query_vectors.data(), dataset.num_queries,
        dataset.dim, k);

    // ========================================
    // GPU k-NN Build
    // ========================================
#ifdef CPHNSW_HAS_CUDA
    std::cout << "\n========================================\n";
    std::cout << "GPU k-NN Build (CAGRA-style)\n";
    std::cout << "========================================\n\n";

    CPHNSWParams params;
    params.dim = dataset.dim;
    params.k = K_ROTATIONS;
    params.M = 32;
    params.ef_construction = 200;
    params.finalize();

    Index index(params);

    Timer build_timer;
    build_timer.start();

    // Use GPU k-NN build with pruning
    index.build_with_gpu_knn(
        dataset.base_vectors.data(),
        dataset.num_base,
        64,     // k for k-NN (oversample for pruning)
        true,   // use pruning
        1.1f    // pruning alpha
    );

    double build_time = build_timer.elapsed_s();

    std::cout << "\nBuild Results:\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << build_time << " s\n";
    std::cout << "  Rate: " << static_cast<int>(dataset.num_base / build_time) << " vec/s\n";
    std::cout << "  Connectivity: " << index.verify_connectivity() << "/" << dataset.num_base << "\n";

    // ========================================
    // Search Benchmark
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Search Benchmark\n";
    std::cout << "========================================\n\n";

    std::vector<size_t> ef_values = {50, 100, 200, 500};

    std::cout << std::setw(8) << "ef"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "QPS"
              << std::setw(14) << "Latency(us)\n";
    std::cout << std::string(46, '-') << "\n";

    for (size_t ef : ef_values) {
        Timer search_timer;
        float total_recall = 0;

        search_timer.start();
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            auto results = index.search(dataset.query_vectors.data() + q * dataset.dim, k, ef);
            total_recall += compute_recall(results, gt[q], k);
        }
        double search_time = search_timer.elapsed_ms();

        float avg_recall = total_recall / dataset.num_queries;
        double qps = dataset.num_queries / (search_time / 1000.0);
        double latency_us = (search_time * 1000.0) / dataset.num_queries;

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << static_cast<int>(qps)
                  << std::setw(14) << std::setprecision(1) << latency_us << "\n";
    }

    // ========================================
    // Calibrated Search
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Calibrated Search (FINGER)\n";
    std::cout << "========================================\n\n";

    index.calibrate(1000);
    auto calib = index.get_calibration();
    std::cout << "Calibration: alpha=" << calib.alpha
              << ", beta=" << calib.beta
              << ", R²=" << calib.r_squared << "\n\n";

    std::cout << std::setw(8) << "ef"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "QPS\n";
    std::cout << std::string(32, '-') << "\n";

    for (size_t ef : ef_values) {
        Timer search_timer;
        float total_recall = 0;

        search_timer.start();
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            auto results = index.search_calibrated(
                dataset.query_vectors.data() + q * dataset.dim, k, ef);
            total_recall += compute_recall(results, gt[q], k);
        }
        double search_time = search_timer.elapsed_ms();

        float avg_recall = total_recall / dataset.num_queries;
        double qps = dataset.num_queries / (search_time / 1000.0);

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << static_cast<int>(qps) << "\n";
    }

    // ========================================
    // Comparison with CPU Build
    // ========================================
    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n\n";

    std::cout << "GPU k-NN Build:\n";
    std::cout << "  Build time: " << std::fixed << std::setprecision(1) << build_time << " s\n";
    std::cout << "  (vs ~480s for CPU sequential = " << std::setprecision(0) << (480.0 / build_time) << "x speedup)\n";

#else
    std::cout << "ERROR: CUDA not available. Recompile with -DCPHNSW_USE_CUDA=ON\n";
    return 1;
#endif

    return 0;
}
