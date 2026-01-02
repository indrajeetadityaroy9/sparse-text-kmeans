/**
 * GPU-Accelerated Benchmark for 2x H100 Configuration.
 *
 * Uses cuBLAS for brute-force similarity computation.
 * Achieves maximum throughput by batching all queries.
 *
 * Usage:
 *   ./eval_gpu_benchmark --sift /path/to/sift --output results/
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "../include/cphnsw/cuda/gpu_search.cuh"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include "utils/common.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

using namespace cphnsw;
using namespace cphnsw::eval;
using namespace cphnsw::cuda;

// Configuration - K can be set at compile time with -DEVAL_K=XX
#ifndef EVAL_K
#define EVAL_K 32
#endif
constexpr size_t COMPILE_TIME_K = EVAL_K;

struct BenchmarkConfig {
    std::string sift_dir;
    std::string gist_dir;
    std::string output_dir = "results";
    size_t M = 32;
    size_t ef_construction = 200;
    size_t limit = 0;
    int num_gpus = 2;
};

void print_usage() {
    std::cout << R"(
GPU-Accelerated CP-HNSW Benchmark
=================================

Usage: eval_gpu_benchmark [options]

Options:
  --sift <path>     Path to SIFT-1M directory
  --gist <path>     Path to GIST-1M directory
  --output <path>   Output directory (default: results/)
  --M <value>       HNSW M parameter (default: 32)
  --ef_c <value>    ef_construction (default: 200)
  --limit <value>   Limit base vectors (for testing)
  --gpus <value>    Number of GPUs (default: 2)
)";
}

// Use Timer from metrics/recall.hpp

void run_benchmark(const Dataset& dataset, const BenchmarkConfig& config) {
    Timer timer;

    std::cout << "\n=== GPU-Accelerated Benchmark ===\n";
    std::cout << "Base vectors: " << dataset.num_base << " x " << dataset.dim << "\n";
    std::cout << "Queries: " << dataset.num_queries << "\n";
    std::cout << "GPUs: " << config.num_gpus << "\n\n";

    // ===========================================================
    // Phase 1: GPU Ground Truth Computation
    // ===========================================================
    std::cout << "[Phase 1] Computing ground truth on GPU...\n";
    timer.start();

    GPUBatchSearch gpu_search(dataset.dim, config.num_gpus);
    gpu_search.set_base_vectors(dataset.base_vectors.data(), dataset.num_base);

    std::vector<uint32_t> gpu_gt_indices(dataset.num_queries * 10);
    std::vector<float> gpu_gt_distances(dataset.num_queries * 10);

    gpu_search.search_batch(
        dataset.query_vectors.data(),
        dataset.num_queries,
        10,
        gpu_gt_indices.data(),
        gpu_gt_distances.data()
    );

    double gt_time = timer.elapsed_s();
    std::cout << "  Ground truth computed in " << std::fixed << std::setprecision(2)
              << gt_time << "s\n";
    std::cout << "  Throughput: " << std::setprecision(0)
              << dataset.num_queries / gt_time << " QPS\n\n";

    // Convert to ground truth format
    std::vector<std::vector<NodeId>> ground_truth(dataset.num_queries);
    for (size_t q = 0; q < dataset.num_queries; ++q) {
        ground_truth[q].resize(10);
        for (size_t i = 0; i < 10; ++i) {
            ground_truth[q][i] = static_cast<NodeId>(gpu_gt_indices[q * 10 + i]);
        }
    }

    // ===========================================================
    // Phase 2: Build CP-HNSW Index
    // ===========================================================
    std::cout << "[Phase 2] Building CP-HNSW index (M=" << config.M
              << ", K=" << COMPILE_TIME_K << ", ef_c=" << config.ef_construction << ")...\n";
    timer.start();

    CPHNSWIndex<uint8_t, COMPILE_TIME_K> index(dataset.dim, config.M, config.ef_construction);
    index.build_with_gpu_knn(dataset.base_vectors.data(), dataset.num_base, 64);

    double build_time = timer.elapsed_s();
    std::cout << "  Build time: " << std::fixed << std::setprecision(2) << build_time << "s ("
              << std::setprecision(0) << dataset.num_base / build_time << " vec/s)\n" << std::flush;

    std::cout << "  Verifying connectivity..." << std::flush;
    size_t connectivity = index.verify_connectivity();
    std::cout << " done\n" << std::flush;

    std::cout << "  Connectivity: " << connectivity << "/" << dataset.num_base
              << " (" << std::setprecision(2) << 100.0 * connectivity / dataset.num_base << "%)\n\n" << std::flush;

    // ===========================================================
    // Phase 3: GPU Brute Force Benchmark (Baseline)
    // ===========================================================
    std::cout << "[Phase 3] GPU Brute Force Search (Baseline)...\n" << std::flush;

    std::vector<uint32_t> bf_indices(dataset.num_queries * 10);
    std::vector<float> bf_distances(dataset.num_queries * 10);

    // Warmup
    std::cout << "  Warmup..." << std::flush;
    gpu_search.search_batch(dataset.query_vectors.data(), 100, 10,
                           bf_indices.data(), bf_distances.data());
    std::cout << " done\n" << std::flush;

    // Benchmark
    timer.start();
    gpu_search.search_batch(
        dataset.query_vectors.data(),
        dataset.num_queries,
        10,
        bf_indices.data(),
        bf_distances.data()
    );
    double bf_time = timer.elapsed_s();
    double bf_qps = dataset.num_queries / bf_time;

    // Compute recall (should be 100% for brute force)
    double bf_recall = 0;
    for (size_t q = 0; q < dataset.num_queries; ++q) {
        std::vector<NodeId> results(10);
        for (size_t i = 0; i < 10; ++i) {
            results[i] = static_cast<NodeId>(bf_indices[q * 10 + i]);
        }
        bf_recall += compute_recall(results, ground_truth[q], 10);
    }
    bf_recall /= dataset.num_queries;

    std::cout << "  QPS: " << std::fixed << std::setprecision(0) << bf_qps << "\n";
    std::cout << "  Recall@10: " << std::setprecision(4) << bf_recall << "\n";
    std::cout << "  Latency: " << std::setprecision(2) << 1e6 / bf_qps << " us/query\n\n";

    // ===========================================================
    // Phase 4: CP-HNSW Graph Search Benchmark
    // ===========================================================
    std::cout << "[Phase 4] CP-HNSW Graph Search...\n\n";

    std::cout << std::setw(8) << "ef"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "QPS"
              << std::setw(14) << "Latency(us)"
              << std::setw(12) << "Speedup\n";
    std::cout << std::string(60, '-') << "\n";

    std::vector<size_t> ef_values = {10, 20, 40, 80, 100, 200, 400};

    for (size_t ef : ef_values) {
        double total_recall = 0;

        timer.start();
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            auto results = index.search(dataset.query_vectors.data() + q * dataset.dim, 10, ef);
            std::vector<NodeId> result_ids;
            for (const auto& r : results) {
                result_ids.push_back(r.id);
            }
            total_recall += compute_recall(result_ids, ground_truth[q], 10);
        }
        double search_time = timer.elapsed_s();

        double qps = dataset.num_queries / search_time;
        double recall = total_recall / dataset.num_queries;
        double speedup = qps / bf_qps;

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(4) << recall
                  << std::setw(12) << std::setprecision(0) << qps
                  << std::setw(14) << std::setprecision(1) << 1e6 / qps
                  << std::setw(12) << std::setprecision(1) << speedup << "x\n";
    }

    // ===========================================================
    // Phase 5: CP-HNSW + Rerank Benchmark
    // ===========================================================
    std::cout << "\n[Phase 5] CP-HNSW + GPU Rerank...\n\n";

    std::cout << std::setw(8) << "ef"
              << std::setw(10) << "rerank_k"
              << std::setw(12) << "Recall@10"
              << std::setw(12) << "QPS"
              << std::setw(14) << "Latency(us)\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<size_t> rerank_values = {50, 100, 200};

    for (size_t ef : {100, 200}) {
        for (size_t rerank_k : rerank_values) {
            double total_recall = 0;

            timer.start();
            for (size_t q = 0; q < dataset.num_queries; ++q) {
                auto results = index.search_and_rerank(
                    dataset.query_vectors.data() + q * dataset.dim,
                    10, ef, rerank_k
                );
                std::vector<NodeId> result_ids;
                for (const auto& r : results) {
                    result_ids.push_back(r.id);
                }
                total_recall += compute_recall(result_ids, ground_truth[q], 10);
            }
            double search_time = timer.elapsed_s();

            double qps = dataset.num_queries / search_time;
            double recall = total_recall / dataset.num_queries;

            std::cout << std::setw(8) << ef
                      << std::setw(10) << rerank_k
                      << std::setw(12) << std::fixed << std::setprecision(4) << recall
                      << std::setw(12) << std::setprecision(0) << qps
                      << std::setw(14) << std::setprecision(1) << 1e6 / qps << "\n";
        }
    }

    std::cout << "\n=== Benchmark Complete ===\n";
}

int main(int argc, char** argv) {
    BenchmarkConfig config;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sift" && i + 1 < argc) {
            config.sift_dir = argv[++i];
        } else if (arg == "--gist" && i + 1 < argc) {
            config.gist_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--M" && i + 1 < argc) {
            config.M = std::stoul(argv[++i]);
        } else if (arg == "--ef_c" && i + 1 < argc) {
            config.ef_construction = std::stoul(argv[++i]);
        } else if (arg == "--limit" && i + 1 < argc) {
            config.limit = std::stoul(argv[++i]);
        } else if (arg == "--gpus" && i + 1 < argc) {
            config.num_gpus = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }

    std::cout << "=== GPU-Accelerated CP-HNSW Benchmark ===\n\n";

    // Print GPU info
    int num_gpus = get_num_gpus();
    std::cout << "Available GPUs: " << num_gpus << "\n";
    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name
                  << " (" << prop.totalGlobalMem / (1024*1024*1024) << " GB)\n";
    }
    std::cout << "\n";

    // Load dataset
    Dataset dataset;
    bool have_data = false;

    if (!config.sift_dir.empty()) {
        std::cout << "Loading SIFT-1M from: " << config.sift_dir << "\n";
        try {
            dataset = load_sift1m_normalized(config.sift_dir);
            std::cout << "  Loaded " << dataset.num_base << " base, "
                      << dataset.num_queries << " queries, dim=" << dataset.dim << "\n";

            if (config.limit > 0 && config.limit < dataset.num_base) {
                dataset.num_base = config.limit;
                dataset.base_vectors.resize(dataset.num_base * dataset.dim);
                std::cout << "  Limited to " << dataset.num_base << " base vectors\n";
            }
            have_data = true;
        } catch (const std::exception& e) {
            std::cerr << "  Error: " << e.what() << "\n";
        }
    }

    if (!config.gist_dir.empty()) {
        std::cout << "Loading GIST-1M from: " << config.gist_dir << "\n";
        try {
            dataset = load_gist1m(config.gist_dir);
            std::cout << "  Loaded " << dataset.num_base << " base, "
                      << dataset.num_queries << " queries, dim=" << dataset.dim << "\n";

            if (config.limit > 0 && config.limit < dataset.num_base) {
                dataset.num_base = config.limit;
                dataset.base_vectors.resize(dataset.num_base * dataset.dim);
                std::cout << "  Limited to " << dataset.num_base << " base vectors\n";
            }
            have_data = true;
        } catch (const std::exception& e) {
            std::cerr << "  Error: " << e.what() << "\n";
        }
    }

    if (!have_data) {
        std::cerr << "\nNo dataset loaded. Use --sift or --gist to specify path.\n";
        print_usage();
        return 1;
    }

    run_benchmark(dataset, config);

    return 0;
}
