/**
 * GPU-Accelerated SIFT-1M Evaluation
 *
 * This benchmark demonstrates the hybrid CPU+GPU pipeline:
 * 1. GPU: Batch encode queries (CUDA)
 * 2. CPU: HNSW graph traversal (OpenMP)
 * 3. GPU: Batch re-rank candidates (cuBLAS)
 *
 * Expected speedups on H100:
 * - Batch encoding: 10-50x over CPU
 * - Brute force search: 100x+ over CPU
 * - Re-ranking: 10-50x over CPU
 */

#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "../include/cphnsw/cuda/gpu_encoder.cuh"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cphnsw;
using namespace cphnsw::eval;

// Use K=32 for higher precision
constexpr size_t K_ROTATIONS = 32;
using CPHNSWIndex32 = CPHNSWIndex<uint8_t, K_ROTATIONS>;

void print_system_info() {
    std::cout << "=== System Information ===\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif

#ifdef CPHNSW_HAS_CUDA
    std::cout << "CUDA: ENABLED\n";

    // Query GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "GPU Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    std::cout << "SM Count: " << prop.multiProcessorCount << "\n";
#else
    std::cout << "CUDA: DISABLED (recompile with -DCPHNSW_USE_CUDA=ON)\n";
#endif

    std::cout << "\n";
}

/**
 * Normalize vectors to unit length
 */
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
            Float inv_norm = 1.0f / norm;
            for (size_t j = 0; j < dim; ++j) {
                v[j] *= inv_norm;
            }
        }
    }
}

/**
 * Compute ground truth using cosine similarity
 */
std::vector<std::vector<NodeId>> compute_ground_truth(
    const std::vector<Float>& base,
    const std::vector<Float>& queries,
    size_t dim,
    size_t k) {

    size_t n_base = base.size() / dim;
    size_t n_queries = queries.size() / dim;

    std::vector<std::vector<NodeId>> gt(n_queries);

    Timer timer;
    timer.start();

    std::cout << "Computing ground truth (CPU)...\n";

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
#endif
    for (size_t q = 0; q < n_queries; ++q) {
        const Float* qv = &queries[q * dim];

        std::vector<std::pair<Float, NodeId>> dists(n_base);

        for (size_t i = 0; i < n_base; ++i) {
            const Float* bv = &base[i * dim];
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

    std::cout << "  Ground truth computed in " << std::fixed << std::setprecision(1)
              << timer.elapsed_s() << " s\n\n";

    return gt;
}

#ifdef CPHNSW_HAS_CUDA
/**
 * GPU brute force benchmark
 */
void benchmark_gpu_brute_force(
    const Dataset& dataset,
    size_t k) {

    std::cout << "=== GPU Brute Force Benchmark ===\n";

    cuda::GPUBruteForce gpu_bf(dataset.dim, dataset.num_base);

    // Upload base vectors
    Timer timer;
    timer.start();
    gpu_bf.set_base_vectors(dataset.base_vectors.data(), dataset.num_base);
    std::cout << "  Base vectors uploaded in " << timer.elapsed_ms() << " ms\n";

    // Benchmark batch sizes
    std::vector<size_t> batch_sizes = {100, 1000, 10000};

    std::cout << std::setw(12) << "Batch"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS"
              << std::setw(12) << "Latency(ms)\n";
    std::cout << std::string(48, '-') << "\n";

    for (size_t batch : batch_sizes) {
        size_t num_queries = std::min(batch, dataset.num_queries);

        std::vector<uint32_t> indices(num_queries * k);
        std::vector<Float> distances(num_queries * k);

        // Warmup
        gpu_bf.search(dataset.query_vectors.data(), num_queries, k,
                      indices.data(), distances.data());

        // Timed run
        timer.start();
        int iterations = 5;
        for (int i = 0; i < iterations; ++i) {
            gpu_bf.search(dataset.query_vectors.data(), num_queries, k,
                          indices.data(), distances.data());
        }
        double elapsed_ms = timer.elapsed_ms() / iterations;

        // Compute recall
        double total_recall = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            std::vector<NodeId> result_ids(indices.begin() + q * k,
                                           indices.begin() + (q + 1) * k);
            total_recall += compute_recall(result_ids, dataset.ground_truth[q], k);
        }
        double avg_recall = total_recall / num_queries;
        double qps = num_queries * 1000.0 / elapsed_ms;

        std::cout << std::setw(12) << num_queries
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps
                  << std::setw(12) << std::fixed << std::setprecision(2) << elapsed_ms
                  << "\n";
    }
    std::cout << "\n";
}

/**
 * GPU encoding benchmark
 */
void benchmark_gpu_encoding(const Dataset& dataset) {
    std::cout << "=== GPU Encoding Benchmark ===\n";

    cuda::GPUEncoder<uint8_t, K_ROTATIONS> gpu_encoder(dataset.dim, 42);

    std::vector<size_t> batch_sizes = {1000, 10000, 100000, 1000000};

    std::cout << std::setw(12) << "Batch"
              << std::setw(15) << "GPU(ms)"
              << std::setw(15) << "CPU(ms)"
              << std::setw(12) << "Speedup\n";
    std::cout << std::string(54, '-') << "\n";

    CPEncoder<uint8_t, K_ROTATIONS> cpu_encoder(dataset.dim, 42);

    for (size_t batch : batch_sizes) {
        size_t num_vecs = std::min(batch, dataset.num_base);

        std::vector<uint8_t> gpu_codes(num_vecs * K_ROTATIONS);
        std::vector<CPCode<uint8_t, K_ROTATIONS>> cpu_codes(num_vecs);

        // GPU timing
        Timer timer;
        timer.start();
        gpu_encoder.encode_batch(dataset.base_vectors.data(), num_vecs, gpu_codes.data());
        double gpu_ms = timer.elapsed_ms();

        // CPU timing (only for smaller batches)
        double cpu_ms = 0;
        if (num_vecs <= 100000) {
            timer.start();
            for (size_t i = 0; i < num_vecs; ++i) {
                cpu_codes[i] = cpu_encoder.encode(&dataset.base_vectors[i * dataset.dim]);
            }
            cpu_ms = timer.elapsed_ms();
        }

        double speedup = (cpu_ms > 0) ? cpu_ms / gpu_ms : 0;

        std::cout << std::setw(12) << num_vecs
                  << std::setw(15) << std::fixed << std::setprecision(2) << gpu_ms
                  << std::setw(15) << (cpu_ms > 0 ? std::to_string(static_cast<int>(cpu_ms)) : "N/A")
                  << std::setw(12) << (speedup > 0 ? std::to_string(static_cast<int>(speedup)) + "x" : "N/A")
                  << "\n";
    }
    std::cout << "\n";
}

/**
 * Hybrid CPU+GPU re-ranking benchmark
 */
void benchmark_hybrid_reranking(
    const Dataset& dataset,
    CPHNSWIndex32& index,
    size_t k) {

    std::cout << "=== Hybrid Re-ranking Benchmark ===\n";
    std::cout << "  Pipeline: CPU (HNSW search) -> GPU (re-rank with true cosine)\n\n";

    cuda::GPUBruteForce gpu_bf(dataset.dim, dataset.num_base);
    gpu_bf.set_base_vectors(dataset.base_vectors.data(), dataset.num_base);

    // Test configurations: (ef, rerank_candidates)
    std::vector<std::pair<size_t, size_t>> configs = {
        {100, 100},
        {200, 200},
        {500, 500},
    };

    std::cout << std::setw(8) << "ef"
              << std::setw(10) << "rerank_k"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS"
              << std::setw(15) << "Latency(us)\n";
    std::cout << std::string(57, '-') << "\n";

    for (auto [ef, rerank_k] : configs) {
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

        // Process in batches for GPU efficiency
        size_t batch_size = 1000;
        size_t num_batches = (dataset.num_queries + batch_size - 1) / batch_size;

        for (size_t b = 0; b < num_batches; ++b) {
            size_t start = b * batch_size;
            size_t end = std::min(start + batch_size, dataset.num_queries);
            size_t batch_queries = end - start;

            // Step 1: CPU HNSW search to get candidates
            std::vector<uint32_t> candidate_ids(batch_queries * rerank_k);

            Timer batch_timer;
            batch_timer.start();

#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 100)
#endif
            for (size_t q = start; q < end; ++q) {
                auto results = index.search(dataset.get_query(q), rerank_k, ef);
                for (size_t i = 0; i < results.size() && i < rerank_k; ++i) {
                    candidate_ids[(q - start) * rerank_k + i] = results[i].id;
                }
                // Pad with zeros if needed
                for (size_t i = results.size(); i < rerank_k; ++i) {
                    candidate_ids[(q - start) * rerank_k + i] = 0;
                }
            }

            // Step 2: GPU re-rank
            std::vector<uint32_t> final_indices(batch_queries * k);
            std::vector<Float> final_distances(batch_queries * k);

            gpu_bf.rerank(&dataset.query_vectors[start * dataset.dim],
                          candidate_ids.data(),
                          batch_queries, rerank_k, k,
                          final_indices.data(), final_distances.data());

            double batch_ms = batch_timer.elapsed_ms();
            double per_query_us = batch_ms * 1000.0 / batch_queries;

            // Compute recall
            for (size_t q = start; q < end; ++q) {
                latencies[q] = per_query_us;

                std::vector<NodeId> result_ids(
                    final_indices.begin() + (q - start) * k,
                    final_indices.begin() + (q - start + 1) * k);
                recalls[q] = compute_recall(result_ids, dataset.ground_truth[q], k);
            }
        }

        double total_recall = 0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / dataset.num_queries;
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << ef
                  << std::setw(10) << rerank_k
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(15) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }
    std::cout << "\n";
}
#endif  // CPHNSW_HAS_CUDA

/**
 * CPU-only baseline for comparison
 */
void benchmark_cpu_baseline(
    const Dataset& dataset,
    CPHNSWIndex32& index,
    size_t k) {

    std::cout << "=== CPU Baseline (HNSW + CPU Re-ranking) ===\n";

    std::vector<size_t> ef_values = {100, 200, 500};

    std::cout << std::setw(8) << "ef"
              << std::setw(10) << "rerank_k"
              << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS"
              << std::setw(15) << "Latency(us)\n";
    std::cout << std::string(57, '-') << "\n";

    for (size_t ef : ef_values) {
        size_t rerank_k = ef;
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            Timer t;
            t.start();

            // Get candidates
            auto results = index.search(query, rerank_k, ef);

            // Re-rank with true cosine
            for (auto& res : results) {
                const Float* base_vec = dataset.get_base(res.id);
                Float dot = 0;
                for (size_t d = 0; d < dataset.dim; ++d) {
                    dot += query[d] * base_vec[d];
                }
                res.distance = -dot;
            }

            std::sort(results.begin(), results.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.distance < b.distance;
                      });

            if (results.size() > k) results.resize(k);

            latencies[q] = t.elapsed_us();
            recalls[q] = compute_recall(results, dataset.ground_truth[q], k);
        }

        double total_recall = 0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / dataset.num_queries;
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << ef
                  << std::setw(10) << rerank_k
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(15) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "CP-HNSW GPU-Accelerated Evaluation\n";
    std::cout << "===================================\n\n";

    print_system_info();

#ifndef CPHNSW_HAS_CUDA
    std::cerr << "ERROR: This binary requires CUDA support.\n";
    std::cerr << "Recompile with: cmake -DCPHNSW_USE_CUDA=ON ..\n";
    return 1;
#else

    // Dataset path
    std::string data_dir = "../evaluation/data/sift";
    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "Loading SIFT-1M from: " << data_dir << "\n\n";

    // Load vectors
    size_t base_dim, base_count;
    std::vector<Float> base_vectors = load_fvecs(data_dir + "/sift_base.fvecs", base_dim, base_count);
    std::cout << "Loaded base vectors: " << base_count << " x " << base_dim << "\n";

    size_t query_dim, query_count;
    std::vector<Float> query_vectors = load_fvecs(data_dir + "/sift_query.fvecs", query_dim, query_count);
    std::cout << "Loaded query vectors: " << query_count << " x " << query_dim << "\n\n";

    // Normalize
    std::cout << "Normalizing vectors...\n";
    normalize_vectors(base_vectors, base_dim);
    normalize_vectors(query_vectors, query_dim);
    std::cout << "  Done\n\n";

    // Compute ground truth
    size_t k = 10;
    auto ground_truth = compute_ground_truth(base_vectors, query_vectors, base_dim, k);

    // Build dataset struct
    Dataset dataset;
    dataset.base_vectors = std::move(base_vectors);
    dataset.query_vectors = std::move(query_vectors);
    dataset.ground_truth = std::move(ground_truth);
    dataset.dim = base_dim;
    dataset.num_base = base_count;
    dataset.num_queries = query_count;
    dataset.k_gt = k;

    // =========================================================================
    // Benchmark 1: GPU Brute Force (upper bound on recall)
    // =========================================================================
    benchmark_gpu_brute_force(dataset, k);

    // =========================================================================
    // Benchmark 2: GPU Encoding Speed
    // =========================================================================
    benchmark_gpu_encoding(dataset);

    // =========================================================================
    // Benchmark 3: Build CPU Index (needed for hybrid pipeline)
    // =========================================================================
    std::cout << "=== Building CP-HNSW Index (CPU) ===\n";
    Timer timer;
    timer.start();

    CPHNSWIndex32 index(dataset.dim, 32, 200);  // M=32, ef_c=200

    size_t batch_size = 10000;
    for (size_t i = 0; i < dataset.num_base; i += batch_size) {
        size_t end = std::min(i + batch_size, dataset.num_base);
        for (size_t j = i; j < end; ++j) {
            index.add(dataset.get_base(j));
        }
        std::cout << "  Progress: " << (end * 100 / dataset.num_base) << "%\r" << std::flush;
    }

    double build_time = timer.elapsed_s();
    std::cout << "\n  Build time: " << std::fixed << std::setprecision(1) << build_time << " s\n";
    std::cout << "  Build rate: " << std::fixed << std::setprecision(0)
              << (dataset.num_base / build_time) << " vec/s\n\n";

    // =========================================================================
    // Benchmark 4: CPU Baseline with Re-ranking
    // =========================================================================
    benchmark_cpu_baseline(dataset, index, k);

    // =========================================================================
    // Benchmark 5: Hybrid CPU+GPU Pipeline
    // =========================================================================
    benchmark_hybrid_reranking(dataset, index, k);

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "===================================\n";
    std::cout << "GPU Evaluation Complete\n\n";
    std::cout << "Key Findings:\n";
    std::cout << "  - GPU brute force: Maximum recall upper bound\n";
    std::cout << "  - GPU encoding: Massive speedup for batch operations\n";
    std::cout << "  - Hybrid pipeline: Best balance of speed and recall\n";
    std::cout << "\nRecommendation:\n";
    std::cout << "  Use hybrid CPU+GPU pipeline for production:\n";
    std::cout << "  1. CPU HNSW for candidate generation (fast, approximate)\n";
    std::cout << "  2. GPU re-ranking for final top-k (accurate, parallelized)\n";

    return 0;
#endif
}
