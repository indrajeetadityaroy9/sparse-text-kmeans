#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "../include/cphnsw/quantizer/cp_encoder.hpp"
#include "../include/cphnsw/distance/hamming.hpp"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cphnsw;
using namespace cphnsw::eval;

// Use K=32 for better gradient resolution (32 bytes per code)
// This provides 2x the resolution vs K=16, still 16x compression vs floats
using CPHNSWIndex32 = CPHNSWIndex<uint8_t, 32>;
using Encoder32 = CPEncoder<uint8_t, 32>;

/**
 * Correlation Test: Measure how well the CP estimator correlates with true cosine similarity.
 *
 * This diagnostic helps identify whether low recall is due to:
 * - High variance in the estimator (correlation 0.3-0.7)
 * - Fundamental issues with the encoding (correlation < 0.3)
 * - Or something else (correlation > 0.7)
 */
void check_correlation(
    const std::vector<Float>& queries,
    const std::vector<Float>& base,
    size_t dim,
    size_t samples = 1000) {

    std::cout << "=== Estimator Correlation Test ===\n";

    // Create encoder
    Encoder32 encoder(dim, 42);

    std::vector<float> true_sims, est_sims;
    true_sims.reserve(samples);
    est_sims.reserve(samples);

    std::mt19937 rng(42);

    size_t n_queries = queries.size() / dim;
    size_t n_base = base.size() / dim;

    for (size_t i = 0; i < samples; ++i) {
        size_t q_idx = rng() % n_queries;
        size_t b_idx = rng() % n_base;

        // True cosine similarity (dot product on normalized vectors)
        float true_sim = 0;
        for (size_t d = 0; d < dim; ++d) {
            true_sim += queries[q_idx * dim + d] * base[b_idx * dim + d];
        }

        // CP estimator
        auto q_cp = encoder.encode_query(&queries[q_idx * dim]);
        auto b_code = encoder.encode(&base[b_idx * dim]);
        float est_sim = -asymmetric_search_distance(q_cp, b_code);  // Negate to get similarity

        true_sims.push_back(true_sim);
        est_sims.push_back(est_sim);
    }

    // Compute Pearson correlation
    float mean_true = 0, mean_est = 0;
    for (size_t i = 0; i < samples; ++i) {
        mean_true += true_sims[i];
        mean_est += est_sims[i];
    }
    mean_true /= static_cast<float>(samples);
    mean_est /= static_cast<float>(samples);

    float cov = 0, var_true = 0, var_est = 0;
    for (size_t i = 0; i < samples; ++i) {
        float dt = true_sims[i] - mean_true;
        float de = est_sims[i] - mean_est;
        cov += dt * de;
        var_true += dt * dt;
        var_est += de * de;
    }

    float correlation = cov / std::sqrt(var_true * var_est);

    // Find min/max
    float true_min = *std::min_element(true_sims.begin(), true_sims.end());
    float true_max = *std::max_element(true_sims.begin(), true_sims.end());
    float est_min = *std::min_element(est_sims.begin(), est_sims.end());
    float est_max = *std::max_element(est_sims.begin(), est_sims.end());

    std::cout << "  Samples: " << samples << "\n";
    std::cout << "  True sim range: [" << std::fixed << std::setprecision(4)
              << true_min << ", " << true_max << "]\n";
    std::cout << "  Est sim range: [" << std::fixed << std::setprecision(2)
              << est_min << ", " << est_max << "]\n";
    std::cout << "  Pearson Correlation: " << std::fixed << std::setprecision(4)
              << correlation << "\n\n";

    // Interpretation
    if (correlation > 0.7) {
        std::cout << "  INTERPRETATION: Estimator is usable (r > 0.7)\n";
        std::cout << "  Issue is likely in graph navigation, not hash quality.\n";
    } else if (correlation > 0.3) {
        std::cout << "  INTERPRETATION: High variance estimator (0.3 < r < 0.7)\n";
        std::cout << "  RECOMMENDATION: Use hybrid construction (true distance for edges)\n";
    } else {
        std::cout << "  INTERPRETATION: Fundamental issue (r < 0.3)\n";
        std::cout << "  RECOMMENDATION: Try K=64 or investigate encoding logic.\n";
    }

    // Print sample pairs
    std::cout << "\n  Sample pairs (True vs Est):\n";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << true_sims[i]
                  << " vs " << std::setprecision(2) << est_sims[i] << "\n";
    }
    std::cout << "\n";
}

/**
 * SIFT-1M Evaluation for CP-HNSW
 *
 * CRITICAL: SIFT vectors are integer histograms (0-255), NOT unit-length.
 * The downloaded ground truth uses L2 distance on unnormalized vectors.
 *
 * For angular distance (cosine similarity), we must:
 * 1. Normalize all vectors to unit length
 * 2. Recompute ground truth using cosine similarity (dot product on normalized vectors)
 */

void print_system_info() {
    std::cout << "=== System Information ===\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP: DISABLED (ground truth will be slow!)\n";
#endif

    std::cout << "Compilation: ";
#ifdef __AVX512F__
    std::cout << "AVX-512 ";
#endif
#ifdef __AVX2__
    std::cout << "AVX2 ";
#endif
    std::cout << "\n\n";
}

/**
 * Normalize vectors to unit length (L2 normalization).
 * SIFT vectors are 0-255 histograms, this converts them for cosine similarity.
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
 * Compute ground truth using cosine similarity (dot product on normalized vectors).
 *
 * NOTE: 10K queries x 1M vectors x 128 dims = 1.28 trillion FLOPs
 * Without OpenMP: 10-20 minutes. With OpenMP: tens of seconds.
 */
std::vector<std::vector<NodeId>> compute_cosine_ground_truth(
    const std::vector<Float>& base,
    const std::vector<Float>& queries,
    size_t dim,
    size_t k) {

    size_t n_base = base.size() / dim;
    size_t n_queries = queries.size() / dim;

    std::vector<std::vector<NodeId>> gt(n_queries);

    Timer timer;
    timer.start();

    std::cout << "Computing ground truth (cosine similarity)...\n";
    std::cout << "  " << n_queries << " queries x " << n_base << " vectors x " << dim << " dims\n";

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 10)
#endif
    for (size_t q = 0; q < n_queries; ++q) {
        const Float* qv = &queries[q * dim];

        // Compute dot products with all base vectors
        std::vector<std::pair<Float, NodeId>> dists(n_base);

        for (size_t i = 0; i < n_base; ++i) {
            const Float* bv = &base[i * dim];
            Float dot = 0;
            for (size_t j = 0; j < dim; ++j) {
                dot += qv[j] * bv[j];
            }
            // Negative dot product: higher similarity = lower "distance"
            dists[i] = {-dot, static_cast<NodeId>(i)};
        }

        // Partial sort to get top k
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        gt[q].resize(k);
        for (size_t i = 0; i < k; ++i) {
            gt[q][i] = dists[i].second;
        }

        // Progress reporting (from thread 0 only)
#ifdef _OPENMP
        if (omp_get_thread_num() == 0 && q % 500 == 0) {
            std::cout << "  Progress: " << q * omp_get_num_threads() * 100 / n_queries << "%\r" << std::flush;
        }
#else
        if (q % 100 == 0) {
            std::cout << "  Progress: " << q * 100 / n_queries << "%\r" << std::flush;
        }
#endif
    }

    double elapsed = timer.elapsed_s();
    std::cout << "  Ground truth computed in " << std::fixed << std::setprecision(1) << elapsed << " s\n\n";

    return gt;
}

/**
 * Run evaluation with given parameters.
 * Now uses K=32 and asymmetric distance for proper gradient navigation.
 */
void run_sift_evaluation(
    const Dataset& dataset,
    size_t M,
    size_t ef_construction,
    size_t k,
    size_t K_rotations = 32) {

    std::cout << "=== Building CP-HNSW Index (K=32, Asymmetric Distance) ===\n";
    std::cout << "  N=" << dataset.num_base << ", dim=" << dataset.dim
              << ", M=" << M << ", ef_c=" << ef_construction << "\n\n";

    Timer timer;
    timer.start();

    CPHNSWIndex32 index(dataset.dim, M, ef_construction);

    // Progress reporting for large builds
    size_t batch_size = 10000;
    size_t num_batches = (dataset.num_base + batch_size - 1) / batch_size;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start = batch * batch_size;
        size_t end = std::min(start + batch_size, dataset.num_base);

        for (size_t i = start; i < end; ++i) {
            index.add(dataset.get_base(i));
        }

        if ((batch + 1) % 10 == 0 || batch + 1 == num_batches) {
            double progress = 100.0 * (batch + 1) / num_batches;
            double elapsed_s = timer.elapsed_s();
            double rate = (end) / elapsed_s;
            std::cout << "  Progress: " << std::fixed << std::setprecision(0)
                      << progress << "% (" << rate << " vec/s)\r" << std::flush;
        }
    }

    double build_time = timer.elapsed_s();
    std::cout << "\n  Build time: " << std::fixed << std::setprecision(2) << build_time << " s\n";
    std::cout << "  Build rate: " << std::fixed << std::setprecision(0)
              << (dataset.num_base / build_time) << " vectors/s\n";

    // Memory estimate
    size_t code_bytes = dataset.num_base * K_rotations;
    size_t graph_bytes = dataset.num_base * M * 2 * sizeof(uint32_t);
    size_t total_bytes = code_bytes + graph_bytes;
    std::cout << "  Index memory: ~" << total_bytes / (1024*1024) << " MB\n";
    std::cout << "  Compression: " << std::fixed << std::setprecision(1)
              << (dataset.num_base * dataset.dim * sizeof(float)) / static_cast<double>(total_bytes) << "x\n";

    // Connectivity check
    timer.start();
    size_t connected = index.verify_connectivity();
    std::cout << "  Connectivity: " << connected << "/" << dataset.num_base
              << (connected == dataset.num_base ? " (fully connected)" : " (DISCONNECTED!)")
              << " [" << std::fixed << std::setprecision(1) << timer.elapsed_ms() << " ms]\n\n";

    // BRUTE FORCE DIAGNOSTIC: Verify hash quality independently of graph
    std::cout << "=== Brute Force Diagnostic (Hash Quality Check) ===\n";
    std::cout << "  Testing asymmetric distance on first 100 queries...\n";
    {
        size_t bf_queries = std::min(static_cast<size_t>(100), dataset.num_queries);
        std::vector<double> bf_recalls(bf_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 4)
#endif
        for (size_t q = 0; q < bf_queries; ++q) {
            const Float* query = dataset.get_query(q);
            auto bf_results = index.brute_force_search(query, k);

            std::vector<NodeId> bf_ids;
            for (const auto& r : bf_results) {
                bf_ids.push_back(r.id);
            }
            bf_recalls[q] = compute_recall(bf_ids, dataset.ground_truth[q], k);
        }

        double bf_total_recall = 0.0;
        for (size_t q = 0; q < bf_queries; ++q) {
            bf_total_recall += bf_recalls[q];
        }
        double bf_avg_recall = bf_total_recall / static_cast<double>(bf_queries);
        std::cout << "  Brute Force Recall@" << k << " (asymmetric dist): "
                  << std::fixed << std::setprecision(4) << bf_avg_recall << "\n";

        if (bf_avg_recall < 0.50) {
            std::cout << "  WARNING: Low brute force recall suggests hash quality issues.\n";
            std::cout << "           The CP-LSH encoding may not preserve similarity well.\n";
        } else {
            std::cout << "  GOOD: Brute force recall > 0.50 indicates hash quality is adequate.\n";
            std::cout << "        Graph navigation should work with asymmetric distance.\n";
        }
        std::cout << "\n";
    }

    // BRUTE FORCE MULTIPROBE TEST: Check if information exists in adjacent buckets
    std::cout << "=== Brute Force Multiprobe Test (Information Check) ===\n";
    std::cout << "  Testing if true neighbors are in top-N CP candidates...\n";
    {
        size_t bf_queries = std::min(static_cast<size_t>(100), dataset.num_queries);
        std::vector<size_t> top_n_values = {10, 20, 50, 100, 200, 500};

        std::cout << std::setw(10) << "Top-N" << std::setw(15) << "Recall@10\n";
        std::cout << std::string(25, '-') << "\n";

        for (size_t top_n : top_n_values) {
            std::vector<double> recalls(bf_queries);

#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 4)
#endif
            for (size_t q = 0; q < bf_queries; ++q) {
                const Float* query = dataset.get_query(q);
                // Get top-N by CP distance (simulates multiprobe effect)
                auto bf_results = index.brute_force_search(query, top_n);

                std::vector<NodeId> bf_ids;
                for (const auto& r : bf_results) {
                    bf_ids.push_back(r.id);
                }

                // How many of the true top-10 are in the top-N CP results?
                size_t hits = 0;
                for (size_t i = 0; i < std::min(k, dataset.ground_truth[q].size()); ++i) {
                    NodeId true_neighbor = dataset.ground_truth[q][i];
                    if (std::find(bf_ids.begin(), bf_ids.end(), true_neighbor) != bf_ids.end()) {
                        ++hits;
                    }
                }
                recalls[q] = static_cast<double>(hits) / static_cast<double>(k);
            }

            double total_recall = 0.0;
            for (size_t q = 0; q < bf_queries; ++q) {
                total_recall += recalls[q];
            }
            double avg_recall = total_recall / static_cast<double>(bf_queries);
            std::cout << std::setw(10) << top_n
                      << std::setw(15) << std::fixed << std::setprecision(4) << avg_recall << "\n";
        }

        std::cout << "\n  INTERPRETATION:\n";
        std::cout << "    - Top-10 recall ~25%: Single probe only finds 1 in 4 neighbors\n";
        std::cout << "    - Top-100 recall >60%: Information exists, need multiprobe during search\n";
        std::cout << "    - Top-100 recall <40%: K=32 too coarse, need K=64\n";
        std::cout << "\n";
    }

    // Evaluate recall vs ef
    std::cout << "=== Recall@" << k << " vs ef ===\n";
    std::cout << std::setw(8) << "ef"
              << std::setw(12) << "Recall"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Mean(us)"
              << std::setw(12) << "P50(us)"
              << std::setw(12) << "P99(us)\n";
    std::cout << std::string(68, '-') << "\n";

    std::vector<size_t> ef_values = {10, 20, 50, 100, 200, 500, 1000};

    for (size_t ef : ef_values) {
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            Timer t;
            t.start();
            auto results = index.search(query, k, ef);
            latencies[q] = t.elapsed_us();
            recalls[q] = compute_recall(results, dataset.ground_truth[q], k);
        }

        double total_recall = 0.0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / static_cast<double>(dataset.num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_p50_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_p99_us
                  << "\n";
    }

    // Multiprobe evaluation
    std::cout << "\n=== Multiprobe (ef=100) ===\n";
    std::cout << std::setw(8) << "probes"
              << std::setw(12) << "Recall"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Mean(us)\n";
    std::cout << std::string(44, '-') << "\n";

    std::vector<size_t> probe_counts = {1, 2, 4, 8, 16};

    for (size_t num_probes : probe_counts) {
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            Timer t;
            t.start();
            auto results = index.search_multiprobe(query, k, 100, num_probes);
            latencies[q] = t.elapsed_us();
            recalls[q] = compute_recall(results, dataset.ground_truth[q], k);
        }

        double total_recall = 0.0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / static_cast<double>(dataset.num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << num_probes
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }

    // Multiprobe with higher ef
    std::cout << "\n=== Multiprobe (ef=200) ===\n";
    std::cout << std::setw(8) << "probes"
              << std::setw(12) << "Recall"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Mean(us)\n";
    std::cout << std::string(44, '-') << "\n";

    for (size_t num_probes : probe_counts) {
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            Timer t;
            t.start();
            auto results = index.search_multiprobe(query, k, 200, num_probes);
            latencies[q] = t.elapsed_us();
            recalls[q] = compute_recall(results, dataset.ground_truth[q], k);
        }

        double total_recall = 0.0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / static_cast<double>(dataset.num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << num_probes
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }

    std::cout << "\n";

    // =========================================================================
    // RE-RANKING EVALUATION: The Production Solution
    // =========================================================================
    // CP-HNSW provides fast candidate generation, but CP-LSH has limited precision
    // on clustered data. The solution: over-fetch with CP, then re-rank with true cosine.
    //
    // This is the recommended production pattern:
    //   1. Use CP-HNSW to get top-N candidates (N > k, e.g., N = 100 for k = 10)
    //   2. Compute true cosine similarity for all N candidates
    //   3. Re-sort by true similarity
    //   4. Return top-k
    // =========================================================================

    std::cout << "=== Re-Ranking Evaluation (Production Mode) ===\n";
    std::cout << "  Strategy: Over-fetch with CP, re-rank with true cosine\n";
    std::cout << "  This overcomes the CP-LSH precision ceiling on clustered data.\n\n";

    std::cout << std::setw(10) << "ef"
              << std::setw(10) << "rerank_k"
              << std::setw(12) << "Recall"
              << std::setw(12) << "QPS"
              << std::setw(12) << "Mean(us)\n";
    std::cout << std::string(56, '-') << "\n";

    // Test various over-fetch amounts
    // rerank_k = how many candidates we fetch and re-rank
    std::vector<std::pair<size_t, size_t>> rerank_configs = {
        {100, 50},    // ef=100, fetch 50 candidates
        {100, 100},   // ef=100, fetch 100 candidates
        {200, 100},   // ef=200, fetch 100 candidates
        {200, 200},   // ef=200, fetch 200 candidates
        {500, 200},   // ef=500, fetch 200 candidates
        {500, 500},   // ef=500, fetch 500 candidates
        {1000, 500},  // ef=1000, fetch 500 candidates
    };

    for (auto [ef, rerank_k] : rerank_configs) {
        std::vector<double> latencies(dataset.num_queries);
        std::vector<double> recalls(dataset.num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 100)
#endif
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            Timer t;
            t.start();

            // Step 1: Get over-fetched candidates from CP-HNSW
            auto cp_results = index.search(query, rerank_k, ef);

            // Step 2: Re-rank using true cosine similarity (dot product)
            for (auto& res : cp_results) {
                // Compute true dot product (cosine similarity for normalized vectors)
                const Float* base_vec = dataset.get_base(res.id);
                Float true_sim = 0;
                for (size_t d = 0; d < dataset.dim; ++d) {
                    true_sim += query[d] * base_vec[d];
                }
                // Store negative dot product as distance (higher sim = lower dist)
                res.distance = -true_sim;
            }

            // Step 3: Re-sort by true distance
            std::sort(cp_results.begin(), cp_results.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.distance < b.distance;
                      });

            // Step 4: Keep only top-k
            if (cp_results.size() > k) {
                cp_results.resize(k);
            }

            latencies[q] = t.elapsed_us();
            recalls[q] = compute_recall(cp_results, dataset.ground_truth[q], k);
        }

        double total_recall = 0.0;
        for (size_t q = 0; q < dataset.num_queries; ++q) {
            total_recall += recalls[q];
        }
        double avg_recall = total_recall / static_cast<double>(dataset.num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(10) << ef
                  << std::setw(10) << rerank_k
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }

    std::cout << "\n  INTERPRETATION:\n";
    std::cout << "    - Re-ranking should significantly improve recall\n";
    std::cout << "    - Production recommendation: ef=200, rerank_k=100-200\n";
    std::cout << "    - Higher rerank_k improves recall but adds O(rerank_k * dim) overhead\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "CP-HNSW SIFT-1M Evaluation\n";
    std::cout << "==========================\n\n";

    print_system_info();

    // Dataset path
    std::string data_dir = "../evaluation/data/sift";
    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "Loading SIFT-1M from: " << data_dir << "\n\n";

    // Load base and query vectors (DO NOT load ground truth)
    size_t base_dim, base_count;
    std::vector<Float> base_vectors = load_fvecs(data_dir + "/sift_base.fvecs", base_dim, base_count);
    std::cout << "Loaded base vectors: " << base_count << " x " << base_dim << "\n";

    size_t query_dim, query_count;
    std::vector<Float> query_vectors = load_fvecs(data_dir + "/sift_query.fvecs", query_dim, query_count);
    std::cout << "Loaded query vectors: " << query_count << " x " << query_dim << "\n\n";

    if (base_dim != query_dim) {
        std::cerr << "ERROR: Dimension mismatch between base and query vectors!\n";
        return 1;
    }

    // Normalize vectors to unit length
    std::cout << "Normalizing vectors to unit length...\n";
    Timer timer;
    timer.start();
    normalize_vectors(base_vectors, base_dim);
    normalize_vectors(query_vectors, query_dim);
    std::cout << "  Normalization complete in " << std::fixed << std::setprecision(1)
              << timer.elapsed_ms() << " ms\n\n";

    // Verify normalization
    Float sample_norm = 0;
    for (size_t j = 0; j < base_dim; ++j) {
        sample_norm += base_vectors[j] * base_vectors[j];
    }
    std::cout << "  Sample vector norm: " << std::fixed << std::setprecision(6)
              << std::sqrt(sample_norm) << " (should be ~1.0)\n\n";

    // Run correlation test BEFORE building index
    check_correlation(query_vectors, base_vectors, base_dim, 10000);

    // Recompute ground truth using cosine similarity
    size_t k = 10;
    auto ground_truth = compute_cosine_ground_truth(base_vectors, query_vectors, base_dim, k);

    // Build dataset struct
    Dataset dataset;
    dataset.base_vectors = std::move(base_vectors);
    dataset.query_vectors = std::move(query_vectors);
    dataset.ground_truth = std::move(ground_truth);
    dataset.dim = base_dim;
    dataset.num_base = base_count;
    dataset.num_queries = query_count;
    dataset.k_gt = k;

    // Run evaluation with K=32 and asymmetric distance
    // HIGH PRECISION CONFIG: M=32, ef_c=200 for robust graph topology
    std::cout << ">>> CONFIGURATION: M=32, ef_c=200, K=32 (High Precision) <<<\n";
    run_sift_evaluation(dataset, 32, 200, k, 32);

    std::cout << "==========================\n";
    std::cout << "SIFT-1M Evaluation Complete\n";
    std::cout << "\nKey Changes in This Version:\n";
    std::cout << "  - Asymmetric distance: Uses query magnitudes for continuous gradient\n";
    std::cout << "  - K=32: Doubled code width for better resolution (32 bytes per code)\n";
    std::cout << "  - Brute force diagnostic: Validates hash quality independently of graph\n";
    std::cout << "  - Re-ranking step: Production-ready pattern for high recall\n";
    std::cout << "\nSuccess Criteria:\n";
    std::cout << "  - CP-HNSW navigation works (Recall@10 > 0.10 at ef=100)\n";
    std::cout << "  - Re-ranking achieves high recall (> 0.70 with rerank_k=200)\n";
    std::cout << "  - System demonstrates: fast candidate generation + accurate re-ranking\n";

    return 0;
}
