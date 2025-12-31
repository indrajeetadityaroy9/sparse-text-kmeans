#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include <iostream>
#include <iomanip>

using namespace cphnsw;
using namespace cphnsw::eval;

int main(int argc, char** argv) {
    std::cout << "CP-HNSW Evaluation\n";
    std::cout << "==================\n\n";

    // Configuration
    size_t n = 10000;          // Number of base vectors
    size_t dim = 128;          // Dimension (SIFT-like)
    size_t num_queries = 100;  // Number of queries
    size_t k = 10;             // k for recall@k
    size_t M = 16;             // HNSW M parameter
    size_t ef_construction = 100;

    std::cout << "Configuration:\n";
    std::cout << "  N = " << n << "\n";
    std::cout << "  dim = " << dim << "\n";
    std::cout << "  queries = " << num_queries << "\n";
    std::cout << "  k = " << k << "\n";
    std::cout << "  M = " << M << "\n";
    std::cout << "  ef_construction = " << ef_construction << "\n\n";

    // Generate random dataset
    std::cout << "Generating random dataset on unit sphere...\n";
    Timer timer;
    timer.start();
    Dataset dataset = generate_random_sphere(n, dim, num_queries, k);
    std::cout << "  Done in " << std::fixed << std::setprecision(2)
              << timer.elapsed_ms() << " ms\n\n";

    // Build index
    std::cout << "Building CP-HNSW index...\n";
    timer.start();

    CPHNSWIndex8 index(dim, M, ef_construction);
    index.add_batch(dataset.base_vectors.data(), dataset.num_base);

    double build_time = timer.elapsed_s();
    std::cout << "  Done in " << std::fixed << std::setprecision(2)
              << build_time << " s\n";
    std::cout << "  Build rate: " << std::fixed << std::setprecision(0)
              << (n / build_time) << " vectors/s\n";

    // Check connectivity
    size_t connected = index.verify_connectivity();
    std::cout << "  Connectivity: " << connected << "/" << n
              << (connected == n ? " (fully connected)" : " (DISCONNECTED!)") << "\n\n";

    // Evaluate recall vs ef
    std::cout << "Evaluating Recall@" << k << " vs ef:\n";
    std::cout << std::setw(8) << "ef" << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS" << std::setw(14) << "Latency(us)\n";
    std::cout << std::string(46, '-') << "\n";

    std::vector<size_t> ef_values = {10, 20, 50, 100, 200};

    for (size_t ef : ef_values) {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            timer.start();
            auto results = index.search(query, k, ef);
            latencies.push_back(timer.elapsed_us());

            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << ef
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(14) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }

    std::cout << "\n";

    // Test multiprobe
    std::cout << "Evaluating Multiprobe (ef=50, varying probes):\n";
    std::cout << std::setw(8) << "probes" << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS\n";
    std::cout << std::string(32, '-') << "\n";

    std::vector<size_t> probe_counts = {1, 2, 4, 8};

    for (size_t num_probes : probe_counts) {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            timer.start();
            auto results = index.search_multiprobe(query, k, 50, num_probes);
            latencies.push_back(timer.elapsed_us());

            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << num_probes
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << "\n";
    }

    // Test FINGER calibration
    std::cout << "\nTesting FINGER Calibration:\n";
    std::cout << std::string(40, '-') << "\n";

    timer.start();
    auto calib = index.calibrate(1000, 42);
    double calib_time = timer.elapsed_ms();

    std::cout << "  Calibration completed in " << std::fixed << std::setprecision(2)
              << calib_time << " ms\n";
    std::cout << "  Alpha: " << std::fixed << std::setprecision(4) << calib.alpha << "\n";
    std::cout << "  Beta: " << std::fixed << std::setprecision(4) << calib.beta << "\n";
    std::cout << "  R²: " << std::fixed << std::setprecision(4) << calib.r_squared << "\n";
    std::cout << "  Samples: " << calib.num_samples << "\n\n";

    // Compare calibrated vs uncalibrated search
    std::cout << "Calibrated vs Uncalibrated (ef=100):\n";
    std::cout << std::setw(16) << "Method" << std::setw(12) << "Recall@" << k
              << std::setw(12) << "QPS\n";
    std::cout << std::string(40, '-') << "\n";

    // Uncalibrated
    {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);
            timer.start();
            auto results = index.search(query, k, 100);
            latencies.push_back(timer.elapsed_us());
            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);
        std::cout << std::setw(16) << "Uncalibrated"
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << "\n";
    }

    // Calibrated
    {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);
            timer.start();
            auto results = index.search_calibrated(query, k, 100);
            latencies.push_back(timer.elapsed_us());
            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);
        std::cout << std::setw(16) << "Calibrated"
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << "\n";
    }

    // Calibrated + Rerank
    {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);
            timer.start();
            auto results = index.search_calibrated_and_rerank(query, k, 100, 50);
            latencies.push_back(timer.elapsed_us());
            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);
        std::cout << std::setw(16) << "Calib+Rerank"
                  << std::setw(12) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << "\n";
    }

    std::cout << "\nEvaluation complete.\n";

    return 0;
}
