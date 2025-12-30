#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

using namespace cphnsw;
using namespace cphnsw::eval;

void print_system_info() {
    std::cout << "=== System Information ===\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "Compilation: ";
#ifdef __AVX512F__
    std::cout << "AVX-512 ";
#endif
#ifdef __AVX2__
    std::cout << "AVX2 ";
#endif
    std::cout << "\n\n";
}

void run_evaluation(size_t n, size_t dim, size_t num_queries, size_t k,
                    size_t M, size_t ef_construction, size_t K_rotations = 16) {
    std::cout << "=== Evaluation: N=" << n << ", dim=" << dim << ", M=" << M
              << ", ef_c=" << ef_construction << " ===\n\n";

    // Generate dataset
    std::cout << "Generating random dataset on unit sphere...\n";
    Timer timer;
    timer.start();
    Dataset dataset = generate_random_sphere(n, dim, num_queries, k);
    double gen_time = timer.elapsed_ms();
    std::cout << "  Dataset generation: " << std::fixed << std::setprecision(2)
              << gen_time << " ms\n";
    std::cout << "  Memory: ~" << (n * dim * sizeof(float)) / (1024*1024) << " MB for vectors\n\n";

    // Build index
    std::cout << "Building CP-HNSW index...\n";
    timer.start();

    CPHNSWIndex8 index(dim, M, ef_construction);

    // Progress reporting for large builds
    size_t batch_size = std::min(n, static_cast<size_t>(10000));
    size_t num_batches = (n + batch_size - 1) / batch_size;

    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start = batch * batch_size;
        size_t end = std::min(start + batch_size, n);
        size_t count = end - start;

        for (size_t i = start; i < end; ++i) {
            index.add(dataset.get_base(i));
        }

        if (num_batches > 1 && (batch + 1) % 10 == 0) {
            std::cout << "  Progress: " << (batch + 1) * 100 / num_batches << "%\n";
        }
    }

    double build_time = timer.elapsed_s();
    std::cout << "  Build time: " << std::fixed << std::setprecision(2) << build_time << " s\n";
    std::cout << "  Build rate: " << std::fixed << std::setprecision(0)
              << (n / build_time) << " vectors/s\n";

    // Estimate memory usage
    size_t code_bytes = n * K_rotations;  // K bytes per code
    size_t graph_bytes = n * M * 2 * sizeof(uint32_t);  // Approximate graph overhead
    size_t total_bytes = code_bytes + graph_bytes;
    std::cout << "  Index memory: ~" << total_bytes / (1024*1024) << " MB\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(1)
              << (n * dim * sizeof(float)) / static_cast<double>(total_bytes) << "x\n";

    // Connectivity check
    timer.start();
    size_t connected = index.verify_connectivity();
    double conn_time = timer.elapsed_ms();
    std::cout << "  Connectivity: " << connected << "/" << n
              << (connected == n ? " (fully connected)" : " (DISCONNECTED!)")
              << " [" << conn_time << " ms]\n\n";

    // Evaluate recall vs ef
    std::cout << "Recall@" << k << " vs ef:\n";
    std::cout << std::setw(8) << "ef"
              << std::setw(14) << "Recall@" << k
              << std::setw(12) << "QPS"
              << std::setw(14) << "Mean(us)"
              << std::setw(12) << "P50(us)"
              << std::setw(12) << "P99(us)\n";
    std::cout << std::string(72, '-') << "\n";

    std::vector<size_t> ef_values = {10, 20, 50, 100, 200, 500};
    if (n >= 100000) {
        ef_values.push_back(1000);
    }

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
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(14) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_p50_us
                  << std::setw(12) << std::fixed << std::setprecision(1) << qps_stats.latency_p99_us
                  << "\n";
    }

    // Multiprobe evaluation
    std::cout << "\nMultiprobe (ef=100):\n";
    std::cout << std::setw(8) << "probes"
              << std::setw(14) << "Recall@" << k
              << std::setw(12) << "QPS"
              << std::setw(14) << "Mean(us)\n";
    std::cout << std::string(48, '-') << "\n";

    std::vector<size_t> probe_counts = {1, 2, 4, 8, 16};

    for (size_t num_probes : probe_counts) {
        std::vector<double> latencies;
        latencies.reserve(num_queries);
        double total_recall = 0.0;

        for (size_t q = 0; q < num_queries; ++q) {
            const Float* query = dataset.get_query(q);

            timer.start();
            auto results = index.search_multiprobe(query, k, 100, num_probes);
            latencies.push_back(timer.elapsed_us());

            total_recall += compute_recall(results, dataset.ground_truth[q], k);
        }

        double avg_recall = total_recall / static_cast<double>(num_queries);
        auto qps_stats = compute_qps_stats(latencies);

        std::cout << std::setw(8) << num_probes
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(12) << std::fixed << std::setprecision(0) << qps_stats.qps
                  << std::setw(14) << std::fixed << std::setprecision(1) << qps_stats.latency_mean_us
                  << "\n";
    }

    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "CP-HNSW Comprehensive Evaluation\n";
    std::cout << "=================================\n\n";

    print_system_info();

    size_t k = 10;
    size_t num_queries = 100;

    // Small scale test (warmup)
    std::cout << ">>> SMALL SCALE TEST (10K vectors) <<<\n";
    run_evaluation(10000, 128, num_queries, k, 16, 100);

    // Medium scale test
    std::cout << ">>> MEDIUM SCALE TEST (100K vectors) <<<\n";
    run_evaluation(100000, 128, num_queries, k, 16, 100);

    // Large scale test
    std::cout << ">>> LARGE SCALE TEST (500K vectors) <<<\n";
    run_evaluation(500000, 128, num_queries, k, 16, 200);

    // Parameter sensitivity: varying M
    std::cout << ">>> PARAMETER SENSITIVITY: M <<<\n";
    std::cout << "Testing with N=50000, dim=128\n\n";
    for (size_t M : {8, 16, 32, 48}) {
        run_evaluation(50000, 128, num_queries, k, M, 100);
    }

    // High-dimensional test
    std::cout << ">>> HIGH-DIMENSIONAL TEST (dim=512) <<<\n";
    run_evaluation(50000, 512, num_queries, k, 16, 100);

    std::cout << "=================================\n";
    std::cout << "Comprehensive evaluation complete.\n";

    return 0;
}
