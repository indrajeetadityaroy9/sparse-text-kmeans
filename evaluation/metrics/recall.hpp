#pragma once

#include "../../include/cphnsw/core/types.hpp"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <numeric>

namespace cphnsw::eval {

/**
 * Compute Recall@k.
 *
 * Recall@k = |retrieved ∩ ground_truth| / k
 *
 * @param retrieved       Retrieved neighbor IDs
 * @param ground_truth    Ground truth neighbor IDs
 * @param k               Number of neighbors to consider
 * @return                Recall score in [0, 1]
 */
inline double compute_recall(
    const std::vector<NodeId>& retrieved,
    const std::vector<NodeId>& ground_truth,
    size_t k) {

    std::unordered_set<NodeId> gt_set(
        ground_truth.begin(),
        ground_truth.begin() + std::min(k, ground_truth.size()));

    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, retrieved.size()); ++i) {
        if (gt_set.count(retrieved[i])) {
            ++hits;
        }
    }

    return static_cast<double>(hits) / static_cast<double>(gt_set.size());
}

/**
 * Compute Recall@k from SearchResults.
 */
inline double compute_recall(
    const std::vector<SearchResult>& results,
    const std::vector<NodeId>& ground_truth,
    size_t k) {

    std::vector<NodeId> retrieved;
    retrieved.reserve(results.size());
    for (const auto& r : results) {
        retrieved.push_back(r.id);
    }

    return compute_recall(retrieved, ground_truth, k);
}

/**
 * QPS measurement result.
 */
struct QPSResult {
    double qps;           // Queries per second
    double latency_mean_us;   // Mean latency in microseconds
    double latency_p50_us;    // Median latency
    double latency_p99_us;    // 99th percentile latency
};

/**
 * Compute QPS statistics from latency measurements.
 *
 * @param latencies_us  Latency measurements in microseconds
 * @return              QPS statistics
 */
inline QPSResult compute_qps_stats(std::vector<double>& latencies_us) {
    QPSResult result;

    if (latencies_us.empty()) {
        result.qps = 0;
        result.latency_mean_us = 0;
        result.latency_p50_us = 0;
        result.latency_p99_us = 0;
        return result;
    }

    // Sort for percentiles
    std::sort(latencies_us.begin(), latencies_us.end());

    // Mean
    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    result.latency_mean_us = sum / static_cast<double>(latencies_us.size());

    // QPS
    result.qps = 1e6 / result.latency_mean_us;

    // Median (p50)
    size_t n = latencies_us.size();
    result.latency_p50_us = latencies_us[n / 2];

    // p99
    size_t p99_idx = static_cast<size_t>(0.99 * n);
    result.latency_p99_us = latencies_us[std::min(p99_idx, n - 1)];

    return result;
}

/**
 * High-resolution timer for benchmarking.
 */
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }

    double elapsed_s() const {
        return elapsed_us() / 1e6;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace cphnsw::eval
