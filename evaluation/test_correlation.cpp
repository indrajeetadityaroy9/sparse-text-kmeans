#include "../include/cphnsw/quantizer/cp_encoder.hpp"
#include "../include/cphnsw/distance/hamming.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <set>

using namespace cphnsw;

// Generate random unit vector
void generate_unit_vector(std::vector<Float>& v, std::mt19937& rng) {
    std::normal_distribution<Float> normal(0.0f, 1.0f);
    Float norm = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = normal(rng);
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] /= norm;
    }
}

// True dot product
Float true_dot_product(const std::vector<Float>& a, const std::vector<Float>& b) {
    Float dot = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

// Pearson correlation coefficient
Float pearson_correlation(const std::vector<Float>& x, const std::vector<Float>& y) {
    size_t n = x.size();
    Float mean_x = 0, mean_y = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    Float cov = 0, var_x = 0, var_y = 0;
    for (size_t i = 0; i < n; ++i) {
        Float dx = x[i] - mean_x;
        Float dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    return cov / std::sqrt(var_x * var_y);
}

int main() {
    std::cout << "CP Estimator Correlation Diagnostic\n";
    std::cout << "====================================\n\n";

    const size_t dim = 128;
    const size_t num_queries = 100;
    const size_t num_base = 1000;
    const size_t num_pairs = num_queries * num_base;

    std::mt19937 rng(42);

    // Create encoder
    CPEncoder<uint8_t, 16> encoder(dim, 42);

    // Generate vectors
    std::vector<std::vector<Float>> queries(num_queries, std::vector<Float>(dim));
    std::vector<std::vector<Float>> base(num_base, std::vector<Float>(dim));

    for (auto& q : queries) generate_unit_vector(q, rng);
    for (auto& b : base) generate_unit_vector(b, rng);

    // Encode all vectors
    std::vector<CPQuery<uint8_t, 16>> query_codes(num_queries);
    std::vector<CPCode<uint8_t, 16>> base_codes(num_base);

    for (size_t i = 0; i < num_queries; ++i) {
        query_codes[i] = encoder.encode_query(queries[i].data());
    }
    for (size_t i = 0; i < num_base; ++i) {
        base_codes[i] = encoder.encode(base[i].data());
    }

    // Compute true vs estimated similarities
    std::vector<Float> true_sims, est_sims;
    true_sims.reserve(num_pairs);
    est_sims.reserve(num_pairs);

    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t b = 0; b < num_base; ++b) {
            Float true_sim = true_dot_product(queries[q], base[b]);
            Float est_sim = -asymmetric_search_distance(query_codes[q], base_codes[b]);

            true_sims.push_back(true_sim);
            est_sims.push_back(est_sim);
        }
    }

    // Compute correlation
    Float correlation = pearson_correlation(true_sims, est_sims);

    // Statistics
    Float true_min = *std::min_element(true_sims.begin(), true_sims.end());
    Float true_max = *std::max_element(true_sims.begin(), true_sims.end());
    Float est_min = *std::min_element(est_sims.begin(), est_sims.end());
    Float est_max = *std::max_element(est_sims.begin(), est_sims.end());

    std::cout << "Configuration:\n";
    std::cout << "  dim = " << dim << "\n";
    std::cout << "  K = 16 (rotations)\n";
    std::cout << "  num_pairs = " << num_pairs << "\n\n";

    std::cout << "Results:\n";
    std::cout << "  True similarity range: [" << std::fixed << std::setprecision(4)
              << true_min << ", " << true_max << "]\n";
    std::cout << "  Est similarity range:  [" << std::fixed << std::setprecision(4)
              << est_min << ", " << est_max << "]\n";
    std::cout << "  Pearson correlation:   " << std::fixed << std::setprecision(4)
              << correlation << "\n\n";

    // Sample pairs
    std::cout << "Sample pairs (True vs Est):\n";
    for (int i = 0; i < 10; ++i) {
        size_t idx = rng() % num_pairs;
        std::cout << "  " << std::fixed << std::setprecision(4)
                  << true_sims[idx] << " vs " << est_sims[idx] << "\n";
    }
    std::cout << "\n";

    // Interpretation
    if (correlation > 0.7) {
        std::cout << "GOOD: Correlation > 0.7 - Estimator is usable for ranking\n";
    } else if (correlation > 0.3) {
        std::cout << "MODERATE: Correlation 0.3-0.7 - High variance, may need multiprobe\n";
    } else if (correlation > 0.0) {
        std::cout << "POOR: Correlation 0.0-0.3 - Estimator barely correlates with true similarity\n";
    } else {
        std::cout << "BROKEN: Negative correlation - Something is wrong!\n";
    }

    // Test: What fraction of true top-10 are in estimated top-10?
    std::cout << "\n=== Ranking Quality Test ===\n";

    size_t total_recall = 0;
    for (size_t q = 0; q < num_queries; ++q) {
        // Get true top-10
        std::vector<std::pair<Float, size_t>> true_ranked(num_base);
        std::vector<std::pair<Float, size_t>> est_ranked(num_base);

        for (size_t b = 0; b < num_base; ++b) {
            true_ranked[b] = {-true_dot_product(queries[q], base[b]), b};
            est_ranked[b] = {asymmetric_search_distance(query_codes[q], base_codes[b]), b};
        }

        std::partial_sort(true_ranked.begin(), true_ranked.begin() + 10, true_ranked.end());
        std::partial_sort(est_ranked.begin(), est_ranked.begin() + 10, est_ranked.end());

        // Count overlap
        std::set<size_t> true_top10;
        for (int i = 0; i < 10; ++i) {
            true_top10.insert(true_ranked[i].second);
        }

        size_t hits = 0;
        for (int i = 0; i < 10; ++i) {
            if (true_top10.count(est_ranked[i].second)) {
                ++hits;
            }
        }
        total_recall += hits;
    }

    Float avg_recall = static_cast<Float>(total_recall) / (num_queries * 10);
    std::cout << "  Average Recall@10 (brute force CP vs true cosine): "
              << std::fixed << std::setprecision(4) << avg_recall << "\n";

    if (avg_recall < 0.1) {
        std::cout << "\n  DIAGNOSIS: The CP estimator does NOT preserve ranking.\n";
        std::cout << "  The fundamental algorithm is not working as intended.\n";
    }

    return 0;
}
