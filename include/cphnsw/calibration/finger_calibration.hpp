#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../distance/hamming.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace cphnsw {

// ============================================================================
// FINGER Calibration Parameters
// ============================================================================

/**
 * FINGERCalibration: Linear correction from asymmetric to true distance.
 *
 * The FINGER paper observes that quantized distances (Hamming, asymmetric dot)
 * have a linear relationship with true distances, but may have bias and scale
 * differences. Calibration fits:
 *
 *     true_distance ≈ alpha * asymmetric_distance + beta
 *
 * This creates a continuous gradient for better search navigation.
 *
 * Usage:
 *     auto calib = FINGERCalibration::calibrate(graph, vectors, dim, num_samples);
 *     float calibrated = calib.apply(asymmetric_dist);
 */
struct FINGERCalibration {
    float alpha = 1.0f;   // Scale factor
    float beta = 0.0f;    // Offset (bias correction)
    float r_squared = 0.0f;  // Goodness of fit (for diagnostics)
    size_t num_samples = 0;  // Number of samples used

    /**
     * Apply calibration to an asymmetric distance.
     *
     * @param asymmetric_dist  Raw asymmetric search distance
     * @return                 Calibrated distance (better approximates true distance)
     */
    float apply(float asymmetric_dist) const {
        return alpha * asymmetric_dist + beta;
    }

    /**
     * Check if calibration is valid (has been computed).
     */
    bool is_valid() const {
        return num_samples > 0 && std::isfinite(alpha) && std::isfinite(beta);
    }
};

// ============================================================================
// FINGER Calibration Algorithm
// ============================================================================

/**
 * FINGERCalibrator: Computes linear calibration from graph edges.
 *
 * Algorithm:
 * 1. Sample random edges (u, v) from the graph
 * 2. For each edge, compute:
 *    - X: asymmetric_search_distance(query_for_u, code_of_v)
 *    - Y: true_cosine_distance(vector_u, vector_v)
 * 3. Fit linear regression: Y = alpha * X + beta
 *
 * The fitted coefficients correct for quantization bias and scale.
 */
template <typename ComponentT, size_t K>
class FINGERCalibrator {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;

    /**
     * Calibrate using graph edges and original vectors.
     *
     * @param graph       The NSW graph with Flash layout
     * @param vectors     Original vectors (row-major, N x dim)
     * @param dim         Vector dimension
     * @param encoder     Encoder for creating query structs
     * @param num_samples Number of edge pairs to sample (default 1000)
     * @param seed        Random seed for reproducibility
     * @return            Calibration parameters
     */
    template <typename Encoder>
    static FINGERCalibration calibrate(
        const Graph& graph,
        const Float* vectors,
        size_t dim,
        const Encoder& encoder,
        size_t num_samples = 1000,
        uint64_t seed = 42) {

        if (graph.size() < 2) {
            return FINGERCalibration{};  // Not enough nodes
        }

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<NodeId> node_dist(0, static_cast<NodeId>(graph.size() - 1));

        std::vector<float> X;  // Asymmetric distances
        std::vector<float> Y;  // True cosine distances
        X.reserve(num_samples);
        Y.reserve(num_samples);

        size_t attempts = 0;
        const size_t max_attempts = num_samples * 10;

        while (X.size() < num_samples && attempts < max_attempts) {
            ++attempts;

            // Sample a random node
            NodeId u = node_dist(rng);

            // Get its neighbors
            auto [neighbors, neighbor_count] = graph.get_neighbors(u);
            if (neighbor_count == 0) continue;

            // Sample a random neighbor
            std::uniform_int_distribution<size_t> neighbor_dist(0, neighbor_count - 1);
            NodeId v = neighbors[neighbor_dist(rng)];
            if (v == INVALID_NODE) continue;

            // Get vectors
            const Float* vec_u = vectors + u * dim;
            const Float* vec_v = vectors + v * dim;

            // Compute true cosine distance
            // For normalized vectors: cos_sim = dot(u, v), cos_dist = 1 - cos_sim
            // But our asymmetric distance is -dot, so true_dist = -dot for consistency
            float true_dot = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                true_dot += vec_u[d] * vec_v[d];
            }
            float true_dist = -true_dot;  // Negative for distance (lower = more similar)

            // Compute asymmetric search distance
            // Create query from u's vector, compare to v's code
            Query query_u = encoder.encode_query(vec_u);
            const Code& code_v = graph.get_code(v);
            float asymm_dist = asymmetric_search_distance(query_u, code_v);

            X.push_back(asymm_dist);
            Y.push_back(true_dist);
        }

        if (X.size() < 10) {
            return FINGERCalibration{};  // Not enough samples
        }

        // Fit linear regression: Y = alpha * X + beta
        // Using ordinary least squares
        return fit_linear_regression(X, Y);
    }

    /**
     * Calibrate using sampled vector pairs (no graph required).
     *
     * Alternative calibration when graph is not available (e.g., before build).
     * Samples random vector pairs from the dataset.
     *
     * @param vectors     Original vectors (row-major, N x dim)
     * @param N           Number of vectors
     * @param dim         Vector dimension
     * @param encoder     Encoder for creating codes and queries
     * @param num_samples Number of pairs to sample
     * @param seed        Random seed
     * @return            Calibration parameters
     */
    template <typename Encoder>
    static FINGERCalibration calibrate_from_vectors(
        const Float* vectors,
        size_t N,
        size_t dim,
        const Encoder& encoder,
        size_t num_samples = 1000,
        uint64_t seed = 42) {

        if (N < 2) {
            return FINGERCalibration{};
        }

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> vec_dist(0, N - 1);

        std::vector<float> X;
        std::vector<float> Y;
        X.reserve(num_samples);
        Y.reserve(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            // Sample two different vectors
            size_t u = vec_dist(rng);
            size_t v = vec_dist(rng);
            if (u == v) {
                v = (v + 1) % N;
            }

            const Float* vec_u = vectors + u * dim;
            const Float* vec_v = vectors + v * dim;

            // True cosine distance
            float true_dot = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                true_dot += vec_u[d] * vec_v[d];
            }
            float true_dist = -true_dot;

            // Asymmetric distance
            Query query_u = encoder.encode_query(vec_u);
            Code code_v = encoder.encode(vec_v);
            float asymm_dist = asymmetric_search_distance(query_u, code_v);

            X.push_back(asymm_dist);
            Y.push_back(true_dist);
        }

        return fit_linear_regression(X, Y);
    }

private:
    /**
     * Fit linear regression using ordinary least squares.
     *
     * Fits Y = alpha * X + beta minimizing sum of squared residuals.
     *
     * @param X  Independent variable (asymmetric distances)
     * @param Y  Dependent variable (true distances)
     * @return   Calibration parameters with alpha, beta, and R²
     */
    static FINGERCalibration fit_linear_regression(
        const std::vector<float>& X,
        const std::vector<float>& Y) {

        size_t n = X.size();
        if (n < 2) {
            return FINGERCalibration{};
        }

        // Compute means
        float mean_x = std::accumulate(X.begin(), X.end(), 0.0f) / n;
        float mean_y = std::accumulate(Y.begin(), Y.end(), 0.0f) / n;

        // Compute covariance and variance
        float cov_xy = 0.0f;
        float var_x = 0.0f;
        float var_y = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            float dx = X[i] - mean_x;
            float dy = Y[i] - mean_y;
            cov_xy += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        // Avoid division by zero
        if (var_x < 1e-10f) {
            return FINGERCalibration{1.0f, 0.0f, 0.0f, n};
        }

        // Compute regression coefficients
        float alpha = cov_xy / var_x;
        float beta = mean_y - alpha * mean_x;

        // Compute R² (coefficient of determination)
        float ss_res = 0.0f;  // Residual sum of squares
        float ss_tot = var_y; // Total sum of squares

        for (size_t i = 0; i < n; ++i) {
            float y_pred = alpha * X[i] + beta;
            float residual = Y[i] - y_pred;
            ss_res += residual * residual;
        }

        float r_squared = (ss_tot > 1e-10f) ? (1.0f - ss_res / ss_tot) : 0.0f;

        return FINGERCalibration{alpha, beta, r_squared, n};
    }
};

}  // namespace cphnsw
