#pragma once

#include "../../include/cphnsw/core/types.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <random>
#include <cmath>

namespace cphnsw::eval {

/**
 * Dataset container.
 */
struct Dataset {
    std::vector<Float> base_vectors;          // N x dim (row-major)
    std::vector<Float> query_vectors;         // Q x dim (row-major)
    std::vector<std::vector<NodeId>> ground_truth;  // Q x k_gt
    size_t dim;
    size_t num_base;
    size_t num_queries;
    size_t k_gt;  // Ground truth k

    /// Get pointer to base vector i
    const Float* get_base(size_t i) const {
        return base_vectors.data() + i * dim;
    }

    /// Get pointer to query vector i
    const Float* get_query(size_t i) const {
        return query_vectors.data() + i * dim;
    }
};

/**
 * Load fvecs format (SIFT, GIST).
 * Format: [dim(4 bytes)][float * dim] repeated
 */
inline std::vector<Float> load_fvecs(const std::string& path, size_t& dim, size_t& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read first dimension
    int32_t d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    dim = static_cast<size_t>(d);

    // Get file size to compute count
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    size_t record_size = sizeof(int32_t) + dim * sizeof(float);
    count = file_size / record_size;

    // Allocate and read
    std::vector<Float> data(count * dim);
    file.seekg(0, std::ios::beg);

    for (size_t i = 0; i < count; ++i) {
        int32_t d_check;
        file.read(reinterpret_cast<char*>(&d_check), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }

    return data;
}

/**
 * Load ivecs format (ground truth).
 * Format: [k(4 bytes)][int32 * k] repeated
 */
inline std::vector<std::vector<NodeId>> load_ivecs(const std::string& path, size_t& k, size_t& count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read first k
    int32_t k_val;
    file.read(reinterpret_cast<char*>(&k_val), sizeof(int32_t));
    k = static_cast<size_t>(k_val);

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    size_t record_size = sizeof(int32_t) + k * sizeof(int32_t);
    count = file_size / record_size;

    // Read
    std::vector<std::vector<NodeId>> data(count);
    std::vector<int32_t> buffer(k);
    file.seekg(0, std::ios::beg);

    for (size_t i = 0; i < count; ++i) {
        int32_t k_check;
        file.read(reinterpret_cast<char*>(&k_check), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(buffer.data()), k * sizeof(int32_t));

        data[i].resize(k);
        for (size_t j = 0; j < k; ++j) {
            data[i][j] = static_cast<NodeId>(buffer[j]);
        }
    }

    return data;
}

/**
 * Load SIFT-1M dataset.
 *
 * Expected files in directory:
 * - sift_base.fvecs (1M x 128)
 * - sift_query.fvecs (10K x 128)
 * - sift_groundtruth.ivecs (10K x 100)
 */
inline Dataset load_sift1m(const std::string& dir) {
    Dataset ds;

    size_t base_dim, base_count;
    ds.base_vectors = load_fvecs(dir + "/sift_base.fvecs", base_dim, base_count);
    ds.dim = base_dim;
    ds.num_base = base_count;

    size_t query_dim, query_count;
    ds.query_vectors = load_fvecs(dir + "/sift_query.fvecs", query_dim, query_count);
    ds.num_queries = query_count;

    if (query_dim != ds.dim) {
        throw std::runtime_error("Query dimension mismatch");
    }

    size_t k_gt, gt_count;
    ds.ground_truth = load_ivecs(dir + "/sift_groundtruth.ivecs", k_gt, gt_count);
    ds.k_gt = k_gt;

    return ds;
}

/**
 * Generate random dataset on unit sphere.
 * Useful for testing and benchmarking.
 */
inline Dataset generate_random_sphere(
    size_t n, size_t dim, size_t num_queries, size_t k_gt = 100, uint64_t seed = 42) {

    Dataset ds;
    ds.dim = dim;
    ds.num_base = n;
    ds.num_queries = num_queries;
    ds.k_gt = k_gt;

    std::mt19937_64 rng(seed);
    std::normal_distribution<Float> normal(0.0f, 1.0f);

    // Generate base vectors (normalized)
    ds.base_vectors.resize(n * dim);
    for (size_t i = 0; i < n; ++i) {
        Float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            Float val = normal(rng);
            ds.base_vectors[i * dim + j] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dim; ++j) {
            ds.base_vectors[i * dim + j] /= norm;
        }
    }

    // Generate query vectors (normalized)
    ds.query_vectors.resize(num_queries * dim);
    for (size_t i = 0; i < num_queries; ++i) {
        Float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            Float val = normal(rng);
            ds.query_vectors[i * dim + j] = val;
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dim; ++j) {
            ds.query_vectors[i * dim + j] /= norm;
        }
    }

    // Compute ground truth (brute force)
    ds.ground_truth.resize(num_queries);

    for (size_t q = 0; q < num_queries; ++q) {
        const Float* query = ds.get_query(q);

        // Compute all distances (cosine = 1 - dot product for unit vectors)
        std::vector<std::pair<Float, NodeId>> distances(n);
        for (size_t i = 0; i < n; ++i) {
            const Float* base = ds.get_base(i);
            Float dot = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                dot += query[j] * base[j];
            }
            // Higher dot = closer, so negate for sorting
            distances[i] = {-dot, static_cast<NodeId>(i)};
        }

        // Partial sort to get top k_gt
        std::partial_sort(distances.begin(), distances.begin() + k_gt, distances.end());

        ds.ground_truth[q].resize(k_gt);
        for (size_t i = 0; i < k_gt; ++i) {
            ds.ground_truth[q][i] = distances[i].second;
        }
    }

    return ds;
}

/**
 * Normalize vectors to unit length (L2 normalization).
 * CRITICAL for angular LSH - all vectors must be on unit sphere.
 *
 * @param vecs   Pointer to vectors (row-major, count x dim)
 * @param count  Number of vectors
 * @param dim    Vector dimension
 */
inline void normalize_vectors(Float* vecs, size_t count, size_t dim) {
    for (size_t i = 0; i < count; ++i) {
        Float* v = vecs + i * dim;
        Float norm_sq = 0;
        for (size_t d = 0; d < dim; ++d) {
            norm_sq += v[d] * v[d];
        }
        Float norm = std::sqrt(norm_sq);
        if (norm > 1e-10f) {
            Float inv_norm = 1.0f / norm;
            for (size_t d = 0; d < dim; ++d) {
                v[d] *= inv_norm;
            }
        }
    }
}

/**
 * Load GIST-1M dataset.
 *
 * CRITICAL: GIST vectors are NOT pre-normalized.
 * This function normalizes all vectors to unit length after loading.
 *
 * Expected files in directory:
 * - gist_base.fvecs (1M x 960)
 * - gist_query.fvecs (1K x 960)
 * - gist_groundtruth.ivecs (1K x 100)
 */
inline Dataset load_gist1m(const std::string& dir) {
    Dataset ds;

    size_t base_dim, base_count;
    ds.base_vectors = load_fvecs(dir + "/gist_base.fvecs", base_dim, base_count);
    ds.dim = base_dim;  // 960
    ds.num_base = base_count;  // 1M

    // CRITICAL: Normalize to unit sphere for angular LSH
    normalize_vectors(ds.base_vectors.data(), ds.num_base, ds.dim);

    size_t query_dim, query_count;
    ds.query_vectors = load_fvecs(dir + "/gist_query.fvecs", query_dim, query_count);
    ds.num_queries = query_count;

    if (query_dim != ds.dim) {
        throw std::runtime_error("Query dimension mismatch");
    }

    // CRITICAL: Also normalize queries
    normalize_vectors(ds.query_vectors.data(), ds.num_queries, ds.dim);

    size_t k_gt, gt_count;
    ds.ground_truth = load_ivecs(dir + "/gist_groundtruth.ivecs", k_gt, gt_count);
    ds.k_gt = k_gt;

    return ds;
}

/**
 * Load SIFT-1M with optional normalization.
 *
 * SIFT vectors may not be pre-normalized. For CP-LSH angular distance,
 * vectors should be normalized to unit length.
 *
 * @param dir        Directory containing SIFT files
 * @param normalize  If true, normalize vectors to unit length
 */
inline Dataset load_sift1m_normalized(const std::string& dir, bool normalize = true) {
    Dataset ds = load_sift1m(dir);

    if (normalize) {
        normalize_vectors(ds.base_vectors.data(), ds.num_base, ds.dim);
        normalize_vectors(ds.query_vectors.data(), ds.num_queries, ds.dim);
    }

    return ds;
}

}  // namespace cphnsw::eval
