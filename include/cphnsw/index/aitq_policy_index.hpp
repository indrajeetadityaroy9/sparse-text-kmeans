#pragma once

#include "../core/types.hpp"
#include "../core/debug.hpp"
#include "../graph/aitq_graph.hpp"
#include "../algorithms/aitq_search_layer.hpp"
#include "../quantizer/aitq_quantizer.hpp"
#include <memory>
#include <random>
#include <atomic>
#include <queue>
#include <cstring>
#include <numeric>
#include <algorithm>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

namespace cphnsw {

/**
 * AITQPolicyIndex: A-ITQ Optimized ANN Index
 *
 * METRIC ALIGNMENT FIX:
 * This index is specifically designed for A-ITQ (Asymmetric ITQ) quantization.
 * It uses AITQGraph for bit-packed code storage and A-ITQ distance for
 * both construction and search, ensuring metric alignment.
 *
 * Key differences from PolicyIndex:
 * - Uses AITQGraph<K> instead of FlatHNSWGraph (bit-packed storage)
 * - Uses AITQSearchLayer for SIMD batch distance computation
 * - Specialized for AITQQuantizer
 *
 * Usage:
 *   // Train A-ITQ quantizer
 *   auto quantizer = std::make_shared<AITQQuantizer<256>>(dim, train_data, N);
 *
 *   // Create index
 *   CPHNSWParams params;
 *   params.dim = dim;
 *   params.k = 256;
 *   AITQPolicyIndex<256> index(params, quantizer);
 *
 *   // Add vectors
 *   index.add_batch_parallel(vectors, N);
 *
 *   // Search
 *   auto results = index.search(query, k);
 *
 * Template parameters:
 * - K: Number of bits per code (e.g., 128, 256)
 */
template <size_t K>
class AITQPolicyIndex {
public:
    using Quantizer = AITQQuantizer<K>;
    using Code = AITQCode<K>;
    using Query = AITQQuery<K>;
    using Graph = AITQGraph<K>;

    /**
     * Construct index with A-ITQ quantizer.
     *
     * @param params     Index configuration
     * @param quantizer  Trained A-ITQ quantizer
     */
    AITQPolicyIndex(CPHNSWParams params, std::shared_ptr<Quantizer> quantizer)
        : params_(finalize_params(std::move(params))),
          quantizer_(std::move(quantizer)),
          graph_(params_),
          rng_(params_.seed),
          query_counter_(0) {}

    /// Get index parameters
    const CPHNSWParams& params() const { return params_; }

    /// Get quantizer
    const Quantizer& quantizer() const { return *quantizer_; }

    /// Get number of vectors in index
    size_t size() const { return graph_.size(); }

    /// Check if empty
    bool empty() const { return graph_.empty(); }

    /// Get vector dimension
    size_t dim() const { return quantizer_->dim(); }

    /// Get the underlying graph (for debugging/testing)
    const Graph& graph() const { return graph_; }

    /**
     * Add a single vector to the index.
     *
     * Uses A-ITQ DISTANCE for graph navigation during construction.
     */
    NodeId add(const Float* vec) {
        // Store original vector
        original_vectors_.insert(original_vectors_.end(), vec, vec + params_.dim);

        // Encode using A-ITQ
        auto query = quantizer_->encode_query(vec);

        // Add node to graph with A-ITQ code
        NodeId id = graph_.add_node(query.code);

        // Handle first node
        if (graph_.size() == 1) {
            return id;
        }

        // Get entry points
        uint64_t qid = query_counter_.fetch_add(1);
        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        // Insert using A-ITQ DISTANCE for graph navigation
        // but TRUE DISTANCE for edge selection (hybrid approach)
        AITQInsert<K>::insert_hybrid(
            id, query, vec, original_vectors_, params_.dim,
            entry_points, graph_, *quantizer_, params_, qid);

        return id;
    }

    /**
     * Add multiple vectors (sequential).
     */
    void add_batch(const Float* vecs, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            add(vecs + i * params_.dim);
        }
    }

    /**
     * Add multiple vectors (parallel).
     *
     * METRIC-ALIGNED CONSTRUCTION:
     * Uses A-ITQ distance during graph navigation.
     */
    void add_batch_parallel(const Float* vecs, size_t count, size_t batch_size = 0) {
        size_t start_id = graph_.size();
        size_t dim = params_.dim;

        if (batch_size == 0) {
#ifdef CPHNSW_USE_OPENMP
            batch_size = static_cast<size_t>(omp_get_max_threads());
#else
            batch_size = 64;
#endif
        }

        // Phase 1: Pre-allocation
        CPHNSW_DEBUG_PHASE(1, "Pre-allocating for " << count << " vectors (A-ITQ)");
        original_vectors_.resize((start_id + count) * dim);
        graph_.reserve_nodes(start_id + count);

        // Phase 2: Encoding (parallel)
        CPHNSW_DEBUG_PHASE(2, "Encoding " << count << " vectors with A-ITQ");
        std::vector<Query> queries(count);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < count; ++i) {
            const Float* vec = vecs + i * dim;

            // Copy vector
            std::memcpy(original_vectors_.data() + (start_id + i) * dim,
                        vec, dim * sizeof(Float));

            // Encode with A-ITQ
            queries[i] = quantizer_->encode_query(vec);
        }

        // Phase 3: Node creation (sequential)
        CPHNSW_DEBUG_PHASE(3, "Creating " << count << " nodes");
        for (size_t i = 0; i < count; ++i) {
            graph_.add_node(queries[i].code);
        }

        // Phase 4: Bootstrap (sequential)
        size_t bootstrap_count = std::min(static_cast<size_t>(1000), count);
        CPHNSW_DEBUG("Bootstrap " << bootstrap_count << " nodes");

        for (size_t i = 0; i < bootstrap_count; ++i) {
            NodeId id = static_cast<NodeId>(start_id + i);
            const Float* vec = vecs + i * dim;
            uint64_t qid = query_counter_.fetch_add(1);

            std::vector<NodeId> entry_points = graph_.get_random_entry_points(
                params_.k_entry, qid);

            AITQInsert<K>::insert_hybrid(
                id, queries[i], vec, original_vectors_, dim,
                entry_points, graph_, *quantizer_, params_, qid);
        }

        // Phase 5: Shuffled parallel linking
        CPHNSW_DEBUG_PHASE(5, "Shuffled parallel linking");
        size_t remaining = count - bootstrap_count;

        std::vector<size_t> shuffled_indices(remaining);
        std::iota(shuffled_indices.begin(), shuffled_indices.end(), bootstrap_count);

        std::mt19937_64 shuffle_rng(params_.seed + 12345);
        std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), shuffle_rng);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (size_t idx = 0; idx < remaining; ++idx) {
            size_t i = shuffled_indices[idx];
            NodeId id = static_cast<NodeId>(start_id + i);
            const Float* vec = vecs + i * dim;

            uint64_t qid = query_counter_.fetch_add(1);
            std::vector<NodeId> entry_points = graph_.get_random_entry_points(
                params_.k_entry, qid);

            AITQInsert<K>::insert_hybrid(
                id, queries[i], vec, original_vectors_, dim,
                entry_points, graph_, *quantizer_, params_, qid);
        }
    }

    /**
     * Search for k nearest neighbors.
     *
     * Uses A-ITQ DISTANCE for graph navigation.
     */
    std::vector<SearchResult> search(const Float* query, size_t k, size_t ef = 0) const {
        if (ef == 0) {
            ef = std::max(k, static_cast<size_t>(10));
        }

        auto q = quantizer_->encode_query(query);
        uint64_t qid = query_counter_.fetch_add(1);

        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        return AITQSearchLayer<K>::search(
            q, entry_points, ef, graph_, *quantizer_, qid);
    }

    /**
     * Search with re-ranking using true distance.
     *
     * Uses A-ITQ DISTANCE for candidate generation,
     * then TRUE DISTANCE for final ranking.
     */
    std::vector<SearchResult> search_and_rerank(
        const Float* query_vec,
        size_t k,
        size_t ef = 0,
        size_t rerank_k = 0) const {

        if (ef == 0) ef = std::max(k * 2, static_cast<size_t>(100));
        if (rerank_k == 0) rerank_k = std::max(k * 5, static_cast<size_t>(100));

        // Candidate generation with A-ITQ distance
        auto q = quantizer_->encode_query(query_vec);
        uint64_t qid = query_counter_.fetch_add(1);

        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        auto candidates = AITQSearchLayer<K>::search(
            q, entry_points, rerank_k, graph_, *quantizer_, qid);

        // Re-rank with TRUE distance
#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for if(candidates.size() > 100)
#endif
        for (size_t i = 0; i < candidates.size(); ++i) {
            NodeId id = candidates[i].id;
            const Float* node_vec = original_vectors_.data() + id * params_.dim;

            Float dot = 0;
            for (size_t d = 0; d < params_.dim; ++d) {
                dot += query_vec[d] * node_vec[d];
            }
            candidates[i].distance = -dot;
        }

        // Sort and return top-k
        if (candidates.size() > k) {
            std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
            candidates.resize(k);
        } else {
            std::sort(candidates.begin(), candidates.end());
        }

        return candidates;
    }

    /**
     * Batch search.
     */
    std::vector<std::vector<SearchResult>> search_batch(
        const Float* queries, size_t num_queries, size_t k, size_t ef = 0) const {

        if (ef == 0) ef = std::max(k, static_cast<size_t>(10));

        std::vector<std::vector<SearchResult>> results(num_queries);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries + i * params_.dim, k, ef);
        }

        return results;
    }

    /**
     * Verify graph connectivity.
     */
    size_t verify_connectivity() const {
        if (graph_.empty()) return 0;

        std::vector<bool> visited(graph_.size(), false);
        std::queue<NodeId> queue;

        queue.push(0);
        visited[0] = true;
        size_t count = 1;

        while (!queue.empty()) {
            NodeId node = queue.front();
            queue.pop();

            auto [neighbors, neighbor_count] = graph_.get_neighbors(node);
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor = neighbors[i];
                if (neighbor != INVALID_NODE && !visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                    ++count;
                }
            }
        }

        return count;
    }

    bool is_connected() const {
        return verify_connectivity() == graph_.size();
    }

    /**
     * Get A-ITQ correlation between estimated and true distance.
     * Useful for diagnostic purposes.
     */
    Float compute_correlation(size_t num_samples = 1000) const {
        if (original_vectors_.empty()) return 0.0f;
        size_t N = original_vectors_.size() / params_.dim;
        return quantizer_->compute_correlation(
            original_vectors_.data(), N, num_samples);
    }

private:
    CPHNSWParams params_;
    std::shared_ptr<Quantizer> quantizer_;
    mutable Graph graph_;
    std::mt19937_64 rng_;
    mutable std::atomic<uint64_t> query_counter_;
    std::vector<Float> original_vectors_;

    static CPHNSWParams finalize_params(CPHNSWParams params) {
        params.finalize();
        return params;
    }
};

// Common type aliases
using AITQPolicyIndex128 = AITQPolicyIndex<128>;
using AITQPolicyIndex256 = AITQPolicyIndex<256>;

}  // namespace cphnsw
