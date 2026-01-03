#pragma once

#include "config.hpp"
#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "../distance/metric_policy.hpp"
#include "../graph/flat_graph.hpp"
#include "../search/search_engine.hpp"
#include "../encoder/cp_encoder.hpp"
#include <vector>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cphnsw {

// ============================================================================
// CPHNSWIndex: Unified Index API
// ============================================================================

/**
 * CPHNSWIndex: The main entry point for CP-HNSW.
 *
 * UNIFIED DESIGN: A single index class that handles both Phase 1 (R=0)
 * and Phase 2 (R>0) via template parameters. When R=0, all residual
 * operations compile away, yielding zero overhead.
 *
 * TYPICAL USAGE:
 *
 *   // Phase 1: Pure RaBitQ (32-bit codes)
 *   CPHNSWIndex<32> index(params);
 *   index.add_batch(vectors, num_vectors);
 *   auto results = index.search(query, search_params);
 *
 *   // Phase 2: With residual (64+32-bit codes)
 *   CPHNSWIndex<64, 32> index_residual(params);
 *   index_residual.add_batch(vectors, num_vectors);
 *   auto results = index_residual.search(query, search_params);
 *
 * @tparam K Primary code bits (32, 64)
 * @tparam R Residual code bits (0 = Phase 1, 16/32 = Phase 2)
 * @tparam Shift Bit-shift for residual weighting (default 2 = 4:1)
 */
template <size_t K, size_t R = 0, int Shift = 2>
class CPHNSWIndex {
public:
    // Type aliases
    using CodeType = ResidualCode<K, R>;
    using QueryType = CodeQuery<K, R, Shift>;
    using Policy = UnifiedMetricPolicy<K, R, Shift>;
    using Graph = FlatGraph<CodeType>;
    using Encoder = CPEncoder<K, R>;
    using Engine = SearchEngine<Policy>;

    // Configuration
    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int WEIGHT_SHIFT = Shift;
    static constexpr bool HAS_RESIDUAL = (R > 0);

    // ========================================================================
    // Construction
    // ========================================================================

    /**
     * Construct index with configuration.
     */
    explicit CPHNSWIndex(const IndexParams& params)
        : params_(params)
        , encoder_(params.dim, params.seed)
        , graph_(params.initial_capacity, params.M)
        , rng_(params.seed) {

        if (params.dim == 0) {
            throw std::invalid_argument("dim must be > 0");
        }
    }

    /**
     * Construct with dimension only (uses default parameters).
     */
    explicit CPHNSWIndex(size_t dim)
        : CPHNSWIndex(IndexParams().set_dim(dim)) {}

    // ========================================================================
    // Adding Vectors
    // ========================================================================

    /**
     * Add a single vector to the index.
     * Thread-safe.
     *
     * @return The ID of the added vector
     */
    NodeId add(const float* vec) {
        // Encode vector
        CodeType code = encoder_.encode(vec);

        // Add to graph
        NodeId id = graph_.add_node(code);

        // Connect to graph using NSW insertion
        if (id > 0) {
            insert_into_graph(id, vec, code);
        }

        // Store original vector for reranking (if enabled)
        {
            std::lock_guard<std::mutex> lock(vectors_mutex_);
            if (id >= vectors_.size()) {
                vectors_.resize(id + 1);
            }
            vectors_[id].assign(vec, vec + params_.dim);
        }

        return id;
    }

    /**
     * Add multiple vectors in batch (parallelized).
     */
    void add_batch(const float* vecs, size_t num_vecs,
                   const BuildParams& build_params = BuildParams()) {

        if (num_vecs == 0) return;

        // Pre-allocate
        graph_.reserve(graph_.size() + num_vecs);
        vectors_.resize(graph_.size() + num_vecs);

        // Encode all vectors in parallel
        std::vector<CodeType> codes(num_vecs);
        encoder_.encode_batch(vecs, num_vecs, codes.data());

        // Add nodes
        std::vector<NodeId> ids(num_vecs);
        for (size_t i = 0; i < num_vecs; ++i) {
            ids[i] = graph_.add_node(codes[i]);
            vectors_[ids[i]].assign(vecs + i * params_.dim,
                                     vecs + (i + 1) * params_.dim);
        }

        // Connect nodes in parallel
        size_t num_threads = build_params.num_threads;
        if (num_threads == 0) {
#ifdef _OPENMP
            num_threads = omp_get_max_threads();
#else
            num_threads = 1;
#endif
        }

#ifdef _OPENMP
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
        for (size_t i = 0; i < num_vecs; ++i) {
            if (ids[i] > 0) {  // Skip first node
                insert_into_graph(ids[i], vecs + i * params_.dim, codes[i]);
            }
        }
    }

    // ========================================================================
    // Searching
    // ========================================================================

    /**
     * Search for k nearest neighbors.
     *
     * @param query Query vector
     * @param params Search parameters
     * @return Vector of (id, distance) pairs, sorted by distance
     */
    std::vector<SearchResult> search(const float* query,
                                      const SearchParams& params = SearchParams()) const {

        if (graph_.empty()) return {};

        // Encode query
        float avg_norm = compute_average_norm();
        QueryType encoded_query = encoder_.encode_query(query, avg_norm);

        // Get entry point(s)
        std::vector<NodeId> entry_points;
        if (params.num_entry_points == 1) {
            entry_points.push_back(graph_.entry_point());
        } else {
            std::mt19937_64 local_rng(rng_());
            entry_points = graph_.get_random_entry_points(params.num_entry_points, local_rng);
        }

        // Perform graph search
        auto candidates = Engine::search(
            encoded_query,
            entry_points,
            params.ef,
            graph_,
            params.rerank ? params.rerank_k : params.k);

        // Rerank with exact distances if enabled
        if (params.rerank && !candidates.empty()) {
            candidates = rerank(query, candidates, params.k);
        }

        // Trim to k
        if (candidates.size() > params.k) {
            candidates.resize(params.k);
        }

        return candidates;
    }

    /**
     * Search with default parameters.
     */
    std::vector<SearchResult> search(const float* query, size_t k) const {
        return search(query, SearchParams().set_k(k));
    }

    // ========================================================================
    // Batch Search
    // ========================================================================

    /**
     * Search for multiple queries in parallel.
     *
     * @param queries Query vectors (num_queries x dim)
     * @param num_queries Number of queries
     * @param params Search parameters
     * @return Vector of result vectors, one per query
     */
    std::vector<std::vector<SearchResult>> search_batch(
        const float* queries,
        size_t num_queries,
        const SearchParams& params = SearchParams()) const {

        std::vector<std::vector<SearchResult>> results(num_queries);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries + i * params_.dim, params);
        }

        return results;
    }

    // ========================================================================
    // Index Information
    // ========================================================================

    size_t size() const { return graph_.size(); }
    bool empty() const { return graph_.empty(); }
    size_t dim() const { return params_.dim; }

    const IndexParams& params() const { return params_; }
    const Graph& graph() const { return graph_; }

    /**
     * Get graph statistics.
     */
    struct Stats {
        size_t num_nodes;
        float avg_degree;
        size_t max_degree;
        size_t isolated_nodes;
    };

    Stats get_stats() const {
        return Stats{
            graph_.size(),
            graph_.average_degree(),
            graph_.max_degree(),
            graph_.count_isolated()
        };
    }

private:
    IndexParams params_;
    Encoder encoder_;
    Graph graph_;
    mutable std::mt19937_64 rng_;

    // Original vectors for reranking
    std::vector<std::vector<float>> vectors_;
    mutable std::mutex vectors_mutex_;

    // Cached average norm for distance scaling
    mutable float cached_avg_norm_ = 1.0f;
    mutable bool norm_valid_ = false;

    /**
     * Insert a node into the graph using NSW algorithm.
     */
    void insert_into_graph(NodeId id, const float* vec, const CodeType& code) {
        // Encode as query for search
        QueryType query = encoder_.encode_query(vec, 1.0f);

        // Find nearest neighbors in existing graph
        auto neighbors = Engine::search(
            query,
            graph_.entry_point(),
            params_.ef_construction,
            graph_,
            params_.M);

        // Add edges
        for (const auto& neighbor : neighbors) {
            if (neighbor.id == id) continue;

            // Bidirectional edges
            graph_.try_add_neighbor_safe(id, neighbor.id,
                                          graph_.get_code(neighbor.id),
                                          neighbor.distance);
            graph_.try_add_neighbor_safe(neighbor.id, id,
                                          code, neighbor.distance);
        }
    }

    /**
     * Rerank candidates using exact distances.
     */
    std::vector<SearchResult> rerank(const float* query,
                                      const std::vector<SearchResult>& candidates,
                                      size_t k) const {
        std::vector<SearchResult> reranked;
        reranked.reserve(candidates.size());

        for (const auto& cand : candidates) {
            if (cand.id >= vectors_.size() || vectors_[cand.id].empty()) {
                reranked.push_back(cand);  // Keep original distance
                continue;
            }

            // Compute exact distance (L2 or dot product depending on normalization)
            float dist = 0.0f;
            const auto& vec = vectors_[cand.id];
            for (size_t i = 0; i < params_.dim; ++i) {
                float diff = query[i] - vec[i];
                dist += diff * diff;
            }

            reranked.push_back({cand.id, dist});
        }

        // Sort by distance
        std::sort(reranked.begin(), reranked.end());

        // Trim to k
        if (reranked.size() > k) {
            reranked.resize(k);
        }

        return reranked;
    }

    /**
     * Compute average norm of stored vectors.
     */
    float compute_average_norm() const {
        if (norm_valid_) return cached_avg_norm_;

        if (vectors_.empty()) return 1.0f;

        double sum = 0.0;
        size_t count = 0;

        for (const auto& vec : vectors_) {
            if (vec.empty()) continue;

            double norm = 0.0;
            for (float x : vec) {
                norm += x * x;
            }
            sum += std::sqrt(norm);
            ++count;
        }

        if (count > 0) {
            cached_avg_norm_ = static_cast<float>(sum / count);
            norm_valid_ = true;
        }

        return cached_avg_norm_;
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1: Pure RaBitQ
using Index32 = CPHNSWIndex<32, 0>;
using Index64 = CPHNSWIndex<64, 0>;

// Phase 2: With residual
using Index64_32 = CPHNSWIndex<64, 32, 2>;
using Index32_16 = CPHNSWIndex<32, 16, 2>;

}  // namespace cphnsw
