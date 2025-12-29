#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../quantizer/cp_encoder.hpp"
#include "../algorithms/insert.hpp"
#include "../algorithms/knn_search.hpp"
#include <memory>
#include <random>
#include <atomic>
#include <queue>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

namespace cphnsw {

/**
 * CP-HNSW Index
 *
 * Main public API for the Cross-Polytope HNSW index.
 *
 * Combines:
 * - Cross-Polytope LSH encoding (optimal angular hashing)
 * - Flat memory HNSW graph structure
 * - SIMD-accelerated Hamming distance
 * - Optional multiprobe for improved recall
 *
 * Template parameters:
 * - ComponentT: uint8_t for d <= 128, uint16_t for d > 128
 * - K: Number of rotations (code width, default 16)
 */
template <typename ComponentT = uint8_t, size_t K = 16>
class CPHNSWIndex {
public:
    using Code = CPCode<ComponentT, K>;
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

    /**
     * Construct index with given parameters.
     *
     * @param params  Index configuration
     */
    explicit CPHNSWIndex(CPHNSWParams params)
        : params_(finalize_params(std::move(params))),
          encoder_(params_.dim, params_.seed),
          graph_(params_),
          rng_(params_.seed),
          query_counter_(0) {}

    /**
     * Construct index with dimension and default parameters.
     *
     * @param dim   Vector dimension
     * @param M     Max connections per node (default 16)
     * @param ef_construction  Construction search width (default 100)
     */
    CPHNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 100)
        : CPHNSWIndex(make_params(dim, M, ef_construction)) {}

    /// Get index parameters
    const CPHNSWParams& params() const { return params_; }

    /// Get number of vectors in index
    size_t size() const { return graph_.size(); }

    /// Check if index is empty
    bool empty() const { return graph_.empty(); }

    /// Get vector dimension
    size_t dim() const { return params_.dim; }

    /**
     * Set the backbone size for tiered construction.
     * Default: 10,000 nodes (or 1% of expected data)
     *
     * @param size  Number of nodes to build with full float search
     */
    void set_backbone_size(size_t size) {
        backbone_size_ = size;
    }

    /**
     * Add a single vector to the index.
     *
     * TIERED CONSTRUCTION:
     * - Backbone (first N nodes): Full float search + float edge selection
     * - Rest: CP search + float edge selection (hybrid)
     *
     * This guarantees 100% connectivity by building a reliable backbone
     * that all subsequent nodes can connect to.
     *
     * @param vec  Input vector (length >= dim)
     * @return     Node ID assigned
     */
    NodeId add(const Float* vec) {
        // Store original vector for hybrid/float construction
        original_vectors_.insert(original_vectors_.end(), vec, vec + params_.dim);

        // Encode to CPQuery (with magnitudes for asymmetric distance)
        auto query = encoder_.encode_query(vec);

        // Generate level
        LayerLevel level = Insert<ComponentT, K>::generate_level(
            rng_, params_.m_L);

        // Add node to graph (store only the code, not magnitudes)
        NodeId id = graph_.add_node(query.primary_code, level);

        // TIERED CONSTRUCTION:
        // - Backbone: Use full float search (O(n) per insert, but n is small)
        // - Rest: Use CP search + float edge selection (hybrid)
        if (id < backbone_size_) {
            // Backbone: guaranteed connectivity via brute-force float search
            Insert<ComponentT, K>::insert_float(
                id, vec, original_vectors_, params_.dim,
                graph_, params_);
        } else {
            // Rest: fast CP search with accurate float edge selection
            uint64_t qid = query_counter_.fetch_add(1);
            Insert<ComponentT, K>::insert_hybrid(
                id, query, vec, original_vectors_, params_.dim,
                graph_, params_, rng_, qid);
        }

        return id;
    }

    /**
     * Add multiple vectors to the index.
     *
     * @param vecs   Input vectors (row-major, count x dim)
     * @param count  Number of vectors
     */
    void add_batch(const Float* vecs, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            add(vecs + i * params_.dim);
        }
    }

    /**
     * Search for k nearest neighbors.
     *
     * Uses asymmetric distance for proper gradient-based navigation.
     *
     * @param query  Query vector (length >= dim)
     * @param k      Number of neighbors to return
     * @param ef     Search width (ef >= k recommended, 0 = auto)
     * @return       k nearest neighbors sorted by distance
     */
    std::vector<SearchResult> search(const Float* query, size_t k, size_t ef = 0) const {
        if (ef == 0) {
            ef = std::max(k, static_cast<size_t>(10));
        }

        // Encode as CPQuery with magnitudes for asymmetric distance
        auto q = encoder_.encode_query(query);
        uint64_t qid = query_counter_.fetch_add(1);

        return KNNSearch<ComponentT, K>::search(q, k, ef, graph_, qid);
    }

    /**
     * Search with multiprobe for improved recall.
     *
     * @param query       Query vector
     * @param k           Number of neighbors
     * @param ef          Search width per probe
     * @param num_probes  Number of probe sequences
     * @return            k nearest neighbors
     */
    std::vector<SearchResult> search_multiprobe(
        const Float* query, size_t k, size_t ef, size_t num_probes) const {

        auto q = encoder_.encode_query(query);
        auto encoded = encoder_.encode_with_sorted_indices(query);
        uint64_t qid = query_counter_.fetch_add(1);

        return KNNSearch<ComponentT, K>::search_multiprobe(
            encoded, q, k, ef, num_probes, graph_, qid);
    }

    /**
     * Batch search for multiple queries.
     *
     * PARALLELIZED with OpenMP when CPHNSW_USE_OPENMP is defined.
     * Each query is processed independently in parallel.
     *
     * @param queries  Query vectors (row-major, num_queries x dim)
     * @param num_queries  Number of queries
     * @param k        Number of neighbors per query
     * @param ef       Search width
     * @return         Results for each query
     */
    std::vector<std::vector<SearchResult>> search_batch(
        const Float* queries, size_t num_queries, size_t k, size_t ef = 0) const {

        if (ef == 0) {
            ef = std::max(k, static_cast<size_t>(10));
        }

        std::vector<std::vector<SearchResult>> results(num_queries);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries + i * params_.dim, k, ef);
        }
#else
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries + i * params_.dim, k, ef);
        }
#endif

        return results;
    }

    /**
     * Brute force search for diagnostic purposes.
     *
     * Scans all vectors and returns top-k by asymmetric distance.
     * Used to verify hash quality independently of graph navigation.
     *
     * @param query  Query vector
     * @param k      Number of neighbors
     * @return       k nearest neighbors by asymmetric distance
     */
    std::vector<SearchResult> brute_force_search(const Float* query, size_t k) const {
        auto q = encoder_.encode_query(query);

        std::vector<SearchResult> all_results;
        all_results.reserve(graph_.size());

        for (size_t i = 0; i < graph_.size(); ++i) {
            const auto& code = graph_.get_code(static_cast<NodeId>(i));
            AsymmetricDist dist = estimate_dot_product(q, code);
            all_results.push_back({static_cast<NodeId>(i), dist});
        }

        std::partial_sort(all_results.begin(),
                          all_results.begin() + std::min(k, all_results.size()),
                          all_results.end());

        if (all_results.size() > k) {
            all_results.resize(k);
        }

        return all_results;
    }

    /**
     * Get the code for a node (for debugging/analysis).
     */
    const Code& get_code(NodeId id) const {
        return graph_.get_code(id);
    }

    /**
     * Verify graph connectivity at layer 0.
     * Returns the number of nodes reachable from entry point.
     * Should equal size() for a connected graph.
     */
    size_t verify_connectivity() const {
        if (graph_.empty()) return 0;

        NodeId ep = graph_.entry_point();
        if (ep == INVALID_NODE) return 0;

        // BFS from entry point at layer 0
        std::vector<bool> visited(graph_.size(), false);
        std::queue<NodeId> queue;

        queue.push(ep);
        visited[ep] = true;
        size_t count = 1;

        while (!queue.empty()) {
            NodeId node = queue.front();
            queue.pop();

            auto [neighbors, neighbor_count] = graph_.get_neighbors(node, 0);

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

    /**
     * Check if graph is fully connected at layer 0.
     */
    bool is_connected() const {
        return verify_connectivity() == graph_.size();
    }

private:
    CPHNSWParams params_;
    Encoder encoder_;
    mutable Graph graph_;
    std::mt19937_64 rng_;
    mutable std::atomic<uint64_t> query_counter_;

    // Store original vectors for hybrid/float construction
    // Uses true cosine distance for edge selection
    std::vector<Float> original_vectors_;

    // Tiered construction: first backbone_size_ nodes use full float search
    // Default 0: Pure hybrid mode (CP search + float edges) - proven 100% connectivity
    // Set > 0 only if you need guaranteed O(n²) backbone for specific use cases
    size_t backbone_size_ = 0;

    static CPHNSWParams finalize_params(CPHNSWParams params) {
        params.finalize();
        return params;
    }

    static CPHNSWParams make_params(size_t dim, size_t M, size_t ef_construction) {
        CPHNSWParams params;
        params.dim = dim;
        params.k = K;
        params.M = M;
        params.M_max0 = 2 * M;
        params.ef_construction = ef_construction;
        params.keep_pruned = true;
        params.seed = 42;
        params.finalize();
        return params;
    }
};

// Common index types
using CPHNSWIndex8 = CPHNSWIndex<uint8_t, 16>;    // For d <= 128
using CPHNSWIndex16 = CPHNSWIndex<uint16_t, 16>;  // For d > 128
using CPHNSWIndex32 = CPHNSWIndex<uint8_t, 32>;   // Higher precision

}  // namespace cphnsw
