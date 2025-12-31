#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../quantizer/cp_encoder.hpp"
#include "search_layer.hpp"
#include "select_neighbors.hpp"
#include <random>
#include <cmath>
#include <atomic>

namespace cphnsw {

// =============================================================================
// Thread Policy Classes
// =============================================================================

/**
 * Single-threaded query counter policy.
 * Uses simple increment on a reference.
 */
struct SingleThreadPolicy {
    uint64_t& counter;

    explicit SingleThreadPolicy(uint64_t& c) : counter(c) {}

    uint64_t next_id() { return ++counter; }
};

/**
 * Multi-threaded query counter policy.
 * Uses atomic fetch_add for thread safety.
 */
struct AtomicThreadPolicy {
    std::atomic<uint64_t>& counter;

    explicit AtomicThreadPolicy(std::atomic<uint64_t>& c) : counter(c) {}

    uint64_t next_id() { return counter.fetch_add(1); }
};

// =============================================================================
// NSW INSERT Algorithm (Simplified for single-layer graph)
// =============================================================================

/**
 * NSW INSERT Algorithm (Flat graph - no hierarchy)
 *
 * Per "Down with the Hierarchy" paper - single layer with random entry points
 * achieves equivalent performance for high-dimensional data.
 *
 * Algorithm:
 * 1. Get random entry points
 * 2. Search for neighbors with ef_construction
 * 3. Select neighbors using diversity heuristic
 * 4. Add bidirectional edges
 *
 * Complexity: O(ef_construction * M * K)
 */
template <typename ComponentT, size_t K>
class Insert {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

private:
    /**
     * Update entry points from search results.
     */
    static std::vector<NodeId> update_entry_points(
        const std::vector<SearchResult>& candidates,
        const std::vector<NodeId>& fallback_eps) {

        std::vector<NodeId> entry_points;
        entry_points.reserve(candidates.size());
        for (const auto& r : candidates) {
            entry_points.push_back(r.id);
        }
        if (entry_points.empty()) {
            return fallback_eps;
        }
        return entry_points;
    }

public:
    // =========================================================================
    // Public API: insert() - Pure CP distance (legacy)
    // =========================================================================

    /**
     * NSW Insert a new node using asymmetric CP distance only.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_query       Query struct with code and magnitudes
     * @param graph           NSW graph (modified)
     * @param params          Index parameters
     * @param rng             Random number generator (unused, for API compat)
     * @param query_counter   Counter for query IDs
     */
    static void insert(
        NodeId new_node_id,
        const Query& new_query,
        Graph& graph,
        const CPHNSWParams& params,
        std::mt19937_64& /* rng */,
        uint64_t& query_counter) {

        // Handle first node (no neighbors to connect to)
        if (graph.size() == 1) {
            return;
        }

        // NSW: Get random entry points
        std::vector<NodeId> entry_points = graph.get_random_entry_points(
            params.k_entry, query_counter);

        SingleThreadPolicy policy(query_counter);

        // Single-layer search for candidates
        auto candidates = SearchLayer<ComponentT, K>::search(
            new_query, entry_points, params.ef_construction, graph,
            policy.next_id());

        // Select neighbors using diversity heuristic
        auto neighbors = SelectNeighbors<ComponentT, K>::select(
            new_query, candidates, params.M, params.keep_pruned, graph);

        // Add bidirectional connections
        for (NodeId neighbor_id : neighbors) {
            // Forward: new_node -> neighbor
            graph.add_neighbor(new_node_id, neighbor_id);

            // Reverse: neighbor -> new_node (with overflow handling)
            bool added = graph.add_neighbor(neighbor_id, new_node_id);
            if (!added) {
                // Prune using Hamming distance
                auto [neighbor_links, link_count] = graph.get_neighbors(neighbor_id);
                std::vector<SearchResult> neighbor_candidates;
                neighbor_candidates.reserve(link_count + 1);

                const auto& neighbor_code = graph.get_code(neighbor_id);
                for (size_t i = 0; i < link_count; ++i) {
                    if (neighbor_links[i] != INVALID_NODE) {
                        HammingDist d = hamming_distance(
                            neighbor_code, graph.get_code(neighbor_links[i]));
                        neighbor_candidates.push_back({neighbor_links[i], static_cast<AsymmetricDist>(d)});
                    }
                }
                HammingDist d_new = hamming_distance(neighbor_code, new_query.primary_code);
                neighbor_candidates.push_back({new_node_id, static_cast<AsymmetricDist>(d_new)});

                auto new_neighbors = SelectNeighbors<ComponentT, K>::select_simple(
                    neighbor_candidates, params.M);
                graph.set_neighbors(neighbor_id, new_neighbors);
            }
        }
    }

    // =========================================================================
    // Public API: insert_hybrid() - CP search + true distance selection
    // =========================================================================

    /**
     * NSW HYBRID INSERT: Uses CP distance for search, TRUE cosine for edge selection.
     *
     * Ensures 100% graph connectivity by using exact float distances
     * for critical edge selection decisions.
     */
    static void insert_hybrid(
        NodeId new_node_id,
        const Query& new_query,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        Graph& graph,
        const CPHNSWParams& params,
        std::mt19937_64& /* rng */,
        uint64_t& query_counter) {

        // Handle first node
        if (graph.size() == 1) {
            return;
        }

        // NSW: Get random entry points
        std::vector<NodeId> entry_points = graph.get_random_entry_points(
            params.k_entry, query_counter);

        SingleThreadPolicy policy(query_counter);

        insert_hybrid_core(
            new_node_id, new_query, new_vec, all_vectors, dim,
            entry_points, graph, params, policy);
    }

    // =========================================================================
    // Public API: insert_hybrid_parallel() - Thread-safe hybrid insert
    // =========================================================================

    /**
     * NSW PARALLEL HYBRID INSERT: Thread-safe version using atomic query counter.
     */
    static void insert_hybrid_parallel(
        NodeId new_node_id,
        const Query& new_query,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        Graph& graph,
        const CPHNSWParams& params,
        std::atomic<uint64_t>& query_counter) {

        // Handle first node
        if (graph.size() == 1) {
            return;
        }

        // NSW: Get random entry points (use atomic counter for seed)
        uint64_t seed = query_counter.fetch_add(1);
        std::vector<NodeId> entry_points = graph.get_random_entry_points(
            params.k_entry, seed);

        AtomicThreadPolicy policy(query_counter);

        insert_hybrid_core(
            new_node_id, new_query, new_vec, all_vectors, dim,
            entry_points, graph, params, policy);
    }

    // NOTE: generate_level() removed for NSW flatten (no hierarchy)

private:
    // =========================================================================
    // Core Implementation (shared between hybrid variants)
    // =========================================================================

    /**
     * NSW Core hybrid insertion logic.
     *
     * Single-layer construction with true cosine distance for edge selection.
     */
    template<typename ThreadPolicy>
    static void insert_hybrid_core(
        NodeId new_node_id,
        const Query& new_query,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        const std::vector<NodeId>& entry_points,
        Graph& graph,
        const CPHNSWParams& params,
        ThreadPolicy& policy) {

        uint64_t qid = policy.next_id();

        // Single-layer search for candidates
        auto candidates = SearchLayer<ComponentT, K>::search(
            new_query, entry_points, params.ef_construction, graph, qid);

        // Select neighbors with true cosine distance
        auto neighbors_with_dist = SelectNeighbors<ComponentT, K>::select_with_true_distance_cached(
            new_vec, all_vectors, dim, candidates, params.M, params.keep_pruned);

        // Add bidirectional connections with cached distances
        for (const auto& [neighbor_id, dist] : neighbors_with_dist) {
            // Forward: new_node -> neighbor
            graph.add_neighbor_with_dist_safe(new_node_id, neighbor_id, dist);

            // Reverse: neighbor -> new_node
            graph.add_neighbor_with_dist_safe(neighbor_id, new_node_id, dist);
        }
    }
};

}  // namespace cphnsw
