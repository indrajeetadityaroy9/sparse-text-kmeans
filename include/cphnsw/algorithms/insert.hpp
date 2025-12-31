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
// INSERT Algorithm (Algorithm 3 from Malkov & Yashunin)
// =============================================================================

/**
 * INSERT Algorithm
 *
 * Inserts a new node into the HNSW graph:
 * 1. Phase 1 (Zoom): Descend from top layer to insert level + 1 with ef=1
 * 2. Phase 2 (Construct): For each layer from insert level down to 0:
 *    - Search for neighbors with ef_construction
 *    - Select neighbors using diversity heuristic
 *    - Add bidirectional edges
 *    - Shrink overflowing neighbor lists
 * 3. Update entry point if new level > top layer
 *
 * Complexity: O(log N * ef_construction * M * K)
 */
template <typename ComponentT, size_t K>
class Insert {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

private:
    // =========================================================================
    // Phase 1: Zoom (shared across all variants)
    // =========================================================================

    /**
     * Phase 1: Greedy descent from top layer to target level.
     *
     * @param new_query     Query for the new node
     * @param ep            Current entry point
     * @param top_layer     Current top layer
     * @param target_level  Stop at this level + 1
     * @param graph         HNSW graph
     * @param policy        Thread policy for query ID generation
     * @return              Best entry points for Phase 2
     */
    template<typename ThreadPolicy>
    static std::vector<NodeId> phase1_zoom(
        const Query& new_query,
        NodeId ep,
        LayerLevel top_layer,
        LayerLevel target_level,
        const Graph& graph,
        ThreadPolicy& policy) {

        std::vector<NodeId> entry_points = {ep};

        for (LayerLevel l = top_layer; l > target_level && l > 0; --l) {
            uint64_t qid = policy.next_id();
            auto results = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, 1, l, graph, qid);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        return entry_points;
    }

    // =========================================================================
    // Phase 2 Helpers
    // =========================================================================

    /**
     * Update entry points from search results.
     */
    static std::vector<NodeId> update_entry_points(
        const std::vector<SearchResult>& candidates,
        NodeId fallback_ep) {

        std::vector<NodeId> entry_points;
        entry_points.reserve(candidates.size());
        for (const auto& r : candidates) {
            entry_points.push_back(r.id);
        }
        if (entry_points.empty()) {
            entry_points.push_back(fallback_ep);
        }
        return entry_points;
    }

public:
    // =========================================================================
    // Public API: insert() - Pure CP distance (legacy)
    // =========================================================================

    /**
     * Insert a new node using asymmetric CP distance only.
     *
     * Uses CP distance for both search and edge selection.
     * May have connectivity issues on adversarial distributions.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_query       Query struct with code and magnitudes
     * @param graph           HNSW graph (modified)
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

        LayerLevel new_level = graph.get_max_layer(new_node_id);
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        // Handle empty graph
        if (ep == INVALID_NODE) {
            graph.set_entry_point(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom
        SingleThreadPolicy policy(query_counter);
        std::vector<NodeId> entry_points = phase1_zoom(
            new_query, ep, top_layer, new_level, graph, policy);

        // Phase 2: Construct with CP distance selection
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            auto candidates = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, params.ef_construction, layer, graph,
                policy.next_id());

            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors = SelectNeighbors<ComponentT, K>::select(
                new_query, candidates, M_limit, params.keep_pruned, graph);

            // Add forward connections
            for (NodeId neighbor_id : neighbors) {
                graph.add_neighbor(new_node_id, layer, neighbor_id);
            }

            // Add reverse connections with overflow handling
            for (NodeId neighbor_id : neighbors) {
                if (graph.get_max_layer(neighbor_id) < layer) continue;

                bool added = graph.add_neighbor(neighbor_id, layer, new_node_id);
                if (!added) {
                    // Prune using Hamming distance
                    auto [neighbor_links, link_count] = graph.get_neighbors(neighbor_id, layer);
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
                        neighbor_candidates, M_limit);
                    graph.set_neighbors(neighbor_id, layer, new_neighbors);
                }
            }

            entry_points = update_entry_points(candidates, ep);
        }

        // Update entry point if needed
        if (new_level > top_layer) {
            graph.set_entry_point(new_node_id, new_level);
        }
    }

    // =========================================================================
    // Public API: insert_hybrid() - CP search + true distance selection
    // =========================================================================

    /**
     * HYBRID INSERT: Uses CP distance for search, TRUE cosine for edge selection.
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

        LayerLevel new_level = graph.get_max_layer(new_node_id);
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        if (ep == INVALID_NODE) {
            graph.set_entry_point(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom
        SingleThreadPolicy policy(query_counter);
        std::vector<NodeId> entry_points = phase1_zoom(
            new_query, ep, top_layer, new_level, graph, policy);

        // Phase 2: Construct with true distance selection
        insert_hybrid_phase2(
            new_node_id, new_query, new_vec, all_vectors, dim,
            entry_points, ep, top_layer, new_level,
            graph, params, policy,
            /* use_safe_methods */ false);
    }

    // =========================================================================
    // Public API: insert_hybrid_parallel() - Thread-safe hybrid insert
    // =========================================================================

    /**
     * PARALLEL HYBRID INSERT: Thread-safe version using atomic query counter.
     *
     * Uses atomic fetch_add for each search to ensure unique query IDs.
     * Uses thread-safe graph methods for concurrent edge modifications.
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

        LayerLevel new_level = graph.get_max_layer(new_node_id);
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        if (ep == INVALID_NODE) {
            graph.set_entry_point_safe(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom with atomic counter
        AtomicThreadPolicy policy(query_counter);
        std::vector<NodeId> entry_points = phase1_zoom(
            new_query, ep, top_layer, new_level, graph, policy);

        // Phase 2: Construct with true distance selection (thread-safe)
        insert_hybrid_phase2(
            new_node_id, new_query, new_vec, all_vectors, dim,
            entry_points, ep, top_layer, new_level,
            graph, params, policy,
            /* use_safe_methods */ true);
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /**
     * Generate random level from exponential distribution.
     *
     * l = floor(-ln(uniform(0,1)) * m_L)
     */
    static LayerLevel generate_level(std::mt19937_64& rng, double m_L,
                                     LayerLevel max_level = 15) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        if (r < 1e-10) r = 1e-10;

        LayerLevel level = static_cast<LayerLevel>(
            std::floor(-std::log(r) * m_L));

        return std::min(level, max_level);
    }

private:
    // =========================================================================
    // Phase 2 Implementation (shared between hybrid variants)
    // =========================================================================

    /**
     * Phase 2: Construction phase for hybrid insertion.
     *
     * Unified implementation for both single-threaded and parallel variants.
     * Selects neighbors using true cosine distance with cached distances.
     */
    template<typename ThreadPolicy>
    static void insert_hybrid_phase2(
        NodeId new_node_id,
        const Query& new_query,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<NodeId> entry_points,
        NodeId ep,
        LayerLevel top_layer,
        LayerLevel new_level,
        Graph& graph,
        const CPHNSWParams& params,
        ThreadPolicy& policy,
        bool use_safe_methods) {

        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            uint64_t qid = policy.next_id();
            auto candidates = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, params.ef_construction, layer, graph, qid);

            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors_with_dist = SelectNeighbors<ComponentT, K>::select_with_true_distance_cached(
                new_vec, all_vectors, dim, candidates, M_limit, params.keep_pruned);

            // Add forward connections with cached distance
            for (const auto& [neighbor_id, dist] : neighbors_with_dist) {
                graph.add_neighbor_with_dist_safe(new_node_id, layer, neighbor_id, dist);
            }

            // Add reverse connections
            for (const auto& [neighbor_id, dist_to_new] : neighbors_with_dist) {
                if (graph.get_max_layer(neighbor_id) < layer) continue;
                graph.add_neighbor_with_dist_safe(neighbor_id, layer, new_node_id, dist_to_new);
            }

            entry_points = update_entry_points(candidates, ep);
        }

        // Update entry point
        if (new_level > top_layer) {
            if (use_safe_methods) {
                graph.set_entry_point_safe(new_node_id, new_level);
            } else {
                graph.set_entry_point(new_node_id, new_level);
            }
        }
    }
};

}  // namespace cphnsw
