#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../quantizer/cp_encoder.hpp"
#include "search_layer.hpp"
#include "select_neighbors.hpp"
#include <random>
#include <cmath>

namespace cphnsw {

/**
 * INSERT Algorithm (Algorithm 3 from Malkov & Yashunin)
 *
 * Inserts a new node into the HNSW graph:
 * 1. Generate random level from exponential distribution
 * 2. Phase 1 (Zoom): Descend from top layer to insert level + 1 with ef=1
 * 3. Phase 2 (Construct): For each layer from insert level down to 0:
 *    - Search for neighbors with ef_construction
 *    - Select neighbors using diversity heuristic
 *    - Add bidirectional edges
 *    - Shrink overflowing neighbor lists
 * 4. Update entry point if new level > top layer
 *
 * CRITICAL: Uses asymmetric distance during search phases to ensure
 * proper gradient-based navigation even with quantized codes.
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

    /**
     * Insert a new node into the graph using asymmetric distance.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_query       Query struct with code and magnitudes
     * @param graph           HNSW graph (modified)
     * @param params          Index parameters
     * @param rng             Random number generator
     * @param query_counter   Atomic counter for query IDs
     */
    static void insert(
        NodeId new_node_id,
        const Query& new_query,
        Graph& graph,
        const CPHNSWParams& params,
        std::mt19937_64& rng,
        uint64_t& query_counter) {

        LayerLevel new_level = graph.get_max_layer(new_node_id);

        // Get current entry point and top layer
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        // Handle empty graph
        if (ep == INVALID_NODE) {
            graph.set_entry_point(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom from top layer down to new_level + 1
        // Use ef=1 for greedy descent with asymmetric distance
        std::vector<NodeId> entry_points = {ep};

        for (LayerLevel l = top_layer; l > new_level && l > 0; --l) {
            auto results = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, 1, l, graph, ++query_counter);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        // Phase 2: Insert at layers min(top_layer, new_level) down to 0
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            // Search for neighbors using asymmetric distance
            auto candidates = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, params.ef_construction, layer, graph,
                ++query_counter);

            // Select best neighbors using heuristic
            // Note: candidate.distance is now asymmetric (continuous)
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors = SelectNeighbors<ComponentT, K>::select(
                new_query, candidates, M_limit, params.keep_pruned, graph);

            // Add connections from new node to neighbors
            for (NodeId neighbor_id : neighbors) {
                graph.add_neighbor(new_node_id, layer, neighbor_id);
            }

            // Add reverse connections and shrink if overflow
            for (NodeId neighbor_id : neighbors) {
                // Check if neighbor exists at this layer
                if (graph.get_max_layer(neighbor_id) < layer) {
                    continue;
                }

                // Add reverse connection
                bool added = graph.add_neighbor(neighbor_id, layer, new_node_id);

                // If failed to add (overflow), need to prune
                // Use symmetric Hamming distance for shrink (simpler, still effective)
                if (!added) {
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

                    // Add new node to candidates
                    HammingDist d_new = hamming_distance(neighbor_code, new_query.primary_code);
                    neighbor_candidates.push_back({new_node_id, static_cast<AsymmetricDist>(d_new)});

                    // Re-select using heuristic (with Hamming distances)
                    auto new_neighbors = SelectNeighbors<ComponentT, K>::select_simple(
                        neighbor_candidates, M_limit);

                    // Update neighbor's connections
                    graph.set_neighbors(neighbor_id, layer, new_neighbors);
                }
            }

            // Update entry points for next layer
            entry_points.clear();
            for (const auto& r : candidates) {
                entry_points.push_back(r.id);
            }
            if (entry_points.empty()) {
                entry_points.push_back(ep);
            }
        }

        // Update global entry point if needed
        if (new_level > top_layer) {
            graph.set_entry_point(new_node_id, new_level);
        }
    }

    /**
     * HYBRID INSERT: Uses CP distance for search, TRUE cosine for edge selection.
     *
     * This ensures 100% graph connectivity by using exact float distances
     * for the critical edge selection decisions.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_query       Query struct with code and magnitudes
     * @param new_vec         Original float vector being inserted
     * @param all_vectors     All original vectors stored so far
     * @param dim             Vector dimension
     * @param graph           HNSW graph (modified)
     * @param params          Index parameters
     * @param rng             Random number generator
     * @param query_counter   Atomic counter for query IDs
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

        // Get current entry point and top layer
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        // Handle empty graph
        if (ep == INVALID_NODE) {
            graph.set_entry_point(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom from top layer down to new_level + 1
        // Use CP distance for fast candidate search
        std::vector<NodeId> entry_points = {ep};

        for (LayerLevel l = top_layer; l > new_level && l > 0; --l) {
            auto results = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, 1, l, graph, ++query_counter);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        // Phase 2: Insert at layers min(top_layer, new_level) down to 0
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            // Search for candidates using CP distance (fast)
            auto candidates = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, params.ef_construction, layer, graph,
                ++query_counter);

            // Select best neighbors using TRUE cosine distance (with cached distances)
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors_with_dist = SelectNeighbors<ComponentT, K>::select_with_true_distance_cached(
                new_vec, all_vectors, dim, candidates, M_limit, params.keep_pruned);

            // Add connections from new node to neighbors (with cached distance)
            for (const auto& [neighbor_id, dist] : neighbors_with_dist) {
                graph.add_neighbor_with_dist_safe(new_node_id, layer, neighbor_id, dist);
            }

            // Add reverse connections using CACHED DISTANCE PRUNING
            for (const auto& [neighbor_id, dist_to_new] : neighbors_with_dist) {
                if (graph.get_max_layer(neighbor_id) < layer) {
                    continue;
                }

                // Use cached distance pruning (no vector fetches in critical section)
                graph.add_neighbor_with_dist_safe(neighbor_id, layer, new_node_id, dist_to_new);
            }

            // Update entry points for next layer
            entry_points.clear();
            for (const auto& r : candidates) {
                entry_points.push_back(r.id);
            }
            if (entry_points.empty()) {
                entry_points.push_back(ep);
            }
        }

        // Update global entry point if needed
        if (new_level > top_layer) {
            graph.set_entry_point(new_node_id, new_level);
        }
    }

    /**
     * PARALLEL HYBRID INSERT: Thread-safe version using atomic query counter.
     *
     * CRITICAL: Uses atomic fetch_add for EACH search to ensure unique query IDs.
     * Uses thread-safe graph methods for concurrent edge modifications.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_query       Query struct with code and magnitudes
     * @param new_vec         Original float vector being inserted
     * @param all_vectors     All original vectors stored so far
     * @param dim             Vector dimension
     * @param graph           HNSW graph (modified)
     * @param params          Index parameters
     * @param query_counter   ATOMIC counter for query IDs (fresh ID per search)
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

        // Get current entry point and top layer
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        // Handle empty graph (should not happen in parallel phase)
        if (ep == INVALID_NODE) {
            graph.set_entry_point_safe(new_node_id, new_level);
            return;
        }

        // Phase 1: Zoom from top layer down to new_level + 1
        // CRITICAL: Fresh query_id for EACH search
        std::vector<NodeId> entry_points = {ep};

        for (LayerLevel l = top_layer; l > new_level && l > 0; --l) {
            uint64_t qid = query_counter.fetch_add(1);  // Fresh ID!
            auto results = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, 1, l, graph, qid);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        // Phase 2: Insert at layers min(top_layer, new_level) down to 0
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            // CRITICAL: Fresh query_id for this search
            uint64_t qid = query_counter.fetch_add(1);
            auto candidates = SearchLayer<ComponentT, K>::search(
                new_query, entry_points, params.ef_construction, layer, graph, qid);

            // Select best neighbors using TRUE cosine distance (with cached distances)
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors_with_dist = SelectNeighbors<ComponentT, K>::select_with_true_distance_cached(
                new_vec, all_vectors, dim, candidates, M_limit, params.keep_pruned);

            // Add connections from new node to neighbors (thread-safe, with cached distance)
            for (const auto& [neighbor_id, dist] : neighbors_with_dist) {
                graph.add_neighbor_with_dist_safe(new_node_id, layer, neighbor_id, dist);
            }

            // Add reverse connections using CACHED DISTANCE PRUNING
            // CRITICAL OPTIMIZATION: No more vector fetches inside critical section!
            for (const auto& [neighbor_id, dist_to_new] : neighbors_with_dist) {
                if (graph.get_max_layer(neighbor_id) < layer) {
                    continue;
                }

                // Compute reverse distance (from neighbor's perspective to new node)
                // For symmetric distance (like cosine), this equals dist_to_new
                // The graph.add_neighbor_with_dist_safe handles overflow via O(M) cached scan
                graph.add_neighbor_with_dist_safe(neighbor_id, layer, new_node_id, dist_to_new);
            }

            // Update entry points for next layer
            entry_points.clear();
            for (const auto& r : candidates) {
                entry_points.push_back(r.id);
            }
            if (entry_points.empty()) {
                entry_points.push_back(ep);
            }
        }

        // Update global entry point if needed (thread-safe)
        if (new_level > top_layer) {
            graph.set_entry_point_safe(new_node_id, new_level);
        }
    }

    /**
     * Generate random level from exponential distribution.
     *
     * l = floor(-ln(uniform(0,1)) * m_L)
     *
     * @param rng     Random number generator
     * @param m_L     Level multiplier (1/ln(M))
     * @param max_level  Maximum allowed level
     * @return        Generated level
     */
    static LayerLevel generate_level(std::mt19937_64& rng, double m_L,
                                     LayerLevel max_level = 15) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);

        // Avoid log(0)
        if (r < 1e-10) r = 1e-10;

        LayerLevel level = static_cast<LayerLevel>(
            std::floor(-std::log(r) * m_L));

        return std::min(level, max_level);
    }
};

}  // namespace cphnsw
