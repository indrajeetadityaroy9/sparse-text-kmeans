#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
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
 * Complexity: O(log N * ef_construction * M * K)
 */
template <typename ComponentT, size_t K>
class Insert {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;

    /**
     * Insert a new node into the graph.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param graph           HNSW graph (modified)
     * @param params          Index parameters
     * @param rng             Random number generator
     * @param query_counter   Atomic counter for query IDs
     */
    static void insert(
        NodeId new_node_id,
        Graph& graph,
        const CPHNSWParams& params,
        std::mt19937_64& rng,
        uint64_t& query_counter) {

        // Get the new node's code and level
        const auto& new_code = graph.get_code(new_node_id);
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
        // Use ef=1 for greedy descent
        std::vector<NodeId> entry_points = {ep};

        for (LayerLevel l = top_layer; l > new_level && l > 0; --l) {
            auto results = SearchLayer<ComponentT, K>::search(
                new_code, entry_points, 1, l, graph, ++query_counter);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        // Phase 2: Insert at layers min(top_layer, new_level) down to 0
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);

            // Search for neighbors at this layer
            auto candidates = SearchLayer<ComponentT, K>::search(
                new_code, entry_points, params.ef_construction, layer, graph,
                ++query_counter);

            // Select best neighbors using heuristic
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors = SelectNeighbors<ComponentT, K>::select(
                new_code, candidates, params.M, params.keep_pruned, graph);

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
                if (!added) {
                    // Get current neighbors of neighbor
                    auto [neighbor_links, link_count] = graph.get_neighbors(neighbor_id, layer);

                    // Build candidate list including new node
                    std::vector<SearchResult> neighbor_candidates;
                    neighbor_candidates.reserve(link_count + 1);

                    const auto& neighbor_code = graph.get_code(neighbor_id);

                    for (size_t i = 0; i < link_count; ++i) {
                        if (neighbor_links[i] != INVALID_NODE) {
                            HammingDist d = hamming_distance(
                                neighbor_code, graph.get_code(neighbor_links[i]));
                            neighbor_candidates.push_back({neighbor_links[i], d});
                        }
                    }

                    // Add new node to candidates
                    HammingDist d_new = hamming_distance(neighbor_code, new_code);
                    neighbor_candidates.push_back({new_node_id, d_new});

                    // Re-select using heuristic
                    auto new_neighbors = SelectNeighbors<ComponentT, K>::select(
                        neighbor_code, neighbor_candidates, M_limit,
                        params.keep_pruned, graph);

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
                entry_points.push_back(ep);  // Fallback
            }
        }

        // Update global entry point if needed
        if (new_level > top_layer) {
            graph.set_entry_point(new_node_id, new_level);
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
