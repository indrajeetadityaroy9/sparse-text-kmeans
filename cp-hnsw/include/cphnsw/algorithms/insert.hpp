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

            // Select best neighbors using TRUE cosine distance (accurate)
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;
            auto neighbors = SelectNeighbors<ComponentT, K>::select_with_true_distance(
                new_vec, all_vectors, dim, candidates, M_limit, params.keep_pruned);

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

                // If failed to add (overflow), need to prune using TRUE distance
                if (!added) {
                    auto [neighbor_links, link_count] = graph.get_neighbors(neighbor_id, layer);

                    std::vector<SearchResult> neighbor_candidates;
                    neighbor_candidates.reserve(link_count + 1);

                    // Get neighbor's original vector
                    const Float* neighbor_vec = &all_vectors[neighbor_id * dim];

                    // Compute TRUE distances from neighbor to all its links
                    for (size_t i = 0; i < link_count; ++i) {
                        if (neighbor_links[i] != INVALID_NODE) {
                            const Float* link_vec = &all_vectors[neighbor_links[i] * dim];
                            Float dot = 0;
                            for (size_t d = 0; d < dim; ++d) {
                                dot += neighbor_vec[d] * link_vec[d];
                            }
                            neighbor_candidates.push_back({neighbor_links[i], -dot});
                        }
                    }

                    // Add new node to candidates with TRUE distance
                    Float dot_new = 0;
                    for (size_t d = 0; d < dim; ++d) {
                        dot_new += neighbor_vec[d] * new_vec[d];
                    }
                    neighbor_candidates.push_back({new_node_id, -dot_new});

                    // Re-select using simple heuristic (all distances are TRUE now)
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
     * FLOAT BACKBONE INSERT: Uses TRUE cosine for BOTH search and edge selection.
     *
     * Used for the first N nodes (backbone) to guarantee 100% connectivity.
     * O(n) per insertion, but n is small for backbone (e.g., 10,000 nodes).
     *
     * After backbone is built, use insert_hybrid for remaining nodes.
     *
     * @param new_node_id     ID of the new node (already added to graph)
     * @param new_vec         Original float vector being inserted
     * @param all_vectors     All original vectors stored so far
     * @param dim             Vector dimension
     * @param graph           HNSW graph (modified)
     * @param params          Index parameters
     */
    static void insert_float(
        NodeId new_node_id,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        Graph& graph,
        const CPHNSWParams& params) {

        LayerLevel new_level = graph.get_max_layer(new_node_id);

        // Get current entry point and top layer
        NodeId ep = graph.entry_point();
        LayerLevel top_layer = graph.top_layer();

        // Handle empty graph
        if (ep == INVALID_NODE) {
            graph.set_entry_point(new_node_id, new_level);
            return;
        }

        // For backbone: brute force search using TRUE cosine distance
        // Find all candidates by scanning existing nodes
        std::vector<SearchResult> all_candidates;
        all_candidates.reserve(new_node_id);

        for (NodeId i = 0; i < new_node_id; ++i) {
            const Float* node_vec = &all_vectors[i * dim];
            Float dot = 0;
            for (size_t d = 0; d < dim; ++d) {
                dot += new_vec[d] * node_vec[d];
            }
            all_candidates.push_back({i, -dot});  // Negative for min-heap
        }

        // Sort by true distance
        std::sort(all_candidates.begin(), all_candidates.end());

        // Insert at layers min(top_layer, new_level) down to 0
        LayerLevel start_layer = std::min(top_layer, new_level);

        for (int l = static_cast<int>(start_layer); l >= 0; --l) {
            LayerLevel layer = static_cast<LayerLevel>(l);
            size_t M_limit = (layer == 0) ? params.M_max0 : params.M;

            // Filter candidates that exist at this layer
            std::vector<SearchResult> layer_candidates;
            for (const auto& c : all_candidates) {
                if (graph.get_max_layer(c.id) >= layer) {
                    layer_candidates.push_back(c);
                    if (layer_candidates.size() >= params.ef_construction) break;
                }
            }

            // Select neighbors using TRUE distance (already computed)
            auto neighbors = SelectNeighbors<ComponentT, K>::select_with_true_distance(
                new_vec, all_vectors, dim, layer_candidates, M_limit, params.keep_pruned);

            // Add connections from new node to neighbors
            for (NodeId neighbor_id : neighbors) {
                graph.add_neighbor(new_node_id, layer, neighbor_id);
            }

            // Add reverse connections and shrink if overflow
            for (NodeId neighbor_id : neighbors) {
                if (graph.get_max_layer(neighbor_id) < layer) {
                    continue;
                }

                bool added = graph.add_neighbor(neighbor_id, layer, new_node_id);

                if (!added) {
                    auto [neighbor_links, link_count] = graph.get_neighbors(neighbor_id, layer);

                    std::vector<SearchResult> neighbor_candidates;
                    neighbor_candidates.reserve(link_count + 1);

                    const Float* neighbor_vec = &all_vectors[neighbor_id * dim];

                    for (size_t i = 0; i < link_count; ++i) {
                        if (neighbor_links[i] != INVALID_NODE) {
                            const Float* link_vec = &all_vectors[neighbor_links[i] * dim];
                            Float dot = 0;
                            for (size_t d = 0; d < dim; ++d) {
                                dot += neighbor_vec[d] * link_vec[d];
                            }
                            neighbor_candidates.push_back({neighbor_links[i], -dot});
                        }
                    }

                    Float dot_new = 0;
                    for (size_t d = 0; d < dim; ++d) {
                        dot_new += neighbor_vec[d] * new_vec[d];
                    }
                    neighbor_candidates.push_back({new_node_id, -dot_new});

                    auto new_neighbors = SelectNeighbors<ComponentT, K>::select_simple(
                        neighbor_candidates, M_limit);

                    graph.set_neighbors(neighbor_id, layer, new_neighbors);
                }
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
