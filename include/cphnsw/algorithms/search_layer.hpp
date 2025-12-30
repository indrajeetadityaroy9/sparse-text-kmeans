#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../graph/priority_queue.hpp"
#include "../distance/hamming.hpp"
#include <vector>

namespace cphnsw {

/**
 * SEARCH-LAYER Algorithm (Algorithm 2 from Malkov & Yashunin)
 *
 * Performs greedy search within a single layer of the HNSW graph.
 *
 * CRITICAL: Uses reconstructed dot product estimation instead of
 * discrete Hamming distance. The Cross-Polytope code tells us which
 * axis a node lies on, and we use the query's actual value at that
 * axis to estimate similarity.
 *
 * Distance = -DotProduct (negative because HNSW minimizes)
 *
 * Maintains two sets:
 * - C (MinHeap): Candidates to explore, ordered by distance ascending
 * - W (MaxHeap): Found neighbors, bounded to size ef
 *
 * Terminates when the closest candidate is worse than the furthest found neighbor.
 *
 * Complexity: O(ef * M * K) where M = avg degree, K = code width
 */
template <typename ComponentT, size_t K>
class SearchLayer {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;

    /**
     * Search a single layer using reconstructed dot product.
     *
     * Uses the query's rotated vectors and node codes to estimate
     * dot product, enabling accurate similarity-based navigation.
     *
     * @param query         Encoded query with rotated vectors
     * @param entry_points  Starting nodes for search
     * @param ef            Search width (candidates to explore)
     * @param layer         Layer level to search
     * @param graph         HNSW graph
     * @param query_id      Unique query ID for visited tracking
     * @return              Up to ef nearest neighbors found
     */
    static std::vector<SearchResult> search(
        const Query& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        LayerLevel layer,
        const Graph& graph,
        uint64_t query_id) {

        // W: found nearest neighbors (max-heap, bounded to ef)
        MaxHeap W;

        // C: candidates to explore (min-heap)
        MinHeap C;

        // Initialize with entry points
        for (NodeId ep : entry_points) {
            // Mark visited
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;  // Already visited
            }

            const auto& ep_code = graph.get_code(ep);
            // Use asymmetric search distance (returns -score for min-heap)
            AsymmetricDist dist = asymmetric_search_distance(query, ep_code);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Greedy search with dot product gradient
        while (!C.empty()) {
            // Get closest candidate (highest dot product)
            SearchResult c = C.top();
            C.pop();

            // Get furthest in W (lowest dot product among found)
            if (W.empty()) break;
            SearchResult f = W.top();

            // Termination: all remaining candidates are worse
            if (c.distance > f.distance) {
                break;
            }

            // Explore neighbors of c at this layer
            auto [neighbors, neighbor_count] = graph.get_neighbors(c.id, layer);

            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = neighbors[i];

                if (neighbor_id == INVALID_NODE) continue;

                // Prefetch next neighbor's code to hide memory latency
                // Look ahead by 2 to overlap fetch with current computation
                if (i + 2 < neighbor_count && neighbors[i + 2] != INVALID_NODE) {
                    __builtin_prefetch(&graph.get_code(neighbors[i + 2]), 0, 3);
                }

                // Check if visited (atomic)
                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                // Compute asymmetric search distance with node
                const auto& neighbor_code = graph.get_code(neighbor_id);
                AsymmetricDist dist = asymmetric_search_distance(query, neighbor_code);

                // Add to candidates if promising
                f = W.top();
                if (dist < f.distance || W.size() < ef) {
                    C.push(neighbor_id, dist);
                    W.try_push(neighbor_id, dist, ef);
                }
            }
        }

        // Extract results sorted by distance (lowest = highest similarity)
        return W.extract_sorted();
    }

    /**
     * Search with single entry point (convenience wrapper).
     */
    static std::vector<SearchResult> search(
        const Query& query,
        NodeId entry_point,
        size_t ef,
        LayerLevel layer,
        const Graph& graph,
        uint64_t query_id) {

        return search(query, std::vector<NodeId>{entry_point}, ef, layer, graph, query_id);
    }

    /**
     * Legacy search using discrete Hamming distance.
     * DEPRECATED: Use asymmetric distance variant for better recall.
     */
    static std::vector<SearchResult> search_hamming(
        const Code& query_code,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        LayerLevel layer,
        const Graph& graph,
        uint64_t query_id) {

        MaxHeap W;
        MinHeap C;

        for (NodeId ep : entry_points) {
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;
            }

            const auto& ep_code = graph.get_code(ep);
            HammingDist dist = hamming_distance(query_code, ep_code);

            C.push(ep, static_cast<AsymmetricDist>(dist));
            W.push(ep, static_cast<AsymmetricDist>(dist));
        }

        while (!C.empty()) {
            SearchResult c = C.top();
            C.pop();

            if (W.empty()) break;
            SearchResult f = W.top();

            if (c.distance > f.distance) {
                break;
            }

            auto [neighbors, neighbor_count] = graph.get_neighbors(c.id, layer);

            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = neighbors[i];
                if (neighbor_id == INVALID_NODE) continue;

                // Prefetch next neighbor's code to hide memory latency
                if (i + 2 < neighbor_count && neighbors[i + 2] != INVALID_NODE) {
                    __builtin_prefetch(&graph.get_code(neighbors[i + 2]), 0, 3);
                }

                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                const auto& neighbor_code = graph.get_code(neighbor_id);
                HammingDist dist = hamming_distance(query_code, neighbor_code);

                f = W.top();
                if (static_cast<AsymmetricDist>(dist) < f.distance || W.size() < ef) {
                    C.push(neighbor_id, static_cast<AsymmetricDist>(dist));
                    W.try_push(neighbor_id, static_cast<AsymmetricDist>(dist), ef);
                }
            }
        }

        return W.extract_sorted();
    }
};

}  // namespace cphnsw
