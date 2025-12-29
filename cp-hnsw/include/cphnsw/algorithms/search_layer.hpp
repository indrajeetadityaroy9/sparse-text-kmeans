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

    /**
     * Search a single layer for nearest neighbors.
     *
     * @param query_code    Encoded query
     * @param entry_points  Starting nodes for search
     * @param ef            Search width (candidates to explore)
     * @param layer         Layer level to search
     * @param graph         HNSW graph
     * @param query_id      Unique query ID for visited tracking
     * @return              Up to ef nearest neighbors found
     */
    static std::vector<SearchResult> search(
        const Code& query_code,
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
            HammingDist dist = hamming_distance(query_code, ep_code);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Greedy search
        while (!C.empty()) {
            // Get closest candidate
            SearchResult c = C.top();
            C.pop();

            // Get furthest in W (worst current neighbor)
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

                // Check if visited (atomic)
                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                // Compute distance
                const auto& neighbor_code = graph.get_code(neighbor_id);
                HammingDist dist = hamming_distance(query_code, neighbor_code);

                // Add to candidates if promising
                f = W.top();
                if (dist < f.distance || W.size() < ef) {
                    C.push(neighbor_id, dist);
                    W.try_push(neighbor_id, dist, ef);
                }
            }
        }

        // Extract results sorted by distance
        return W.extract_sorted();
    }

    /**
     * Search with single entry point (convenience wrapper).
     */
    static std::vector<SearchResult> search(
        const Code& query_code,
        NodeId entry_point,
        size_t ef,
        LayerLevel layer,
        const Graph& graph,
        uint64_t query_id) {

        return search(query_code, std::vector<NodeId>{entry_point}, ef, layer, graph, query_id);
    }
};

}  // namespace cphnsw
