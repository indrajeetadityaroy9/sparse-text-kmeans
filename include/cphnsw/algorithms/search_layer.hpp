#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../graph/priority_queue.hpp"
#include "../distance/hamming.hpp"
#include <vector>
#include <array>

namespace cphnsw {

/**
 * NSW SEARCH-LAYER Algorithm (Simplified for single-layer graph)
 *
 * Performs greedy search within the NSW graph.
 *
 * FLASH MEMORY LAYOUT: Uses NeighborBlock for cache-local access.
 * Neighbor codes are stored IN the block, eliminating pointer chasing.
 *
 * SIMD BATCH OPTIMIZATION: Uses SoA (transposed) code layout to enable
 * true SIMD parallelism. Instead of computing distances one neighbor at
 * a time, we process 8 (AVX2) or 16 (AVX-512) neighbors simultaneously:
 *
 *   Scalar: for each neighbor: for each K: gather + accumulate
 *   Batch:  for each K: SIMD load 8/16 bytes → gather 8/16 values → accumulate
 *
 * This achieves 3-5x speedup on distance computation.
 *
 * CRITICAL: Uses reconstructed dot product estimation instead of
 * discrete Hamming distance. The Cross-Polytope code tells us which
 * axis a node lies on, and we use the query's actual value at that
 * axis to estimate similarity.
 *
 * Distance = -DotProduct (negative because we minimize)
 *
 * Maintains two sets:
 * - C (MinHeap): Candidates to explore, ordered by distance ascending
 * - W (MaxHeap): Found neighbors, bounded to size ef
 *
 * Termination: when the closest candidate is worse than the furthest found neighbor.
 *
 * Complexity: O(ef * M * K) where M = avg degree, K = code width
 */
template <typename ComponentT, size_t K>
class SearchLayer {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Block = NeighborBlock<ComponentT, K>;

    /**
     * NSW + Flash Search using reconstructed dot product.
     *
     * FLASH OPTIMIZATION: Uses neighbor codes stored in NeighborBlock
     * for cache-local access. No pointer chasing to separate codes array.
     *
     * @param query         Encoded query with rotated vectors
     * @param entry_points  Starting nodes for search
     * @param ef            Search width (candidates to explore)
     * @param graph         NSW graph with Flash layout
     * @param query_id      Unique query ID for visited tracking
     * @return              Up to ef nearest neighbors found
     */
    static std::vector<SearchResult> search(
        const Query& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
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

            // FLASH: Get neighbor block (contains IDs + codes + distances)
            const Block& block = graph.get_neighbor_block(c.id);
            // THREAD SAFETY: Use acquire semantics to see consistent data written by inserters
            size_t neighbor_count = block.count.load(std::memory_order_acquire);

            // Prefetch next candidate's block while processing current
            if (!C.empty()) {
                graph.prefetch_neighbor_block(C.top().id);
            }

            // SIMD BATCH OPTIMIZATION: Compute ALL neighbor distances at once
            // using the SoA (transposed) layout. This enables true SIMD
            // parallelism by processing 8 (AVX2) or 16 (AVX-512) neighbors
            // simultaneously.
            //
            // Trade-off: We compute distances for all neighbors, including
            // already-visited ones. This wastes ~10-20% of distance computations
            // but keeps the SIMD pipeline fully utilized. Net gain: 2-4x.
            alignas(64) std::array<AsymmetricDist, FLASH_MAX_M> batch_distances;
            asymmetric_search_distance_batch_soa<ComponentT, K>(
                query, block.codes_transposed, neighbor_count,
                batch_distances.data());

            // Process batch results: filter visited and update heaps
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = block.ids[i];

                if (neighbor_id == INVALID_NODE) continue;

                // Check if visited (atomic)
                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                // Use pre-computed batch distance
                AsymmetricDist dist = batch_distances[i];

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
        const Graph& graph,
        uint64_t query_id) {

        return search(query, std::vector<NodeId>{entry_point}, ef, graph, query_id);
    }
};

}  // namespace cphnsw
