#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../graph/priority_queue.hpp"
#include "../distance/hamming.hpp"
#include "finger_calibration.hpp"
#include <array>

namespace cphnsw {

// ============================================================================
// Calibrated Distance Functions
// ============================================================================

/**
 * Compute calibrated asymmetric search distance.
 *
 * Applies FINGER linear calibration to the raw asymmetric distance,
 * providing a better approximation of true cosine distance.
 *
 * @param query  Query with rotated vectors
 * @param code   Node code
 * @param calib  FINGER calibration parameters
 * @return       Calibrated distance
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist calibrated_search_distance(
    const CPQuery<ComponentT, K>& query,
    const CPCode<ComponentT, K>& code,
    const FINGERCalibration& calib) {

    // Compute raw asymmetric distance
    AsymmetricDist raw_dist = asymmetric_search_distance(query, code);

    // Apply calibration
    return calib.apply(raw_dist);
}

/**
 * Compute calibrated distance from raw code pointer (Flash layout).
 *
 * @param query  Query with rotated vectors
 * @param code   Pointer to K components of neighbor code
 * @param calib  FINGER calibration parameters
 * @return       Calibrated distance
 */
template <typename ComponentT, size_t K>
inline AsymmetricDist calibrated_search_distance_ptr(
    const CPQuery<ComponentT, K>& query,
    const ComponentT* code,
    const FINGERCalibration& calib) {

    // Compute raw asymmetric distance
    AsymmetricDist raw_dist = asymmetric_search_distance_ptr<ComponentT, K>(query, code);

    // Apply calibration
    return calib.apply(raw_dist);
}

/**
 * Batch compute calibrated distances for Flash layout (legacy AoS version).
 *
 * @param query           Query with rotated vectors
 * @param neighbor_codes  Contiguous neighbor codes from NeighborBlock
 * @param num_neighbors   Number of neighbors
 * @param calib           FINGER calibration parameters
 * @param out_distances   Output array (must have num_neighbors elements)
 */
template <typename ComponentT, size_t K>
inline void calibrated_search_distance_batch(
    const CPQuery<ComponentT, K>& query,
    const ComponentT* neighbor_codes,
    size_t num_neighbors,
    const FINGERCalibration& calib,
    AsymmetricDist* out_distances) {

    // Compute raw asymmetric distances
    asymmetric_search_distance_batch<ComponentT, K>(
        query, neighbor_codes, num_neighbors, out_distances);

    // Apply calibration to all
    for (size_t i = 0; i < num_neighbors; ++i) {
        out_distances[i] = calib.apply(out_distances[i]);
    }
}

/**
 * Batch compute calibrated distances using SoA (transposed) layout.
 *
 * SIMD OPTIMIZED: Uses the transposed code layout for true SIMD parallelism.
 * Processes 8 (AVX2) or 16 (AVX-512) neighbors simultaneously.
 *
 * @param query              Query with rotated vectors
 * @param codes_transposed   Transposed codes [K][FLASH_MAX_M]
 * @param num_neighbors      Number of neighbors
 * @param calib              FINGER calibration parameters
 * @param out_distances      Output array (must have num_neighbors elements)
 */
template <typename ComponentT, size_t K>
inline void calibrated_search_distance_batch_soa(
    const CPQuery<ComponentT, K>& query,
    const ComponentT codes_transposed[K][64],
    size_t num_neighbors,
    const FINGERCalibration& calib,
    AsymmetricDist* out_distances) {

    // Compute raw asymmetric distances using SIMD batch kernel
    asymmetric_search_distance_batch_soa<ComponentT, K>(
        query, codes_transposed, num_neighbors, out_distances);

    // Apply calibration to all (vectorizable scalar loop)
    for (size_t i = 0; i < num_neighbors; ++i) {
        out_distances[i] = calib.apply(out_distances[i]);
    }
}

// ============================================================================
// Calibration-Aware Search Layer
// ============================================================================

/**
 * CalibratedSearchLayer: Search with FINGER calibration.
 *
 * Wraps the standard search layer to apply calibration during distance
 * computation. Use this for improved recall after calibration.
 */
template <typename ComponentT, size_t K>
class CalibratedSearchLayer {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Block = NeighborBlock<ComponentT, K>;

    /**
     * Calibrated NSW Search.
     *
     * Same algorithm as SearchLayer but applies FINGER calibration
     * to all distance computations for better ranking.
     *
     * @param query         Encoded query with rotated vectors
     * @param entry_points  Starting nodes for search
     * @param ef            Search width
     * @param graph         NSW graph with Flash layout
     * @param calib         FINGER calibration parameters
     * @param query_id      Unique query ID for visited tracking
     * @return              Up to ef nearest neighbors
     */
    static std::vector<SearchResult> search(
        const Query& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        const Graph& graph,
        const FINGERCalibration& calib,
        uint64_t query_id) {

        // Import priority queues from graph module
        MaxHeap W;
        MinHeap C;

        // Initialize with entry points
        for (NodeId ep : entry_points) {
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;
            }

            const auto& ep_code = graph.get_code(ep);
            AsymmetricDist dist = calibrated_search_distance(query, ep_code, calib);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Greedy search with calibrated distances
        while (!C.empty()) {
            SearchResult c = C.top();
            C.pop();

            if (W.empty()) break;
            SearchResult f = W.top();

            if (c.distance > f.distance) {
                break;
            }

            // Flash: Get neighbor block
            const Block& block = graph.get_neighbor_block(c.id);
            size_t neighbor_count = block.count;

            // Prefetch next candidate's block
            if (!C.empty()) {
                graph.prefetch_neighbor_block(C.top().id);
            }

            // SIMD BATCH OPTIMIZATION: Compute ALL calibrated neighbor distances
            // at once using the SoA (transposed) layout.
            alignas(64) std::array<AsymmetricDist, FLASH_MAX_M> batch_distances;
            calibrated_search_distance_batch_soa<ComponentT, K>(
                query, block.codes_transposed, neighbor_count, calib,
                batch_distances.data());

            // Process batch results
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = block.ids[i];

                if (neighbor_id == INVALID_NODE) continue;

                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                // Use pre-computed batch distance
                AsymmetricDist dist = batch_distances[i];

                f = W.top();
                if (dist < f.distance || W.size() < ef) {
                    C.push(neighbor_id, dist);
                    W.try_push(neighbor_id, dist, ef);
                }
            }
        }

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
        const FINGERCalibration& calib,
        uint64_t query_id) {

        return search(query, std::vector<NodeId>{entry_point}, ef, graph, calib, query_id);
    }
};

}  // namespace cphnsw
