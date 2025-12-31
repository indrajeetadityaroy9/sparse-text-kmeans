#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../quantizer/multiprobe.hpp"
#include "search_layer.hpp"
#include <vector>
#include <algorithm>
#include <unordered_set>

namespace cphnsw {

/**
 * NSW K-NN Search Algorithm (Simplified for single-layer graph)
 *
 * Finds the k nearest neighbors using random entry points:
 * 1. Get k_entry random entry points
 * 2. Search with full ef
 * 3. Return top-k results
 *
 * Per "Down with the Hierarchy" - random entry points work equivalently
 * for high-dimensional data due to hub highways.
 *
 * CRITICAL: Uses asymmetric distance (weighted by query magnitudes)
 * to enable proper gradient descent in quantized space.
 *
 * Complexity: O(ef * M * K)
 */
template <typename ComponentT, size_t K>
class KNNSearch {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Query = CPQuery<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

    /**
     * NSW K-NN search using asymmetric distance.
     *
     * Uses query magnitudes for continuous gradient navigation.
     *
     * @param query       Encoded query with magnitudes
     * @param k           Number of neighbors to return
     * @param ef          Search width (ef >= k recommended)
     * @param graph       NSW graph
     * @param query_id    Starting query ID (will be incremented)
     * @return            k nearest neighbors sorted by distance
     */
    static std::vector<SearchResult> search(
        const Query& query,
        size_t k,
        size_t ef,
        const Graph& graph,
        uint64_t& query_id) {

        if (graph.empty()) {
            return {};
        }

        // NSW: Get random entry points
        std::vector<NodeId> entry_points = graph.get_random_entry_points(
            graph.params().k_entry, query_id);

        // Single-layer search with full ef
        auto results = SearchLayer<ComponentT, K>::search(
            query, entry_points, std::max(ef, k), graph, ++query_id);

        // Return top-k
        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

    /**
     * Multiprobe K-NN search for improved recall.
     *
     * Generates multiple probe codes and unions results.
     * Uses asymmetric distance for consistent ranking.
     *
     * @param encoded      Query with sorted indices for multiprobe
     * @param query        Query with magnitudes for asymmetric distance
     * @param k            Number of neighbors to return
     * @param ef           Search width per probe
     * @param num_probes   Number of probe sequences to try
     * @param graph        NSW graph
     * @param query_id     Starting query ID
     * @return             k nearest neighbors
     */
    static std::vector<SearchResult> search_multiprobe(
        const typename Encoder::EncodedWithSortedIndices& encoded,
        const Query& query,
        size_t k,
        size_t ef,
        size_t num_probes,
        const Graph& graph,
        uint64_t& query_id) {

        // Generate probe sequence
        MultiprobeGenerator<ComponentT, K> mpg;
        auto probes = mpg.generate(encoded, num_probes);

        // Collect results from all probes
        std::unordered_set<NodeId> seen_ids;
        std::vector<SearchResult> all_results;
        all_results.reserve(k * num_probes);

        for (const auto& probe : probes) {
            // Create a query struct for this probe
            Query probe_query;
            probe_query.primary_code = probe.code;
            probe_query.magnitudes = query.magnitudes;  // Reuse magnitudes
            probe_query.original_indices = query.original_indices;
            probe_query.rotated_vecs = query.rotated_vecs;  // CRITICAL: Copy rotated vectors

            auto results = search(probe_query, k, ef, graph, query_id);

            for (const auto& r : results) {
                if (seen_ids.insert(r.id).second) {
                    // Re-compute distance with primary query for fair comparison
                    AsymmetricDist true_dist = asymmetric_search_distance(query, graph.get_code(r.id));
                    all_results.push_back({r.id, true_dist});
                }
            }
        }

        // Sort by true distance
        std::sort(all_results.begin(), all_results.end());

        // Return top-k
        if (all_results.size() > k) {
            all_results.resize(k);
        }

        return all_results;
    }

    /**
     * Batch K-NN search for multiple queries.
     *
     * @param queries      Query structs with magnitudes
     * @param k            Number of neighbors per query
     * @param ef           Search width
     * @param graph        NSW graph
     * @param query_id     Starting query ID
     * @return             Results for each query
     */
    static std::vector<std::vector<SearchResult>> search_batch(
        const std::vector<Query>& queries,
        size_t k,
        size_t ef,
        const Graph& graph,
        uint64_t& query_id) {

        std::vector<std::vector<SearchResult>> results;
        results.reserve(queries.size());

        for (const auto& query : queries) {
            results.push_back(search(query, k, ef, graph, query_id));
        }

        return results;
    }
};

}  // namespace cphnsw
