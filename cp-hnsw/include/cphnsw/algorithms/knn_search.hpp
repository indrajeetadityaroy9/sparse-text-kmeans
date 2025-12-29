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
 * K-NN Search Algorithm
 *
 * Finds the k nearest neighbors using the HNSW hierarchy:
 * 1. Descend from top layer to layer 1 with ef=1 (greedy)
 * 2. Search layer 0 with full ef
 * 3. Return top-k results
 *
 * Also includes multiprobe variant for improved recall.
 *
 * Complexity: O(log N + ef * M * K)
 */
template <typename ComponentT, size_t K>
class KNNSearch {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;
    using Encoder = CPEncoder<ComponentT, K>;

    /**
     * Standard K-NN search.
     *
     * @param query_code  Encoded query
     * @param k           Number of neighbors to return
     * @param ef          Search width (ef >= k recommended)
     * @param graph       HNSW graph
     * @param query_id    Starting query ID (will be incremented)
     * @return            k nearest neighbors sorted by distance
     */
    static std::vector<SearchResult> search(
        const Code& query_code,
        size_t k,
        size_t ef,
        const Graph& graph,
        uint64_t& query_id) {

        NodeId ep = graph.entry_point();
        if (ep == INVALID_NODE) {
            return {};
        }

        LayerLevel top_layer = graph.top_layer();
        std::vector<NodeId> entry_points = {ep};

        // Phase 1: Descend from top layer to layer 1 with ef=1
        for (int l = static_cast<int>(top_layer); l > 0; --l) {
            auto results = SearchLayer<ComponentT, K>::search(
                query_code, entry_points, 1, static_cast<LayerLevel>(l),
                graph, ++query_id);

            if (!results.empty()) {
                entry_points = {results[0].id};
            }
        }

        // Phase 2: Search layer 0 with full ef
        auto results = SearchLayer<ComponentT, K>::search(
            query_code, entry_points, std::max(ef, k), 0, graph, ++query_id);

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
     *
     * @param encoded      Query with sorted indices for multiprobe
     * @param k            Number of neighbors to return
     * @param ef           Search width per probe
     * @param num_probes   Number of probe sequences to try
     * @param graph        HNSW graph
     * @param query_id     Starting query ID
     * @return             k nearest neighbors
     */
    static std::vector<SearchResult> search_multiprobe(
        const typename Encoder::EncodedWithSortedIndices& encoded,
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
            auto results = search(probe.code, k, ef, graph, query_id);

            for (const auto& r : results) {
                if (seen_ids.insert(r.id).second) {
                    // Re-compute distance with primary code for fair comparison
                    HammingDist true_dist = hamming_distance(encoded.code, graph.get_code(r.id));
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
     * @param queries      Query codes
     * @param k            Number of neighbors per query
     * @param ef           Search width
     * @param graph        HNSW graph
     * @param query_id     Starting query ID
     * @return             Results for each query
     */
    static std::vector<std::vector<SearchResult>> search_batch(
        const std::vector<Code>& queries,
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
