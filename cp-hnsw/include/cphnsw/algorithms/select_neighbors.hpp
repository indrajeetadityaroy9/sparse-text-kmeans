#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../distance/hamming.hpp"
#include <vector>
#include <algorithm>

namespace cphnsw {

/**
 * SELECT-NEIGHBORS-HEURISTIC (Algorithm 4 from Malkov & Yashunin)
 *
 * Selects neighbors to connect, ensuring spatial diversity.
 * A candidate is accepted only if it's closer to the base than
 * to any already-selected neighbor.
 *
 * This heuristic is critical for graph connectivity - it prevents
 * clustering of neighbors and ensures the graph can navigate
 * across different regions of the space.
 */
template <typename ComponentT, size_t K>
class SelectNeighbors {
public:
    using Graph = FlatHNSWGraph<ComponentT, K>;
    using Code = CPCode<ComponentT, K>;

    /**
     * Select neighbors using the diversity heuristic.
     *
     * @param base_code     Code of the base node
     * @param candidates    Candidate neighbors (modified: will be sorted)
     * @param M             Maximum neighbors to select
     * @param keep_pruned   Keep pruned connections for robustness
     * @param graph         HNSW graph
     * @return              Selected neighbor IDs
     */
    static std::vector<NodeId> select(
        const Code& base_code,
        std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned,
        const Graph& graph) {

        // Sort candidates by distance to base (ascending)
        std::sort(candidates.begin(), candidates.end());

        std::vector<NodeId> result;
        result.reserve(M);

        std::vector<SearchResult> discarded;

        for (const auto& candidate : candidates) {
            if (result.size() >= M) break;

            // Check diversity: is candidate closer to base than to any selected?
            bool is_good = true;

            for (NodeId selected_id : result) {
                const auto& selected_code = graph.get_code(selected_id);
                const auto& candidate_code = graph.get_code(candidate.id);

                HammingDist dist_to_selected = hamming_distance(candidate_code, selected_code);

                // Diversity heuristic: reject if closer to existing neighbor
                if (dist_to_selected < candidate.distance) {
                    is_good = false;
                    break;
                }
            }

            if (is_good) {
                result.push_back(candidate.id);
            } else {
                discarded.push_back(candidate);
            }
        }

        // Optionally add pruned connections for robustness
        if (keep_pruned && result.size() < M) {
            std::sort(discarded.begin(), discarded.end());
            for (const auto& d : discarded) {
                if (result.size() >= M) break;
                result.push_back(d.id);
            }
        }

        return result;
    }

    /**
     * Simple selection without diversity heuristic (just top-M by distance).
     * Faster but may produce less connected graphs.
     */
    static std::vector<NodeId> select_simple(
        std::vector<SearchResult>& candidates,
        size_t M) {

        std::sort(candidates.begin(), candidates.end());

        std::vector<NodeId> result;
        result.reserve(std::min(candidates.size(), M));

        for (size_t i = 0; i < std::min(candidates.size(), M); ++i) {
            result.push_back(candidates[i].id);
        }

        return result;
    }
};

}  // namespace cphnsw
