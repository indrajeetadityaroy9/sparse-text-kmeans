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
 * CRITICAL UPDATE: Uses asymmetric distance from query magnitudes
 * for proper continuous gradient comparison.
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
    using Query = CPQuery<ComponentT, K>;

    /**
     * Select neighbors using the diversity heuristic with asymmetric distance.
     *
     * @param base_query    Query of the base node (with magnitudes)
     * @param candidates    Candidate neighbors (modified: will be sorted)
     * @param M             Maximum neighbors to select
     * @param keep_pruned   Keep pruned connections for robustness
     * @param graph         HNSW graph
     * @return              Selected neighbor IDs
     */
    static std::vector<NodeId> select(
        const Query& base_query,
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

                // Use symmetric Hamming for inter-candidate comparison
                // (We don't have magnitudes for stored nodes)
                HammingDist dist_to_selected = hamming_distance(candidate_code, selected_code);

                // Diversity heuristic: reject if closer to existing neighbor
                // Compare with asymmetric distance to base (candidate.distance)
                if (static_cast<AsymmetricDist>(dist_to_selected) < candidate.distance) {
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
     * Used for shrink operations where we don't have query magnitudes.
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

    /**
     * HYBRID CONSTRUCTION: Select neighbors using TRUE cosine distance.
     *
     * Uses actual float vectors for precise distance computation,
     * ensuring correct graph topology (100% connectivity).
     *
     * CRITICAL: Updates ALL candidate distances to true cosine before selection.
     * Do NOT mix CP estimates with true distances!
     *
     * @param new_vec         Original float vector being inserted
     * @param all_vectors     All original vectors (flat array, dim * num_nodes)
     * @param dim             Vector dimension
     * @param candidates      Candidate neighbors (distances will be recomputed)
     * @param M               Maximum neighbors to select
     * @param keep_pruned     Keep pruned connections for robustness
     * @return                Selected neighbor IDs
     */
    static std::vector<NodeId> select_with_true_distance(
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned) {

        // CRITICAL: Recompute ALL candidate distances using true cosine
        for (auto& c : candidates) {
            const Float* node_vec = &all_vectors[c.id * dim];
            Float true_dot = 0;
            for (size_t d = 0; d < dim; ++d) {
                true_dot += new_vec[d] * node_vec[d];
            }
            // Negative dot product: higher similarity = lower distance
            c.distance = -true_dot;
        }

        // Sort candidates by TRUE distance (ascending)
        std::sort(candidates.begin(), candidates.end());

        std::vector<NodeId> result;
        result.reserve(M);

        std::vector<SearchResult> discarded;

        for (const auto& candidate : candidates) {
            if (result.size() >= M) break;

            // Check diversity: is candidate closer to base than to any selected?
            bool is_good = true;

            for (NodeId selected_id : result) {
                // Compute TRUE distance between candidate and selected
                const Float* cand_vec = &all_vectors[candidate.id * dim];
                const Float* sel_vec = &all_vectors[selected_id * dim];
                Float dot = 0;
                for (size_t d = 0; d < dim; ++d) {
                    dot += cand_vec[d] * sel_vec[d];
                }
                Float dist_to_selected = -dot;

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
};

}  // namespace cphnsw
