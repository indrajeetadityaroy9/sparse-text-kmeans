#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../distance/hamming.hpp"
#include <vector>
#include <algorithm>
#include <utility>

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
    using Query = CPQuery<ComponentT, K>;

private:
    // =========================================================================
    // Core Diversity Heuristic (shared implementation)
    // =========================================================================

    /**
     * Apply diversity heuristic to sorted candidates.
     *
     * Template parameters allow different distance computations and result types.
     *
     * @param candidates      Sorted candidates (ascending by distance to base)
     * @param M               Maximum neighbors to select
     * @param keep_pruned     Include pruned candidates if below M
     * @param compute_dist    Functor: (candidate_id, selected_id) -> float
     * @param emit_result     Functor: (candidate) -> void (adds to result)
     * @param result_size     Functor: () -> size_t (current result size)
     */
    template<typename DistFunc, typename EmitFunc, typename SizeFunc>
    static void apply_diversity_heuristic(
        const std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned,
        DistFunc compute_dist,
        EmitFunc emit_result,
        SizeFunc result_size) {

        std::vector<SearchResult> discarded;

        for (const auto& candidate : candidates) {
            if (result_size() >= M) break;

            bool is_good = true;

            // Check diversity against all selected neighbors
            for (size_t i = 0; i < result_size() && is_good; ++i) {
                float dist_to_selected = compute_dist(candidate, i);
                if (dist_to_selected < candidate.distance) {
                    is_good = false;
                }
            }

            if (is_good) {
                emit_result(candidate);
            } else {
                discarded.push_back(candidate);
            }
        }

        // Add pruned connections for robustness
        if (keep_pruned) {
            std::sort(discarded.begin(), discarded.end());
            for (const auto& d : discarded) {
                if (result_size() >= M) break;
                emit_result(d);
            }
        }
    }

public:
    // =========================================================================
    // select() - Diversity heuristic with CP/Hamming distance
    // =========================================================================

    /**
     * Select neighbors using the diversity heuristic with asymmetric distance.
     *
     * Uses Hamming distance for inter-candidate comparison since stored nodes
     * don't have magnitude information.
     */
    static std::vector<NodeId> select(
        const Query& /* base_query */,
        std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned,
        const Graph& graph) {

        std::sort(candidates.begin(), candidates.end());

        std::vector<NodeId> result;
        result.reserve(M);

        auto compute_dist = [&](const SearchResult& candidate, size_t selected_idx) -> float {
            const auto& candidate_code = graph.get_code(candidate.id);
            const auto& selected_code = graph.get_code(result[selected_idx]);
            return static_cast<float>(hamming_distance(candidate_code, selected_code));
        };

        auto emit = [&](const SearchResult& c) { result.push_back(c.id); };
        auto size = [&]() { return result.size(); };

        apply_diversity_heuristic(candidates, M, keep_pruned, compute_dist, emit, size);

        return result;
    }

    // =========================================================================
    // select_simple() - Top-M without diversity (for shrink operations)
    // =========================================================================

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

    // =========================================================================
    // select_with_true_distance() - Full float verification
    // =========================================================================

    /**
     * HYBRID CONSTRUCTION: Select neighbors using TRUE cosine distance.
     *
     * Recomputes ALL candidate distances using float vectors for precise
     * distance computation, ensuring correct graph topology.
     */
    static std::vector<NodeId> select_with_true_distance(
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned) {

        // Recompute all distances using true cosine
        recompute_true_distances(new_vec, all_vectors, dim, candidates);
        std::sort(candidates.begin(), candidates.end());

        std::vector<NodeId> result;
        result.reserve(M);

        auto compute_dist = [&](const SearchResult& candidate, size_t selected_idx) -> float {
            return compute_true_distance(
                all_vectors, dim, candidate.id, result[selected_idx]);
        };

        auto emit = [&](const SearchResult& c) { result.push_back(c.id); };
        auto size = [&]() { return result.size(); };

        apply_diversity_heuristic(candidates, M, keep_pruned, compute_dist, emit, size);

        return result;
    }

    // =========================================================================
    // select_with_true_distance_cached() - Optimized with cached distances
    // =========================================================================

    /**
     * Select neighbors using true distance WITH cached distances.
     *
     * OPTIMIZATION: Returns (NodeId, distance) pairs to enable O(1) pruning
     * in the graph by caching edge distances alongside links.
     *
     * BANDWIDTH OPTIMIZATION: Only verifies top 2*M candidates with float math.
     */
    static std::vector<std::pair<NodeId, float>> select_with_true_distance_cached(
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<SearchResult>& candidates,
        size_t M,
        bool keep_pruned) {

        // Bandwidth optimization: limit to top 2*M by CP distance
        std::sort(candidates.begin(), candidates.end());
        size_t verify_limit = std::min(candidates.size(), M * 2);
        if (candidates.size() > verify_limit) {
            candidates.resize(verify_limit);
        }

        // Recompute distances with prefetching
        recompute_true_distances_prefetch(new_vec, all_vectors, dim, candidates);
        std::sort(candidates.begin(), candidates.end());

        std::vector<std::pair<NodeId, float>> result;
        result.reserve(M);

        auto compute_dist = [&](const SearchResult& candidate, size_t selected_idx) -> float {
            return compute_true_distance(
                all_vectors, dim, candidate.id, result[selected_idx].first);
        };

        auto emit = [&](const SearchResult& c) {
            result.emplace_back(c.id, c.distance);
        };
        auto size = [&]() { return result.size(); };

        apply_diversity_heuristic(candidates, M, keep_pruned, compute_dist, emit, size);

        return result;
    }

private:
    // =========================================================================
    // Distance Computation Helpers
    // =========================================================================

    /**
     * Compute true cosine distance between two stored vectors.
     */
    static float compute_true_distance(
        const std::vector<Float>& all_vectors,
        size_t dim,
        NodeId id_a,
        NodeId id_b) {

        const Float* vec_a = &all_vectors[id_a * dim];
        const Float* vec_b = &all_vectors[id_b * dim];
        Float dot = 0;
        for (size_t d = 0; d < dim; ++d) {
            dot += vec_a[d] * vec_b[d];
        }
        return -dot;  // Negative for distance ordering
    }

    /**
     * Recompute all candidate distances using true cosine.
     */
    static void recompute_true_distances(
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<SearchResult>& candidates) {

        for (auto& c : candidates) {
            const Float* node_vec = &all_vectors[c.id * dim];
            Float true_dot = 0;
            for (size_t d = 0; d < dim; ++d) {
                true_dot += new_vec[d] * node_vec[d];
            }
            c.distance = -true_dot;
        }
    }

    /**
     * Recompute distances with prefetching for better cache utilization.
     */
    static void recompute_true_distances_prefetch(
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        std::vector<SearchResult>& candidates) {

        for (size_t i = 0; i < candidates.size(); ++i) {
            // Prefetch 4 steps ahead
            if (i + 4 < candidates.size()) {
                const char* prefetch_addr = reinterpret_cast<const char*>(
                    &all_vectors[candidates[i + 4].id * dim]);
                #if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(prefetch_addr, 0, 3);
                #endif
            }

            const Float* node_vec = &all_vectors[candidates[i].id * dim];
            Float true_dot = 0;
            for (size_t d = 0; d < dim; ++d) {
                true_dot += new_vec[d] * node_vec[d];
            }
            candidates[i].distance = -true_dot;
        }
    }
};

}  // namespace cphnsw
