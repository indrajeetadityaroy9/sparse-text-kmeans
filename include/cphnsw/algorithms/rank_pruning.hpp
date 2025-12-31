#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace cphnsw {

/**
 * Rank-Based Pruning for NSW Graph Construction (CAGRA-style).
 *
 * Given a k-NN graph, prunes edges to create a navigable NSW graph
 * with degree ≤ M. The pruning criterion:
 *
 *   Edge (u,v) is "detourable" if there exists a selected neighbor s where:
 *       D(u,s) + D(s,v) ≤ α × D(u,v)
 *
 * This creates a graph that balances local connectivity with navigability.
 *
 * Key parameters:
 * - α (alpha): Detour threshold (default 1.1)
 *   - α = 1.0-1.05: Aggressive pruning, forces long-range connections
 *   - α = 1.1: Balanced default
 *   - α = 1.2+: Less aggressive, keeps more local edges
 *
 * Reference: CAGRA (NeurIPS 2023)
 */
class RankBasedPruning {
public:
    /**
     * Prune k-NN neighbors for a single node.
     *
     * Given candidate neighbors sorted by distance, selects up to M neighbors
     * using rank-based pruning.
     *
     * @param candidates     Candidate neighbors sorted by distance (ascending)
     * @param distances      Distances to candidates (same order)
     * @param num_candidates Number of candidates
     * @param M              Maximum degree
     * @param alpha          Detour threshold (default 1.1)
     * @param distance_func  Function to compute D(s,v) given neighbor indices
     * @return               Selected neighbor indices (up to M)
     */
    template <typename DistanceFunc>
    static std::vector<NodeId> prune_neighbors(
        const NodeId* candidates,
        const float* distances,
        size_t num_candidates,
        size_t M,
        float alpha,
        DistanceFunc distance_func) {

        std::vector<NodeId> selected;
        selected.reserve(M);

        std::vector<bool> pruned(num_candidates, false);

        for (size_t i = 0; i < num_candidates && selected.size() < M; ++i) {
            if (pruned[i]) continue;

            NodeId v = candidates[i];
            float dist_u_v = distances[i];

            // Check if any selected neighbor makes this edge detourable
            bool is_detourable = false;
            for (NodeId s : selected) {
                float dist_u_s = 0.0f;  // Already known: s was selected before v
                // Find dist_u_s from earlier in candidates
                for (size_t j = 0; j < i; ++j) {
                    if (candidates[j] == s) {
                        dist_u_s = distances[j];
                        break;
                    }
                }

                float dist_s_v = distance_func(s, v);

                if (dist_u_s + dist_s_v <= alpha * dist_u_v) {
                    is_detourable = true;
                    break;
                }
            }

            if (!is_detourable) {
                selected.push_back(v);

                // Mark candidates that this selection makes detourable
                for (size_t j = i + 1; j < num_candidates; ++j) {
                    if (pruned[j]) continue;

                    NodeId w = candidates[j];
                    float dist_u_w = distances[j];
                    float dist_v_w = distance_func(v, w);

                    if (dist_u_v + dist_v_w <= alpha * dist_u_w) {
                        pruned[j] = true;
                    }
                }
            }
        }

        return selected;
    }

    /**
     * Prune k-NN graph to NSW graph (batch version).
     *
     * Processes all nodes and creates a pruned graph with degree ≤ M.
     *
     * @param knn_neighbors  k-NN neighbor indices [N x k]
     * @param knn_distances  k-NN distances [N x k]
     * @param N              Number of nodes
     * @param k              Neighbors per node in k-NN
     * @param M              Target maximum degree
     * @param vectors        Original vectors [N x dim] for distance computation
     * @param dim            Vector dimension
     * @param alpha          Detour threshold
     * @return               Pruned neighbor lists [N x M] (INVALID_NODE for empty slots)
     */
    static std::vector<std::vector<NodeId>> prune_knn_graph(
        const uint32_t* knn_neighbors,
        const float* knn_distances,
        size_t N,
        size_t k,
        size_t M,
        const Float* vectors,
        size_t dim,
        float alpha = 1.1f) {

        std::vector<std::vector<NodeId>> pruned_graph(N);

        // Distance computation lambda
        // Uses 1 - dot for proper metric (compatible with GPU k-NN output)
        auto compute_distance = [&](NodeId a, NodeId b) -> float {
            const Float* va = vectors + a * dim;
            const Float* vb = vectors + b * dim;
            float dot = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                dot += va[d] * vb[d];
            }
            return 1.0f - dot;  // Cosine distance: 1 - similarity
        };

        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < N; ++i) {
            const uint32_t* neighbors = knn_neighbors + i * k;
            const float* distances = knn_distances + i * k;

            // Convert to NodeId (filter invalid)
            std::vector<NodeId> valid_neighbors;
            std::vector<float> valid_distances;
            valid_neighbors.reserve(k);
            valid_distances.reserve(k);

            for (size_t j = 0; j < k; ++j) {
                if (neighbors[j] != UINT32_MAX && neighbors[j] != i) {
                    valid_neighbors.push_back(static_cast<NodeId>(neighbors[j]));
                    valid_distances.push_back(distances[j]);
                }
            }

            if (valid_neighbors.empty()) {
                pruned_graph[i] = {};
                continue;
            }

            // Prune
            pruned_graph[i] = prune_neighbors(
                valid_neighbors.data(),
                valid_distances.data(),
                valid_neighbors.size(),
                M,
                alpha,
                compute_distance);
        }

        return pruned_graph;
    }

    /**
     * Make graph bidirectional by adding reverse edges.
     *
     * For NSW, we want the graph to be undirected (or at least have
     * good reverse connectivity). This adds reverse edges where missing.
     *
     * @param graph   Pruned neighbor lists [N x variable]
     * @param M       Maximum degree (soft limit after reverse edges)
     */
    static void add_reverse_edges(std::vector<std::vector<NodeId>>& graph, size_t M) {
        size_t N = graph.size();

        // Collect reverse edges to add
        std::vector<std::vector<NodeId>> reverse_edges(N);

        for (size_t u = 0; u < N; ++u) {
            for (NodeId v : graph[u]) {
                // Check if reverse edge exists
                auto& neighbors_v = graph[v];
                bool has_reverse = std::find(neighbors_v.begin(), neighbors_v.end(),
                                              static_cast<NodeId>(u)) != neighbors_v.end();
                if (!has_reverse) {
                    reverse_edges[v].push_back(static_cast<NodeId>(u));
                }
            }
        }

        // Add reverse edges (respecting M limit)
        for (size_t v = 0; v < N; ++v) {
            for (NodeId u : reverse_edges[v]) {
                if (graph[v].size() < M) {
                    graph[v].push_back(u);
                }
            }
        }
    }

    /**
     * Verify graph connectivity using BFS.
     *
     * @param graph  Neighbor lists
     * @return       Number of nodes reachable from node 0
     */
    static size_t verify_connectivity(const std::vector<std::vector<NodeId>>& graph) {
        if (graph.empty()) return 0;

        size_t N = graph.size();
        std::vector<bool> visited(N, false);
        std::vector<NodeId> queue;
        queue.reserve(N);

        visited[0] = true;
        queue.push_back(0);
        size_t head = 0;

        while (head < queue.size()) {
            NodeId u = queue[head++];
            for (NodeId v : graph[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    queue.push_back(v);
                }
            }
        }

        return queue.size();
    }

    /**
     * Repair disconnected components by adding random edges.
     *
     * Finds disconnected nodes and connects them to random nodes
     * in the main component.
     *
     * @param graph  Neighbor lists (modified in place)
     * @param seed   Random seed
     */
    static void repair_connectivity(std::vector<std::vector<NodeId>>& graph,
                                     uint64_t seed = 42) {
        if (graph.empty()) return;

        size_t N = graph.size();
        std::vector<bool> visited(N, false);
        std::vector<NodeId> main_component;
        main_component.reserve(N);

        // BFS from node 0
        visited[0] = true;
        main_component.push_back(0);
        size_t head = 0;

        while (head < main_component.size()) {
            NodeId u = main_component[head++];
            for (NodeId v : graph[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    main_component.push_back(v);
                }
            }
        }

        if (main_component.size() == N) return;  // Already connected

        // Find disconnected nodes
        std::vector<NodeId> disconnected;
        for (size_t i = 0; i < N; ++i) {
            if (!visited[i]) {
                disconnected.push_back(static_cast<NodeId>(i));
            }
        }

        // Connect each disconnected node to a random node in main component
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, main_component.size() - 1);

        for (NodeId u : disconnected) {
            NodeId v = main_component[dist(rng)];

            // Add bidirectional edge
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
    }
};

}  // namespace cphnsw
