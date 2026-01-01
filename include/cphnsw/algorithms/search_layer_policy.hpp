#pragma once

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../graph/priority_queue.hpp"
#include "../quantizer/quantizer_policy.hpp"
#include <vector>
#include <array>

namespace cphnsw {

/**
 * PolicySearchLayer: Quantizer-Agnostic NSW Search Algorithm
 *
 * CRITICAL FIX FOR METRIC MISMATCH:
 * This search layer uses the quantizer's distance function instead of
 * hardcoding a specific distance metric. This ensures the graph is
 * navigated using the SAME metric used during construction.
 *
 * If you build with FHT distance, search with FHT distance.
 * If you build with A-ITQ distance, search with A-ITQ distance.
 *
 * The quantizer policy provides:
 * - search_distance(query, code): Single distance computation
 * - batch_search_distance_soa(query, codes, n, out): SIMD batch computation
 *
 * Template parameters:
 * - QuantizerT: Quantizer policy type (CPFHTPolicy or AITQQuantizer)
 * - GraphT: Graph type (FlatHNSWGraph)
 */
template <typename QuantizerT, typename GraphT>
class PolicySearchLayer {
public:
    using Query = typename QuantizerT::Query;
    using Code = typename QuantizerT::Code;

    /**
     * Search using the quantizer's distance metric.
     *
     * @param query         Encoded query
     * @param entry_points  Starting nodes
     * @param ef            Search width
     * @param graph         NSW graph
     * @param quantizer     Quantizer policy (defines the distance metric)
     * @param query_id      Query ID for visited tracking
     * @return              Up to ef nearest neighbors
     */
    static std::vector<SearchResult> search(
        const Query& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        const GraphT& graph,
        const QuantizerT& quantizer,
        uint64_t query_id) {

        MaxHeap W;  // Found neighbors (max-heap, bounded to ef)
        MinHeap C;  // Candidates to explore (min-heap)

        // Initialize with entry points
        for (NodeId ep : entry_points) {
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;
            }

            const auto& ep_code = graph.get_code(ep);
            // USE QUANTIZER'S DISTANCE - This is the critical fix!
            AsymmetricDist dist = quantizer.search_distance(query, ep_code);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Greedy search
        while (!C.empty()) {
            SearchResult c = C.top();
            C.pop();

            if (W.empty()) break;
            SearchResult f = W.top();

            // Termination: all remaining candidates are worse
            if (c.distance > f.distance) {
                break;
            }

            // Get neighbor block
            const auto& block = graph.get_neighbor_block(c.id);
            // THREAD SAFETY: Use acquire semantics to see consistent data written by inserters
            size_t neighbor_count = block.count.load(std::memory_order_acquire);

            // Prefetch next candidate's block
            if (!C.empty()) {
                graph.prefetch_neighbor_block(C.top().id);
            }

            // BATCH DISTANCE COMPUTATION using quantizer's SIMD kernel
            alignas(64) std::array<AsymmetricDist, FLASH_MAX_M> batch_distances;
            quantizer.batch_search_distance_soa(
                query,
                block.codes_transposed,
                neighbor_count,
                batch_distances.data());

            // Process batch results
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = block.ids[i];
                if (neighbor_id == INVALID_NODE) continue;

                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

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

    /// Single entry point convenience wrapper
    static std::vector<SearchResult> search(
        const Query& query,
        NodeId entry_point,
        size_t ef,
        const GraphT& graph,
        const QuantizerT& quantizer,
        uint64_t query_id) {

        return search(query, std::vector<NodeId>{entry_point}, ef,
                      graph, quantizer, query_id);
    }
};

/**
 * PolicyInsert: Quantizer-Agnostic Insert Algorithm
 *
 * Uses the quantizer's distance for candidate search during construction.
 * This ensures the graph topology is optimized for the search metric.
 */
template <typename QuantizerT, typename GraphT>
class PolicyInsert {
public:
    using Query = typename QuantizerT::Query;
    using Code = typename QuantizerT::Code;

    /**
     * Insert a node using quantizer-consistent distance.
     *
     * HYBRID APPROACH:
     * - Graph navigation: Uses quantizer's distance (ensures metric alignment)
     * - Edge selection: Uses true float distance (ensures graph quality)
     *
     * @param new_node_id     ID of new node
     * @param new_query       Query encoding
     * @param new_vec         Original vector (for hybrid edge selection)
     * @param all_vectors     All vectors (for true distance computation)
     * @param dim             Vector dimension
     * @param entry_points    Search entry points
     * @param graph           NSW graph
     * @param quantizer       Quantizer policy
     * @param params          Index parameters
     * @param query_id        Query ID for visited tracking
     */
    static void insert_hybrid(
        NodeId new_node_id,
        const Query& new_query,
        const Float* new_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        const std::vector<NodeId>& entry_points,
        GraphT& graph,
        const QuantizerT& quantizer,
        const CPHNSWParams& params,
        uint64_t query_id) {

        if (graph.size() == 1) return;

        // Search for candidates using QUANTIZER'S DISTANCE
        auto candidates = PolicySearchLayer<QuantizerT, GraphT>::search(
            new_query, entry_points, params.ef_construction,
            graph, quantizer, query_id);

        // Select neighbors with TRUE distance for edge quality
        auto neighbors = select_with_true_distance(
            new_vec, all_vectors, dim, candidates, params.M);

        // Add bidirectional edges
        for (const auto& [neighbor_id, dist] : neighbors) {
            graph.add_neighbor_with_dist_safe(new_node_id, neighbor_id, dist);
            graph.add_neighbor_with_dist_safe(neighbor_id, new_node_id, dist);
        }
    }

private:
    /// Select neighbors using true cosine distance
    static std::vector<std::pair<NodeId, float>> select_with_true_distance(
        const Float* base_vec,
        const std::vector<Float>& all_vectors,
        size_t dim,
        const std::vector<SearchResult>& candidates,
        size_t M) {

        std::vector<std::pair<float, NodeId>> scored;
        scored.reserve(candidates.size());

        for (const auto& c : candidates) {
            const Float* neighbor_vec = all_vectors.data() + c.id * dim;
            Float dot = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                dot += base_vec[d] * neighbor_vec[d];
            }
            scored.emplace_back(-dot, c.id);  // Negative for distance
        }

        std::sort(scored.begin(), scored.end());

        std::vector<std::pair<NodeId, float>> result;
        result.reserve(std::min(scored.size(), M));

        // Diversity heuristic
        for (const auto& [dist, id] : scored) {
            if (result.size() >= M) break;

            bool is_diverse = true;
            const Float* cand_vec = all_vectors.data() + id * dim;

            for (const auto& [existing_id, _] : result) {
                const Float* exist_vec = all_vectors.data() + existing_id * dim;
                Float dot_to_exist = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dot_to_exist += cand_vec[d] * exist_vec[d];
                }
                Float dist_to_exist = -dot_to_exist;

                Float dist_to_base = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    dist_to_base += base_vec[d] * cand_vec[d];
                }
                dist_to_base = -dist_to_base;

                if (dist_to_exist < dist_to_base) {
                    is_diverse = false;
                    break;
                }
            }

            if (is_diverse) {
                result.emplace_back(id, dist);
            }
        }

        return result;
    }
};

}  // namespace cphnsw
