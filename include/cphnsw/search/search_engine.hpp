#pragma once

#include "../core/codes.hpp"
#include "../distance/metric_policy.hpp"
#include "../graph/flat_graph.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <array>

namespace cphnsw {

// ============================================================================
// Search Result
// ============================================================================

struct SearchResult {
    NodeId id;
    DistanceType distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }

    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

// ============================================================================
// Priority Queues for Search
// ============================================================================

// Min-heap: candidates ordered by distance (closest first)
using MinHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::greater<SearchResult>>;

// Max-heap: results ordered by distance (furthest first for bounded set)
using MaxHeap = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::less<SearchResult>>;

// ============================================================================
// SearchEngine: Unified greedy graph search with policy-based metrics
// ============================================================================

/**
 * SearchEngine: Generic NSW search engine using policy-based design.
 *
 * UNIFIED DESIGN: A single search implementation that works with any
 * MetricPolicy. The policy handles all distance computation, allowing
 * the search logic to remain unchanged across Phase 1 (R=0) and Phase 2 (R>0).
 *
 * ALGORITHM: Standard NSW greedy beam search:
 *   1. Initialize candidates C and results W with entry points
 *   2. While C is not empty:
 *      a. Pop closest candidate c
 *      b. If c is worse than worst in W, terminate (convergence)
 *      c. For each neighbor n of c:
 *         - Compute distance using MetricPolicy
 *         - If promising, add to C and W
 *   3. Return top-k from W
 *
 * SIMD OPTIMIZATION: Uses batch distance computation via MetricPolicy
 * for all neighbors in a block simultaneously.
 *
 * @tparam Policy The MetricPolicy type (e.g., UnifiedMetricPolicy<K, R, Shift>)
 */
template <typename Policy>
class SearchEngine {
public:
    using CodeType = typename Policy::CodeType;
    using QueryType = typename Policy::QueryType;
    using SoALayoutType = typename Policy::SoALayoutType;
    using Graph = FlatGraph<CodeType>;

    /**
     * Perform k-NN search on the graph.
     *
     * @param query Pre-computed query structure with binary codes and scalars
     * @param entry_points Initial nodes to start search from
     * @param ef Search width (number of candidates to explore)
     * @param graph The NSW graph to search
     * @param k Number of results to return (default: ef)
     * @return Vector of (id, distance) pairs, sorted by distance ascending
     */
    static std::vector<SearchResult> search(
        const QueryType& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        const Graph& graph,
        size_t k = 0) {

        if (k == 0) k = ef;

        // W: found nearest neighbors (max-heap for bounded size)
        MaxHeap W;

        // C: candidates to explore (min-heap for closest first)
        MinHeap C;

        // Get query epoch for visited tracking
        uint64_t query_id = graph.new_query();

        // Initialize with entry points
        for (NodeId ep : entry_points) {
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;  // Already visited
            }

            const auto& ep_code = graph.get_code(ep);
            DistanceType dist = Policy::compute_distance(query, ep_code);

            C.push({ep, dist});
            W.push({ep, dist});
        }

        // Temporary storage for batch distances
        alignas(64) std::array<DistanceType, MAX_NEIGHBORS> batch_distances;

        // Greedy search
        while (!C.empty()) {
            // Get closest candidate
            SearchResult c = C.top();
            C.pop();

            // Check termination condition
            if (W.empty()) break;
            SearchResult worst = W.top();

            if (c.distance > worst.distance) {
                break;  // All remaining candidates are worse
            }

            // Get neighbor block
            const auto& block = graph.get_neighbor_block(c.id);
            size_t neighbor_count = block.size();

            if (neighbor_count == 0) continue;

            // Prefetch next candidate's block while processing current
            if (!C.empty()) {
                graph.prefetch_neighbor_block(C.top().id);
            }

            // SIMD BATCH OPTIMIZATION: Compute all neighbor distances at once
            Policy::compute_distance_batch(
                query,
                block.get_soa_layout(),
                neighbor_count,
                batch_distances.data());

            // Process batch results
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = block.get_id(i);
                if (neighbor_id == INVALID_NODE) continue;

                // Check visited (atomic)
                if (graph.check_and_mark_visited(neighbor_id, query_id)) {
                    continue;
                }

                DistanceType dist = batch_distances[i];

                // Add to candidates if promising
                worst = W.top();
                if (dist < worst.distance || W.size() < ef) {
                    C.push({neighbor_id, dist});

                    // Bounded insert into W
                    if (W.size() < ef) {
                        W.push({neighbor_id, dist});
                    } else if (dist < W.top().distance) {
                        W.pop();
                        W.push({neighbor_id, dist});
                    }
                }
            }
        }

        // Extract top-k results sorted by distance
        std::vector<SearchResult> results;
        results.reserve(std::min(k, W.size()));

        while (!W.empty() && results.size() < k) {
            results.push_back(W.top());
            W.pop();
        }

        // Reverse to get ascending order (closest first)
        std::reverse(results.begin(), results.end());

        return results;
    }

    /**
     * Search with single entry point.
     */
    static std::vector<SearchResult> search(
        const QueryType& query,
        NodeId entry_point,
        size_t ef,
        const Graph& graph,
        size_t k = 0) {

        return search(query, std::vector<NodeId>{entry_point}, ef, graph, k);
    }

    /**
     * Search using graph's default entry point.
     */
    static std::vector<SearchResult> search(
        const QueryType& query,
        size_t ef,
        const Graph& graph,
        size_t k = 0) {

        NodeId ep = graph.entry_point();
        if (ep == INVALID_NODE) {
            return {};
        }
        return search(query, ep, ef, graph, k);
    }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1 search engines (no residual)
using SearchEngine32 = SearchEngine<UnifiedMetricPolicy<32, 0, 0>>;
using SearchEngine64 = SearchEngine<UnifiedMetricPolicy<64, 0, 0>>;

// Phase 2 search engines (with residual)
using SearchEngine64_32 = SearchEngine<UnifiedMetricPolicy<64, 32, 2>>;
using SearchEngine32_16 = SearchEngine<UnifiedMetricPolicy<32, 16, 2>>;

// ============================================================================
// SearchEngineTraits: Compile-time information
// ============================================================================

template <typename Engine>
struct SearchEngineTraits {
    using Policy = typename Engine::Policy;
    using CodeType = typename Engine::CodeType;
    using QueryType = typename Engine::QueryType;
    using Graph = typename Engine::Graph;
};

}  // namespace cphnsw
