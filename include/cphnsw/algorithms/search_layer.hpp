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
            size_t neighbor_count = block.count;

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

// ============================================================================
// RaBitQ Search Layer (Phase 1 Optimization)
// ============================================================================

/**
 * RaBitQSearchLayer: NSW search using XOR + PopCount distance.
 *
 * PHASE 1 OPTIMIZATION (PhD Portfolio):
 * Replaces expensive SIMD Gather with pure bitwise operations:
 * - Old: _mm256_i32gather_ps = 10-11 cycles per instruction
 * - New: XOR + PopCount = 1-2 cycles total
 *
 * Distance formula (for normalized vectors):
 *   Dist = C1 + C2 * Hamming(query, node)
 *
 * CRITICAL: C1, C2 are pre-computed ONCE per query - no per-neighbor lookups!
 *
 * Expected speedup: ~3x on distance computation (100M/s → 300M/s)
 *
 * Trade-off: Uses symmetric Hamming instead of asymmetric dot product.
 * This causes recall drop (~10-15%) which Phase 2 (residual) will recover.
 */
template <size_t K>
class RaBitQSearchLayer {
public:
    using Graph = FlatHNSWGraph<uint8_t, K>;  // RaBitQ only supports uint8_t
    using Query = RaBitQQuery<K>;
    using Block = NeighborBlock<uint8_t, K>;

    /**
     * NSW Search using RaBitQ XOR + PopCount distance.
     *
     * @param query         RaBitQ query with pre-computed C1, C2 scalars
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
            if (graph.check_and_mark_visited(ep, query_id)) {
                continue;
            }

            // Compute RaBitQ distance to entry point
            const auto& ep_block = graph.get_neighbor_block(ep);
            // Get this node's own binary signs from graph
            BinaryCode<K> ep_signs = get_node_binary_signs(graph, ep);
            AsymmetricDist dist = rabitq_distance_scalar(query, ep_signs);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Pre-allocate batch distance buffer
        alignas(64) std::array<AsymmetricDist, FLASH_MAX_M> batch_distances;

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
            const Block& block = graph.get_neighbor_block(c.id);
            size_t neighbor_count = block.count;

            // Prefetch next candidate's block
            if (!C.empty()) {
                graph.prefetch_neighbor_block(C.top().id);
            }

            // RABITQ BATCH: Compute all neighbor distances using XOR + PopCount
            rabitq_hamming_block(query, block, neighbor_count, batch_distances.data());

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

private:
    /**
     * Get binary signs for a node from the graph.
     *
     * IMPLEMENTATION NOTE: The graph stores codes for each node.
     * We need to extract binary signs from the CP code.
     * For now, use a simple extraction from the node's code.
     */
    static BinaryCode<K> get_node_binary_signs(const Graph& graph, NodeId node_id) {
        BinaryCode<K> result;
        result.clear();

        const auto& code = graph.get_code(node_id);
        for (size_t r = 0; r < K; ++r) {
            bool is_negative = CPCode<uint8_t, K>::decode_sign_negative(code.components[r]);
            result.set_sign(r, is_negative);
        }

        return result;
    }

    /**
     * Compute RaBitQ distances for all neighbors in a block.
     *
     * Dispatches to AVX-512, AVX2, or scalar based on availability.
     */
    static void rabitq_hamming_block(
        const Query& query,
        const Block& block,
        size_t neighbor_count,
        AsymmetricDist* out_distances) {

#if CPHNSW_HAS_AVX512
        // Use AVX-512 batch kernel
        rabitq_hamming_block_avx512<K>(
            query,
            block.get_signs_transposed(),
            neighbor_count,
            out_distances);
#elif CPHNSW_HAS_AVX2
        // AVX2 fallback - process 4 at a time
        alignas(32) uint32_t batch_hamming[4];

        size_t n = 0;
        for (; n + 4 <= neighbor_count; n += 4) {
            rabitq_hamming_batch4_avx2<K>(
                query.binary.signs,
                block.get_signs_transposed(),
                n,
                batch_hamming);

            for (size_t i = 0; i < 4; ++i) {
                out_distances[n + i] =
                    query.c1 + query.c2 * static_cast<float>(batch_hamming[i]);
            }
        }

        // Scalar remainder
        for (; n < neighbor_count; ++n) {
            BinaryCode<K> neighbor_signs = block.get_neighbor_binary_signs(n);
            out_distances[n] = rabitq_distance_scalar(query, neighbor_signs);
        }
#else
        // Scalar fallback
        for (size_t n = 0; n < neighbor_count; ++n) {
            BinaryCode<K> neighbor_signs = block.get_neighbor_binary_signs(n);
            out_distances[n] = rabitq_distance_scalar(query, neighbor_signs);
        }
#endif
    }
};

// Common type aliases for RaBitQ search
using RaBitQSearchLayer32 = RaBitQSearchLayer<32>;
using RaBitQSearchLayer64 = RaBitQSearchLayer<64>;

// ============================================================================
// Residual Search Layer (Phase 2 Integration)
// ============================================================================

/**
 * ResidualSearchLayer: NSW search using Phase 2 Residual Quantization.
 *
 * PHASE 2 OPTIMIZATION (PhD Portfolio):
 * Combines primary (K-bit) + residual (R-bit) codes for improved precision.
 *
 * Distance formula (integer-only until final conversion):
 *   CombinedDist = (PrimaryHamming << Shift) + ResidualHamming
 *
 * Template Parameters:
 * - K: Primary code width in bits (typically 64)
 * - R: Residual code width in bits (typically K/2 = 32)
 * - Shift: Bit-shift weighting (default 2 = 4:1 primary:residual ratio)
 *
 * Expected improvement: Graph-only recall ~50% → ~70% (vs Phase 1 alone)
 */
template <size_t K, size_t R = K / 2, int Shift = 2>
class ResidualSearchLayer {
public:
    using Query = ResidualQuery<K, R>;
    using Code = ResidualBinaryCode<K, R>;
    using Block = ResidualNeighborBlock<K, R>;

    /**
     * NSW Search using Residual distance.
     *
     * @param query         ResidualQuery with primary + residual binary codes
     * @param entry_points  Starting nodes for search
     * @param ef            Search width (candidates to explore)
     * @param codes         Array of ResidualBinaryCode (indexed by NodeId) - for node distances
     * @param blocks        Array of ResidualNeighborBlock (indexed by NodeId) - for neighbor storage
     * @param num_nodes     Total number of nodes in graph
     * @param visited       Visited markers array
     * @param query_id      Unique query ID for visited tracking
     * @return              Up to ef nearest neighbors found
     */
    static std::vector<SearchResult> search(
        const Query& query,
        const std::vector<NodeId>& entry_points,
        size_t ef,
        const Code* codes,
        const Block* blocks,
        size_t num_nodes,
        std::atomic<uint64_t>* visited,
        uint64_t query_id) {

        // W: found nearest neighbors (max-heap, bounded to ef)
        MaxHeap W;

        // C: candidates to explore (min-heap)
        MinHeap C;

        // Initialize with entry points
        for (NodeId ep : entry_points) {
            if (ep >= num_nodes) continue;

            // Check and mark visited (atomic)
            uint64_t expected = visited[ep].load(std::memory_order_relaxed);
            if (expected == query_id) continue;
            if (!visited[ep].compare_exchange_strong(expected, query_id)) continue;

            // Compute distance to entry point using its code
            const Code& ep_code = codes[ep];
            float dist = compute_node_distance(query, ep_code);

            C.push(ep, dist);
            W.push(ep, dist);
        }

        // Pre-allocate batch distance buffer
        alignas(64) std::array<float, FLASH_MAX_M> batch_distances;

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

            // Get neighbor block for current candidate
            if (c.id >= num_nodes) continue;
            const Block& block = blocks[c.id];
            size_t neighbor_count = block.count;

            // Prefetch next candidate's block
            if (!C.empty() && C.top().id < num_nodes) {
                __builtin_prefetch(&blocks[C.top().id], 0, 3);
            }

            // Compute all neighbor distances using residual distance
            residual_distance_block(
                query, block, neighbor_count, batch_distances.data());

            // Process batch results
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor_id = block.ids[i];

                if (neighbor_id == INVALID_NODE || neighbor_id >= num_nodes) continue;

                // Check and mark visited (atomic)
                uint64_t expected = visited[neighbor_id].load(std::memory_order_relaxed);
                if (expected == query_id) continue;
                if (!visited[neighbor_id].compare_exchange_strong(expected, query_id)) continue;

                float dist = batch_distances[i];

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
        const Code* codes,
        const Block* blocks,
        size_t num_nodes,
        std::atomic<uint64_t>* visited,
        uint64_t query_id) {

        return search(query, std::vector<NodeId>{entry_point}, ef,
                      codes, blocks, num_nodes, visited, query_id);
    }

private:
    /**
     * Compute distance to a node using its code.
     */
    static float compute_node_distance(const Query& query, const Code& node_code) {
        uint32_t combined = residual_distance_integer_scalar<K, R, Shift>(
            query.primary, query.residual,
            node_code.primary, node_code.residual);

        // Convert to float
        return query.base + query.scale * static_cast<float>(combined);
    }

    /**
     * Compute residual distances for all neighbors in a block.
     */
    static void residual_distance_block(
        const Query& query,
        const Block& block,
        size_t neighbor_count,
        float* out_distances) {

        constexpr size_t PRIM_WORDS = (K + 63) / 64;
        constexpr size_t RES_WORDS = (R + 63) / 64;

#if CPHNSW_HAS_AVX512
        // AVX-512 batch processing - 8 neighbors at a time
        alignas(64) uint32_t batch_combined[8];

        size_t n = 0;
        for (; n + 8 <= neighbor_count; n += 8) {
            residual_hamming_batch8_avx512<K, R, Shift>(
                query.primary.signs,
                query.residual.signs,
                block.primary_signs_transposed,
                block.residual_signs_transposed,
                n,
                batch_combined);

            for (size_t i = 0; i < 8; ++i) {
                out_distances[n + i] =
                    query.base + query.scale * static_cast<float>(batch_combined[i]);
            }
        }

        // Scalar remainder
        for (; n < neighbor_count; ++n) {
            BinaryCode<K> prim;
            BinaryCode<R> res;

            for (size_t w = 0; w < PRIM_WORDS; ++w) {
                prim.signs[w] = block.primary_signs_transposed[w][n];
            }
            for (size_t w = 0; w < RES_WORDS; ++w) {
                res.signs[w] = block.residual_signs_transposed[w][n];
            }

            uint32_t combined = residual_distance_integer_scalar<K, R, Shift>(
                query.primary, query.residual, prim, res);

            out_distances[n] = query.base + query.scale * static_cast<float>(combined);
        }
#else
        // Scalar fallback
        for (size_t n = 0; n < neighbor_count; ++n) {
            BinaryCode<K> prim;
            BinaryCode<R> res;

            for (size_t w = 0; w < PRIM_WORDS; ++w) {
                prim.signs[w] = block.primary_signs_transposed[w][n];
            }
            for (size_t w = 0; w < RES_WORDS; ++w) {
                res.signs[w] = block.residual_signs_transposed[w][n];
            }

            uint32_t combined = residual_distance_integer_scalar<K, R, Shift>(
                query.primary, query.residual, prim, res);

            out_distances[n] = query.base + query.scale * static_cast<float>(combined);
        }
#endif
    }
};

// Common type aliases for Residual search
using ResidualSearchLayer64_32 = ResidualSearchLayer<64, 32, 2>;
using ResidualSearchLayer32_16 = ResidualSearchLayer<32, 16, 2>;

}  // namespace cphnsw
