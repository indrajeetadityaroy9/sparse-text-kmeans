#pragma once

#include "../core/types.hpp"
#include "../core/debug.hpp"
#include "../graph/flat_graph.hpp"
#include "../algorithms/search_layer_policy.hpp"
#include "../quantizer/quantizer_policy.hpp"
#include <memory>
#include <random>
#include <atomic>
#include <queue>
#include <cstring>
#include <numeric>
#include <algorithm>

#ifdef CPHNSW_USE_OPENMP
#include <omp.h>
#endif

namespace cphnsw {

/**
 * PolicyIndex: Quantizer-Agnostic ANN Index
 *
 * METRIC ALIGNMENT FIX:
 * This index uses a QuantizerPolicy to ensure the SAME distance metric
 * is used for both construction and search. This fixes the metric mismatch
 * problem where building with FHT distance but searching with A-ITQ distance
 * causes massive recall degradation.
 *
 * The quantizer policy provides:
 * - encode(): Vector -> Code (stored in index)
 * - encode_query(): Vector -> Query (used during search)
 * - search_distance(): Query x Code -> float (graph navigation)
 * - batch_search_distance_soa(): SIMD batch distance computation
 *
 * Usage with different quantizers:
 *
 *   // FHT-based (random projections)
 *   auto fht_policy = std::make_shared<CPFHTPolicy<uint8_t, 32>>(dim, seed);
 *   PolicyIndex<CPFHTPolicy<uint8_t, 32>> fht_index(params, fht_policy);
 *
 *   // A-ITQ (learned projections)
 *   auto aitq_policy = std::make_shared<AITQQuantizer<256>>(dim, train_data, N);
 *   PolicyIndex<AITQQuantizer<256>> aitq_index(params, aitq_policy);
 *
 * Both will build graphs optimized for their respective distance metrics.
 */
// Type trait to extract ComponentT from CPCode<ComponentT, K>
namespace detail {
    template<typename T>
    struct code_component_type {
        using type = uint8_t;  // Default fallback
    };

    template<typename ComponentT, size_t K>
    struct code_component_type<CPCode<ComponentT, K>> {
        using type = ComponentT;
    };

    template<typename ComponentT, size_t K>
    struct code_component_type<CPQuery<ComponentT, K>> {
        using type = ComponentT;
    };
}

/**
 * PolicyIndex: Quantizer-Agnostic ANN Index for CPCode-based quantizers.
 *
 * NOTE: This index is designed for CPFHTPolicy and similar quantizers that
 * use CPCode<ComponentT, K>. For A-ITQ quantizers, use AITQPolicyIndex.
 *
 * The ComponentT is automatically extracted from the quantizer's Code type:
 * - CPFHTPolicy<uint8_t, K> -> uses FlatHNSWGraph<uint8_t, K>
 * - CPFHTPolicy<uint16_t, K> -> uses FlatHNSWGraph<uint16_t, K>
 */
template <typename QuantizerT>
class PolicyIndex {
public:
    using Quantizer = QuantizerT;
    using Code = typename QuantizerT::Code;
    using Query = typename QuantizerT::Query;
    static constexpr size_t K = QuantizerT::CodeWidth;

    // Extract ComponentT from the quantizer's Code type
    // This ensures proper handling of uint8_t vs uint16_t for different dimensions
    using ComponentT = typename detail::code_component_type<Code>::type;

    // Graph type - uses the SAME component type as the quantizer
    using Graph = FlatHNSWGraph<ComponentT, K>;

    /**
     * Construct index with quantizer policy.
     *
     * @param params     Index configuration
     * @param quantizer  Quantizer policy (defines distance metric)
     */
    PolicyIndex(CPHNSWParams params, std::shared_ptr<Quantizer> quantizer)
        : params_(finalize_params(std::move(params))),
          quantizer_(std::move(quantizer)),
          graph_(params_),
          rng_(params_.seed),
          query_counter_(0) {}

    /// Get index parameters
    const CPHNSWParams& params() const { return params_; }

    /// Get quantizer
    const Quantizer& quantizer() const { return *quantizer_; }

    /// Get number of vectors in index
    size_t size() const { return graph_.size(); }

    /// Check if empty
    bool empty() const { return graph_.empty(); }

    /// Get vector dimension
    size_t dim() const { return quantizer_->dim(); }

    /**
     * Add a single vector to the index.
     *
     * Uses QUANTIZER'S DISTANCE for graph navigation during construction.
     * This ensures the graph topology is optimized for the search metric.
     */
    NodeId add(const Float* vec) {
        // Store original vector
        original_vectors_.insert(original_vectors_.end(), vec, vec + params_.dim);

        // Encode using quantizer
        auto query = quantizer_->encode_query(vec);

        // Add node to graph
        // Convert the quantizer's code to the graph's storage format
        auto graph_code = convert_to_graph_code(query);
        NodeId id = graph_.add_node(graph_code);

        // Handle first node
        if (graph_.size() == 1) {
            return id;
        }

        // Get entry points
        uint64_t qid = query_counter_.fetch_add(1);
        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        // Insert using QUANTIZER'S DISTANCE for graph navigation
        // but TRUE DISTANCE for edge selection (hybrid approach)
        PolicyInsert<Quantizer, Graph>::insert_hybrid(
            id, query, vec, original_vectors_, params_.dim,
            entry_points, graph_, *quantizer_, params_, qid);

        return id;
    }

    /**
     * Add multiple vectors (sequential).
     */
    void add_batch(const Float* vecs, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            add(vecs + i * params_.dim);
        }
    }

    /**
     * Add multiple vectors (parallel).
     *
     * METRIC-ALIGNED CONSTRUCTION:
     * Uses quantizer's distance during graph navigation.
     */
    void add_batch_parallel(const Float* vecs, size_t count, size_t batch_size = 0) {
        size_t start_id = graph_.size();
        size_t dim = params_.dim;

        if (batch_size == 0) {
#ifdef CPHNSW_USE_OPENMP
            batch_size = static_cast<size_t>(omp_get_max_threads());
#else
            batch_size = 64;
#endif
        }

        // Phase 1: Pre-allocation
        CPHNSW_DEBUG_PHASE(1, "Pre-allocating for " << count << " vectors");
        original_vectors_.resize((start_id + count) * dim);
        graph_.reserve_nodes(start_id + count);

        // Phase 2: Encoding (parallel)
        CPHNSW_DEBUG_PHASE(2, "Encoding " << count << " vectors");
        std::vector<Query> queries(count);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel
        {
            std::vector<Float> local_buffer(quantizer_->padded_dim());

            #pragma omp for schedule(static)
            for (size_t i = 0; i < count; ++i) {
                const Float* vec = vecs + i * dim;

                // Copy vector
                std::memcpy(original_vectors_.data() + (start_id + i) * dim,
                            vec, dim * sizeof(Float));

                // Encode
                queries[i] = quantizer_->encode_query_with_buffer(vec, local_buffer.data());
            }
        }
#else
        for (size_t i = 0; i < count; ++i) {
            const Float* vec = vecs + i * dim;
            std::memcpy(original_vectors_.data() + (start_id + i) * dim,
                        vec, dim * sizeof(Float));
            queries[i] = quantizer_->encode_query(vec);
        }
#endif

        // Phase 3: Node creation (sequential)
        CPHNSW_DEBUG_PHASE(3, "Creating " << count << " nodes");
        for (size_t i = 0; i < count; ++i) {
            auto graph_code = convert_to_graph_code(queries[i]);
            graph_.add_node(graph_code);
        }

        // Phase 4: Bootstrap (sequential)
        size_t bootstrap_count = std::min(static_cast<size_t>(1000), count);
        CPHNSW_DEBUG("Bootstrap " << bootstrap_count << " nodes");

        for (size_t i = 0; i < bootstrap_count; ++i) {
            NodeId id = static_cast<NodeId>(start_id + i);
            const Float* vec = vecs + i * dim;
            uint64_t qid = query_counter_.fetch_add(1);

            std::vector<NodeId> entry_points = graph_.get_random_entry_points(
                params_.k_entry, qid);

            PolicyInsert<Quantizer, Graph>::insert_hybrid(
                id, queries[i], vec, original_vectors_, dim,
                entry_points, graph_, *quantizer_, params_, qid);
        }

        // Phase 5: Shuffled parallel linking
        CPHNSW_DEBUG_PHASE(5, "Shuffled parallel linking");
        size_t remaining = count - bootstrap_count;

        std::vector<size_t> shuffled_indices(remaining);
        std::iota(shuffled_indices.begin(), shuffled_indices.end(), bootstrap_count);

        std::mt19937_64 shuffle_rng(params_.seed + 12345);
        std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), shuffle_rng);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 64)
#endif
        for (size_t idx = 0; idx < remaining; ++idx) {
            size_t i = shuffled_indices[idx];
            NodeId id = static_cast<NodeId>(start_id + i);
            const Float* vec = vecs + i * dim;

            uint64_t qid = query_counter_.fetch_add(1);
            std::vector<NodeId> entry_points = graph_.get_random_entry_points(
                params_.k_entry, qid);

            PolicyInsert<Quantizer, Graph>::insert_hybrid(
                id, queries[i], vec, original_vectors_, dim,
                entry_points, graph_, *quantizer_, params_, qid);
        }
    }

    /**
     * Search for k nearest neighbors.
     *
     * Uses QUANTIZER'S DISTANCE for graph navigation.
     */
    std::vector<SearchResult> search(const Float* query, size_t k, size_t ef = 0) const {
        if (ef == 0) {
            ef = std::max(k, static_cast<size_t>(10));
        }

        auto q = quantizer_->encode_query(query);
        uint64_t qid = query_counter_.fetch_add(1);

        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        return PolicySearchLayer<Quantizer, Graph>::search(
            q, entry_points, ef, graph_, *quantizer_, qid);
    }

    /**
     * Search with re-ranking using true distance.
     *
     * Uses QUANTIZER'S DISTANCE for candidate generation,
     * then TRUE DISTANCE for final ranking.
     */
    std::vector<SearchResult> search_and_rerank(
        const Float* query_vec,
        size_t k,
        size_t ef = 0,
        size_t rerank_k = 0) const {

        if (ef == 0) ef = std::max(k * 2, static_cast<size_t>(100));
        if (rerank_k == 0) rerank_k = std::max(k * 5, static_cast<size_t>(100));

        // Candidate generation with quantizer's distance
        auto q = quantizer_->encode_query(query_vec);
        uint64_t qid = query_counter_.fetch_add(1);

        std::vector<NodeId> entry_points = graph_.get_random_entry_points(
            params_.k_entry, qid);

        auto candidates = PolicySearchLayer<Quantizer, Graph>::search(
            q, entry_points, rerank_k, graph_, *quantizer_, qid);

        // Re-rank with TRUE distance
#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for if(candidates.size() > 100)
#endif
        for (size_t i = 0; i < candidates.size(); ++i) {
            NodeId id = candidates[i].id;
            const Float* node_vec = original_vectors_.data() + id * params_.dim;

            Float dot = 0;
            for (size_t d = 0; d < params_.dim; ++d) {
                dot += query_vec[d] * node_vec[d];
            }
            candidates[i].distance = -dot;
        }

        // Sort and return top-k
        if (candidates.size() > k) {
            std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end());
            candidates.resize(k);
        } else {
            std::sort(candidates.begin(), candidates.end());
        }

        return candidates;
    }

    /**
     * Batch search.
     */
    std::vector<std::vector<SearchResult>> search_batch(
        const Float* queries, size_t num_queries, size_t k, size_t ef = 0) const {

        if (ef == 0) ef = std::max(k, static_cast<size_t>(10));

        std::vector<std::vector<SearchResult>> results(num_queries);

#ifdef CPHNSW_USE_OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search(queries + i * params_.dim, k, ef);
        }

        return results;
    }

    /**
     * Verify graph connectivity.
     */
    size_t verify_connectivity() const {
        if (graph_.empty()) return 0;

        std::vector<bool> visited(graph_.size(), false);
        std::queue<NodeId> queue;

        queue.push(0);
        visited[0] = true;
        size_t count = 1;

        while (!queue.empty()) {
            NodeId node = queue.front();
            queue.pop();

            auto [neighbors, neighbor_count] = graph_.get_neighbors(node);
            for (size_t i = 0; i < neighbor_count; ++i) {
                NodeId neighbor = neighbors[i];
                if (neighbor != INVALID_NODE && !visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                    ++count;
                }
            }
        }

        return count;
    }

    bool is_connected() const {
        return verify_connectivity() == graph_.size();
    }

private:
    CPHNSWParams params_;
    std::shared_ptr<Quantizer> quantizer_;
    mutable Graph graph_;
    std::mt19937_64 rng_;
    mutable std::atomic<uint64_t> query_counter_;
    std::vector<Float> original_vectors_;

    static CPHNSWParams finalize_params(CPHNSWParams params) {
        params.finalize();
        return params;
    }

    /**
     * Convert quantizer's query code to graph's storage format.
     *
     * For CPFHTPolicy: Query has primary_code (CPCode) - direct copy
     * For other quantizers with CPCode-based queries: extract primary_code
     *
     * NOTE: A-ITQ uses AITQCode (bit-packed) which is incompatible with
     * FlatHNSWGraph's CPCode format. Use AITQPolicyIndex for A-ITQ.
     */
    CPCode<ComponentT, K> convert_to_graph_code(const Query& query) const {
        CPCode<ComponentT, K> graph_code;

        // Use compile-time detection of query structure
        // CPQuery<ComponentT, K> has a primary_code member of type CPCode<ComponentT, K>
        if constexpr (requires { query.primary_code; }) {
            // FHT-style query with primary_code field
            // Direct copy - ComponentT matches between query and graph
            for (size_t i = 0; i < K; ++i) {
                graph_code.components[i] = query.primary_code.components[i];
                graph_code.magnitudes[i] = query.primary_code.magnitudes[i];
            }
        } else if constexpr (requires { query.code; query.proj; }) {
            // A-ITQ style query - NOT SUPPORTED by FlatHNSWGraph!
            // This path exists for error detection at compile time
            static_assert(!requires { query.code; query.proj; },
                "AITQQuantizer is not compatible with PolicyIndex. "
                "Use AITQPolicyIndex instead for A-ITQ quantizers.");
        } else {
            // Unknown query type - zero initialize
            for (size_t i = 0; i < K; ++i) {
                graph_code.components[i] = ComponentT{0};
                graph_code.magnitudes[i] = 0;
            }
        }

        return graph_code;
    }
};

}  // namespace cphnsw
