#pragma once

/**
 * ResidualCPHNSWIndex: Phase 2 Integrated Index
 *
 * Combines the Phase 1 RaBitQ symmetric distance with Phase 2 residual
 * quantization for improved graph-only recall.
 *
 * Key differences from CPHNSWIndex:
 * - Uses ResidualCPEncoder instead of CPEncoder
 * - Stores ResidualBinaryCode (primary + residual) per node
 * - Uses ResidualNeighborBlock for cache-friendly neighbor storage
 * - Uses ResidualSearchLayer for graph traversal
 *
 * Expected improvement: Graph-only recall ~50% -> ~70%
 */

#include "../core/types.hpp"
#include "../graph/flat_graph.hpp"
#include "../quantizer/residual_encoder.hpp"
#include "../algorithms/search_layer.hpp"
#include <memory>
#include <vector>
#include <atomic>
#include <random>
#include <algorithm>
#include <cstring>

namespace cphnsw {

/**
 * ResidualCPHNSWIndex: ANN Index with Phase 2 Residual Quantization
 *
 * Template Parameters:
 * - K: Primary code width in bits (typically 64)
 * - R: Residual code width in bits (typically K/2 = 32)
 * - Shift: Bit-shift weighting (default 2 = 4:1 primary:residual ratio)
 */
template <size_t K = 64, size_t R = K / 2, int Shift = 2>
class ResidualCPHNSWIndex {
public:
    using Code = ResidualBinaryCode<K, R>;
    using Query = ResidualQuery<K, R>;
    using Block = ResidualNeighborBlock<K, R>;
    using Encoder = ResidualCPEncoder<K, R>;
    using Search = ResidualSearchLayer<K, R, Shift>;

    static constexpr size_t PRIMARY_BITS = K;
    static constexpr size_t RESIDUAL_BITS = R;
    static constexpr int SHIFT = Shift;

    /**
     * Construct index with given parameters.
     *
     * @param dim              Vector dimension
     * @param M                Max connections per node (default 16)
     * @param ef_construction  Construction search width (default 100)
     * @param seed             Random seed for rotation chains
     */
    ResidualCPHNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 100,
                        uint64_t seed = 42)
        : dim_(dim),
          M_(M),
          ef_construction_(ef_construction),
          encoder_(dim, seed),
          rng_(seed),
          query_counter_(1),  // Start at 1 to avoid collision with initialized visited=0
          visited_size_(10000) {

        // Pre-allocate for typical usage
        codes_.reserve(10000);
        blocks_.reserve(10000);
        original_vectors_.reserve(10000 * dim);

        // Initialize visited array
        visited_ = std::make_unique<std::atomic<uint64_t>[]>(visited_size_);
        for (size_t i = 0; i < visited_size_; ++i) {
            visited_[i].store(0, std::memory_order_relaxed);
        }
    }

    /// Get number of vectors in index
    size_t size() const { return codes_.size(); }

    /// Check if index is empty
    bool empty() const { return codes_.empty(); }

    /// Get vector dimension
    size_t dim() const { return dim_; }

    /// Get M (max connections)
    size_t M() const { return M_; }

    /**
     * Add a single vector to the index.
     *
     * @param vec  Input vector (length >= dim)
     * @return     Node ID assigned
     */
    NodeId add(const Float* vec) {
        NodeId id = static_cast<NodeId>(codes_.size());

        // Store original vector for reranking
        original_vectors_.insert(original_vectors_.end(), vec, vec + dim_);

        // Encode to residual binary code
        Code code = encoder_.encode(vec);
        codes_.push_back(code);

        // Add neighbor block
        blocks_.emplace_back();

        // Expand visited array if needed
        if (id >= visited_size_) {
            // Need to reallocate - atomics aren't copyable
            size_t new_size = std::max(visited_size_ * 2, static_cast<size_t>(id + 1));
            auto new_visited = std::make_unique<std::atomic<uint64_t>[]>(new_size);
            for (size_t i = 0; i < new_size; ++i) {
                new_visited[i].store(0, std::memory_order_relaxed);
            }
            visited_ = std::move(new_visited);
            visited_size_ = new_size;
        }

        // Connect to graph
        if (id == 0) {
            // First node - nothing to connect
            entry_point_ = 0;
        } else {
            // Search for neighbors using existing graph
            Query query = encoder_.encode_query(vec);
            uint64_t qid = query_counter_.fetch_add(1);

            auto candidates = Search::search(
                query,
                entry_point_,
                ef_construction_,
                codes_.data(),
                blocks_.data(),
                codes_.size() - 1,  // Don't include new node yet
                visited_.get(),
                qid);

            // Select best M neighbors using TRUE L2 distance (hybrid approach)
            // This ensures high-quality graph edges even with quantization noise
            std::vector<std::pair<float, NodeId>> scored_candidates;
            scored_candidates.reserve(candidates.size());

            for (const auto& c : candidates) {
                const Float* neighbor_vec = original_vectors_.data() + c.id * dim_;
                float l2_dist = 0.0f;
                for (size_t d = 0; d < dim_; ++d) {
                    float diff = vec[d] - neighbor_vec[d];
                    l2_dist += diff * diff;
                }
                scored_candidates.emplace_back(l2_dist, c.id);
            }

            // Sort by true L2 distance
            std::sort(scored_candidates.begin(), scored_candidates.end());

            size_t num_neighbors = std::min(scored_candidates.size(), M_);

            // Add bidirectional edges
            Block& new_block = blocks_[id];
            for (size_t i = 0; i < num_neighbors; ++i) {
                NodeId neighbor_id = scored_candidates[i].second;

                // Add neighbor to new node
                add_neighbor(new_block, neighbor_id, codes_[neighbor_id]);

                // Add new node to neighbor (bidirectional)
                add_neighbor(blocks_[neighbor_id], id, code);
            }

            // Update entry point (use highest degree node for better connectivity)
            if (new_block.count > blocks_[entry_point_].count) {
                entry_point_ = id;
            }
        }

        return id;
    }

    /**
     * Add multiple vectors to the index.
     *
     * @param vecs   Input vectors (row-major, count x dim)
     * @param count  Number of vectors
     */
    void add_batch(const Float* vecs, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            add(vecs + i * dim_);
        }
    }

    /**
     * Search for k nearest neighbors (graph-only, no reranking).
     *
     * @param query  Query vector
     * @param k      Number of neighbors to return
     * @param ef     Search width (default = k)
     * @return       Up to k nearest neighbors
     */
    std::vector<SearchResult> search(const Float* query_vec, size_t k,
                                     size_t ef = 0) const {
        if (empty()) return {};

        if (ef == 0) ef = std::max(k, size_t(100));

        Query query = encoder_.encode_query(query_vec);
        uint64_t qid = query_counter_.fetch_add(1);

        auto results = Search::search(
            query,
            entry_point_,
            ef,
            codes_.data(),
            blocks_.data(),
            codes_.size(),
            visited_.get(),
            qid);

        // Return top-k
        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }

    /**
     * Search with hybrid reranking using original float vectors.
     *
     * @param query_vec  Query vector
     * @param k          Number of neighbors to return
     * @param ef         Search width for graph traversal
     * @param rerank_k   Number of candidates to rerank (default 200)
     * @return           Up to k nearest neighbors with float distances
     */
    std::vector<SearchResult> search_and_rerank(
        const Float* query_vec, size_t k, size_t ef, size_t rerank_k = 200) const {

        if (empty()) return {};

        if (ef == 0) ef = std::max(k, size_t(100));

        // Get candidates from graph search
        auto candidates = search(query_vec, std::max(k, rerank_k), ef);

        // Rerank using float L2 distance
        for (auto& c : candidates) {
            const Float* vec = original_vectors_.data() + c.id * dim_;
            Float dist = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                Float diff = query_vec[i] - vec[i];
                dist += diff * diff;
            }
            c.distance = dist;
        }

        // Sort by true distance
        std::sort(candidates.begin(), candidates.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });

        // Return top-k
        if (candidates.size() > k) {
            candidates.resize(k);
        }

        return candidates;
    }

    /**
     * Get the encoder (for external encoding).
     */
    const Encoder& encoder() const { return encoder_; }

    /**
     * Get entry point node.
     */
    NodeId entry_point() const { return entry_point_; }

    /**
     * Get original vector for a node.
     */
    const Float* get_vector(NodeId id) const {
        return original_vectors_.data() + id * dim_;
    }

    /**
     * Get residual code for a node.
     */
    const Code& get_code(NodeId id) const {
        return codes_[id];
    }

private:
    /**
     * Add a neighbor to a block (with proper SoA transposed storage).
     */
    void add_neighbor(Block& block, NodeId neighbor_id, const Code& neighbor_code) {
        if (block.count >= FLASH_MAX_M) {
            // Block is full - could implement pruning here
            return;
        }

        size_t idx = block.count;
        block.ids[idx] = neighbor_id;

        // Store in SoA transposed layout
        constexpr size_t PRIM_WORDS = (K + 63) / 64;
        constexpr size_t RES_WORDS = (R + 63) / 64;

        for (size_t w = 0; w < PRIM_WORDS; ++w) {
            block.primary_signs_transposed[w][idx] = neighbor_code.primary.signs[w];
        }
        for (size_t w = 0; w < RES_WORDS; ++w) {
            block.residual_signs_transposed[w][idx] = neighbor_code.residual.signs[w];
        }

        ++block.count;
    }

    size_t dim_;
    size_t M_;
    size_t ef_construction_;

    Encoder encoder_;
    std::mt19937 rng_;
    mutable std::atomic<uint64_t> query_counter_;

    std::vector<Code> codes_;
    std::vector<Block> blocks_;
    std::vector<Float> original_vectors_;
    mutable std::unique_ptr<std::atomic<uint64_t>[]> visited_;
    mutable size_t visited_size_ = 0;

    NodeId entry_point_ = 0;
};

// Common type aliases
using ResidualIndex64_32 = ResidualCPHNSWIndex<64, 32, 2>;
using ResidualIndex32_16 = ResidualCPHNSWIndex<32, 16, 2>;

}  // namespace cphnsw
