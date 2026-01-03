#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include "neighbor_block.hpp"
#include "visitation_table.hpp"
#include <vector>
#include <random>
#include <mutex>
#include <stdexcept>

namespace cphnsw {

// ============================================================================
// FlatGraph: Unified NSW graph storage
// ============================================================================

/**
 * FlatGraph: Memory-efficient Navigable Small World graph.
 *
 * UNIFIED DESIGN: Templated on CodeT, automatically adapts storage layout
 * based on whether the code has residual bits (Phase 2) or not (Phase 1).
 *
 * FLASH LAYOUT: Uses NeighborBlock to store neighbor codes inline for
 * cache-local distance computation during search.
 *
 * THREAD SAFETY:
 *   - add_node(): Thread-safe with mutex
 *   - get_neighbor_block(): Read-only, thread-safe
 *   - add_neighbor_safe(): Thread-safe with per-node spinlock
 *   - check_and_mark_visited(): Thread-safe with atomic CAS
 *
 * @tparam CodeT The code type (ResidualCode<K, R>)
 */
template <typename CodeT>
class FlatGraph {
public:
    using CodeType = CodeT;
    using Block = UnifiedNeighborBlock<CodeT>;
    using SoALayout = typename Block::SoALayout;

    static constexpr size_t PRIMARY_BITS = CodeT::PRIMARY_BITS;
    static constexpr size_t RESIDUAL_BITS = CodeT::RESIDUAL_BITS;
    static constexpr bool HAS_RESIDUAL = CodeT::HAS_RESIDUAL;

    // ========================================================================
    // Construction
    // ========================================================================

    /**
     * Construct graph with initial capacity.
     *
     * @param capacity Initial node capacity
     * @param max_neighbors Maximum neighbors per node (M parameter)
     */
    explicit FlatGraph(size_t capacity = 1024, size_t max_neighbors = 32)
        : max_neighbors_(max_neighbors)
        , entry_point_(INVALID_NODE)
        , visited_(capacity) {

        if (max_neighbors > MAX_NEIGHBORS) {
            throw std::invalid_argument("max_neighbors exceeds MAX_NEIGHBORS");
        }

        nodes_.reserve(capacity);
        codes_.reserve(capacity);
    }

    // ========================================================================
    // Node Management
    // ========================================================================

    /**
     * Add a new node to the graph.
     * Thread-safe.
     *
     * @param code The node's binary code
     * @return The new node's ID
     */
    NodeId add_node(const CodeT& code) {
        std::lock_guard<std::mutex> lock(nodes_mutex_);

        NodeId id = static_cast<NodeId>(nodes_.size());
        nodes_.emplace_back();
        codes_.push_back(code);

        // Update visitation table capacity
        if (id >= visited_.capacity()) {
            visited_.resize(id + 1024);
        }

        // Set first node as entry point
        if (entry_point_ == INVALID_NODE) {
            entry_point_ = id;
        }

        return id;
    }

    /**
     * Reserve capacity for nodes.
     * NOT thread-safe.
     */
    void reserve(size_t capacity) {
        nodes_.reserve(capacity);
        codes_.reserve(capacity);
        visited_.resize(capacity);
    }

    /**
     * Get the number of nodes.
     */
    size_t size() const { return nodes_.size(); }

    /**
     * Check if graph is empty.
     */
    bool empty() const { return nodes_.empty(); }

    // ========================================================================
    // Entry Point Management
    // ========================================================================

    NodeId entry_point() const { return entry_point_; }

    void set_entry_point(NodeId id) {
        if (id < nodes_.size()) {
            entry_point_ = id;
        }
    }

    /**
     * Get random entry points for search.
     */
    std::vector<NodeId> get_random_entry_points(size_t count, std::mt19937_64& rng) const {
        std::vector<NodeId> result;
        if (nodes_.empty()) return result;

        result.reserve(count);
        std::uniform_int_distribution<NodeId> dist(0, static_cast<NodeId>(nodes_.size() - 1));

        for (size_t i = 0; i < count && result.size() < nodes_.size(); ++i) {
            NodeId id = dist(rng);
            // Avoid duplicates
            bool duplicate = false;
            for (NodeId existing : result) {
                if (existing == id) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                result.push_back(id);
            }
        }

        return result;
    }

    // ========================================================================
    // Code Access
    // ========================================================================

    const CodeT& get_code(NodeId id) const {
        return codes_[id];
    }

    CodeT& get_code(NodeId id) {
        return codes_[id];
    }

    // ========================================================================
    // Neighbor Block Access
    // ========================================================================

    /**
     * Get neighbor block for a node (read-only).
     */
    const Block& get_neighbor_block(NodeId id) const {
        return nodes_[id];
    }

    /**
     * Get neighbor block for a node (mutable).
     */
    Block& get_neighbor_block(NodeId id) {
        return nodes_[id];
    }

    /**
     * Prefetch neighbor block into cache.
     */
    void prefetch_neighbor_block(NodeId id) const {
        if (id < nodes_.size()) {
            cphnsw::prefetch(&nodes_[id], 3);
        }
    }

    // ========================================================================
    // Neighbor Management
    // ========================================================================

    /**
     * Add a neighbor to a node (NOT thread-safe).
     */
    bool add_neighbor(NodeId node, NodeId neighbor, const CodeT& neighbor_code, float distance) {
        if (node >= nodes_.size()) return false;
        return nodes_[node].add(neighbor, neighbor_code, distance);
    }

    /**
     * Add a neighbor to a node (thread-safe).
     */
    bool add_neighbor_safe(NodeId node, NodeId neighbor, const CodeT& neighbor_code, float distance) {
        if (node >= nodes_.size()) return false;
        return nodes_[node].add_safe(neighbor, neighbor_code, distance);
    }

    /**
     * Try to add a neighbor, replacing worst if full (NOT thread-safe).
     */
    bool try_add_neighbor(NodeId node, NodeId neighbor, const CodeT& neighbor_code, float distance) {
        if (node >= nodes_.size()) return false;
        return nodes_[node].try_add(neighbor, neighbor_code, distance);
    }

    /**
     * Try to add a neighbor (thread-safe).
     */
    bool try_add_neighbor_safe(NodeId node, NodeId neighbor, const CodeT& neighbor_code, float distance) {
        if (node >= nodes_.size()) return false;
        return nodes_[node].try_add_safe(neighbor, neighbor_code, distance);
    }

    /**
     * Get neighbor count for a node.
     */
    size_t neighbor_count(NodeId id) const {
        return (id < nodes_.size()) ? nodes_[id].size() : 0;
    }

    /**
     * Check if two nodes are neighbors.
     */
    bool are_neighbors(NodeId a, NodeId b) const {
        return (a < nodes_.size()) && nodes_[a].contains(b);
    }

    // ========================================================================
    // Visitation (for search)
    // ========================================================================

    /**
     * Start a new query and get its ID.
     */
    uint64_t new_query() {
        return visited_.new_query();
    }

    /**
     * Check and mark a node as visited.
     * @return true if already visited, false if first visit
     */
    bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
        return visited_.check_and_mark(id, query_id);
    }

    bool is_visited(NodeId id, uint64_t query_id) const {
        return visited_.is_visited(id, query_id);
    }

    // ========================================================================
    // Graph Analysis
    // ========================================================================

    /**
     * Compute average degree (neighbors per node).
     */
    float average_degree() const {
        if (nodes_.empty()) return 0.0f;

        size_t total = 0;
        for (const auto& block : nodes_) {
            total += block.size();
        }
        return static_cast<float>(total) / nodes_.size();
    }

    /**
     * Count isolated nodes (no neighbors).
     */
    size_t count_isolated() const {
        size_t count = 0;
        for (const auto& block : nodes_) {
            if (block.empty()) ++count;
        }
        return count;
    }

    /**
     * Get maximum degree in the graph.
     */
    size_t max_degree() const {
        size_t max = 0;
        for (const auto& block : nodes_) {
            if (block.size() > max) max = block.size();
        }
        return max;
    }

    // ========================================================================
    // Configuration
    // ========================================================================

    size_t max_neighbors() const { return max_neighbors_; }

private:
    std::vector<Block> nodes_;
    std::vector<CodeT> codes_;
    size_t max_neighbors_;
    NodeId entry_point_;
    mutable VisitationTable visited_;
    mutable std::mutex nodes_mutex_;
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1: Pure binary codes
using FlatGraph32 = FlatGraph<ResidualCode<32, 0>>;
using FlatGraph64 = FlatGraph<ResidualCode<64, 0>>;

// Phase 2: With residual
using FlatGraph64_32 = FlatGraph<ResidualCode<64, 32>>;
using FlatGraph32_16 = FlatGraph<ResidualCode<32, 16>>;

}  // namespace cphnsw
