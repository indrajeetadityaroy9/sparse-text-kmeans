#pragma once

#include "../core/types.hpp"
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>
#include <cassert>

namespace cphnsw {

/**
 * FlatHNSWGraph: Memory-efficient HNSW graph structure.
 *
 * CRITICAL: Uses flat memory layout instead of vector<vector>.
 *
 * Benefits:
 * - Contiguous memory for better cache locality
 * - 16 bytes per node metadata vs 96+ bytes with vector<vector>
 * - Codes stored in parallel array for SIMD-friendly access
 *
 * Memory layout:
 * - links_pool: All links stored contiguously
 *   [Node0_L0_links | Node0_L1_links | Node1_L0_links | ...]
 * - nodes: Compact metadata (offset + layer info)
 * - codes: Parallel array of CP codes
 * - visited_markers: For thread-safe visited tracking
 */
template <typename ComponentT, size_t K>
class FlatHNSWGraph {
public:
    /// Maximum supported layers (exponential decay means rarely > 10)
    static constexpr size_t MAX_LAYERS = 16;

    /**
     * Compact node metadata.
     * 24 bytes aligned for cache efficiency.
     */
    struct alignas(8) CompactNode {
        uint32_t link_offset;       // Index into links_pool
        uint8_t max_layer;          // Highest layer this node belongs to
        uint8_t link_counts[16];    // Actual neighbor count per layer (supports MAX_LAYERS=16)
        uint8_t _padding[3];        // Pad to 24 bytes

        CompactNode() : link_offset(0), max_layer(0), _padding{} {
            std::memset(link_counts, 0, sizeof(link_counts));
        }
    };
    static_assert(sizeof(CompactNode) == 24, "CompactNode must be 24 bytes");

    /**
     * Construct graph with given parameters.
     */
    explicit FlatHNSWGraph(const CPHNSWParams& params)
        : params_(params), entry_point_(INVALID_NODE), top_layer_(0) {

        // Reserve initial capacity
        nodes_.reserve(1024);
        codes_.reserve(1024);
        visited_markers_.reserve(1024);

        // Estimate links per node: M_max0 + M * avg_layers
        // Conservatively estimate 2 layers average
        size_t links_per_node = params.M_max0 + params.M * 2;
        links_pool_.reserve(1024 * links_per_node);
    }

    /// Get number of nodes
    size_t size() const { return nodes_.size(); }

    /// Check if empty
    bool empty() const { return nodes_.empty(); }

    /// Get entry point
    NodeId entry_point() const { return entry_point_; }

    /// Get top layer
    LayerLevel top_layer() const { return top_layer_; }

    /// Set entry point and top layer
    void set_entry_point(NodeId node, LayerLevel layer) {
        entry_point_ = node;
        top_layer_ = layer;
    }

    /// Get parameters
    const CPHNSWParams& params() const { return params_; }

    /**
     * Add a new node with given code and level.
     *
     * @param code   CP code for the node
     * @param level  Maximum layer for this node
     * @return       Node ID
     */
    NodeId add_node(const CPCode<ComponentT, K>& code, LayerLevel level) {
        NodeId id = static_cast<NodeId>(nodes_.size());

        // Add compact node metadata
        CompactNode node;
        node.max_layer = level;
        node.link_offset = static_cast<uint32_t>(links_pool_.size());

        // Reserve space for links at all layers
        // Layer 0: M_max0 slots, Layers 1+: M slots each
        size_t total_slots = params_.M_max0;
        for (LayerLevel l = 1; l <= level; ++l) {
            total_slots += params_.M;
        }

        // Extend links pool with INVALID_NODE placeholders
        size_t old_size = links_pool_.size();
        links_pool_.resize(old_size + total_slots, INVALID_NODE);

        nodes_.push_back(node);
        codes_.push_back(code);
        visited_markers_.push_back(std::make_unique<std::atomic<uint64_t>>(0));

        return id;
    }

    /**
     * Get code for a node.
     */
    const CPCode<ComponentT, K>& get_code(NodeId id) const {
        assert(id < codes_.size());
        return codes_[id];
    }

    /**
     * Get mutable code reference.
     */
    CPCode<ComponentT, K>& get_code_mut(NodeId id) {
        assert(id < codes_.size());
        return codes_[id];
    }

    /**
     * Get node metadata.
     */
    const CompactNode& get_node(NodeId id) const {
        assert(id < nodes_.size());
        return nodes_[id];
    }

    /**
     * Get maximum layer for a node.
     */
    LayerLevel get_max_layer(NodeId id) const {
        return nodes_[id].max_layer;
    }

    /**
     * Get neighbor count at a specific layer.
     */
    size_t get_neighbor_count(NodeId id, LayerLevel layer) const {
        const auto& node = nodes_[id];
        assert(layer <= node.max_layer);
        return node.link_counts[layer];
    }

    /**
     * Get neighbors at a specific layer.
     *
     * @param id     Node ID
     * @param layer  Layer level
     * @return       Pointer to neighbor array and count
     */
    std::pair<const NodeId*, size_t> get_neighbors(NodeId id, LayerLevel layer) const {
        const auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        uint32_t offset = get_layer_offset(node, layer);
        size_t count = node.link_counts[layer];

        return {links_pool_.data() + offset, count};
    }

    /**
     * Add a neighbor to a node at a specific layer.
     *
     * @param id        Node ID
     * @param layer     Layer level
     * @param neighbor  Neighbor to add
     * @return          true if added, false if layer is full
     */
    bool add_neighbor(NodeId id, LayerLevel layer, NodeId neighbor) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        size_t max_count = (layer == 0) ? params_.M_max0 : params_.M;
        size_t current_count = node.link_counts[layer];

        if (current_count >= max_count) {
            return false;
        }

        uint32_t offset = get_layer_offset(node, layer);
        links_pool_[offset + current_count] = neighbor;
        node.link_counts[layer] = static_cast<uint8_t>(current_count + 1);

        return true;
    }

    /**
     * Set neighbors for a node at a specific layer (replaces existing).
     *
     * @param id         Node ID
     * @param layer      Layer level
     * @param neighbors  New neighbor list
     */
    void set_neighbors(NodeId id, LayerLevel layer, const std::vector<NodeId>& neighbors) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        size_t max_count = (layer == 0) ? params_.M_max0 : params_.M;
        size_t count = std::min(neighbors.size(), max_count);

        uint32_t offset = get_layer_offset(node, layer);

        // Copy neighbors
        for (size_t i = 0; i < count; ++i) {
            links_pool_[offset + i] = neighbors[i];
        }

        // Clear remaining slots
        for (size_t i = count; i < max_count; ++i) {
            links_pool_[offset + i] = INVALID_NODE;
        }

        node.link_counts[layer] = static_cast<uint8_t>(count);
    }

    /**
     * Check if a node was visited in the current query.
     * Uses atomic compare-exchange for thread safety.
     *
     * @param id        Node ID
     * @param query_id  Current query ID
     * @return          true if already visited, false if newly marked
     */
    bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
        uint64_t expected = visited_markers_[id]->load(std::memory_order_relaxed);

        if (expected == query_id) {
            return true;  // Already visited
        }

        // Try to mark as visited
        return !visited_markers_[id]->compare_exchange_weak(
            expected, query_id, std::memory_order_relaxed);
    }

    /**
     * Reset visited markers (alternative to incrementing query_id).
     */
    void reset_visited() {
        for (auto& marker : visited_markers_) {
            marker->store(0, std::memory_order_relaxed);
        }
    }

private:
    CPHNSWParams params_;
    NodeId entry_point_;
    LayerLevel top_layer_;

    // Flat storage
    std::vector<CompactNode> nodes_;
    std::vector<CPCode<ComponentT, K>> codes_;
    std::vector<NodeId> links_pool_;
    mutable std::vector<std::unique_ptr<std::atomic<uint64_t>>> visited_markers_;

    /**
     * Compute offset in links_pool for a specific layer.
     */
    uint32_t get_layer_offset(const CompactNode& node, LayerLevel layer) const {
        uint32_t offset = node.link_offset;

        // Layer 0 starts at offset
        if (layer == 0) {
            return offset;
        }

        // Skip layer 0 (M_max0 slots)
        offset += static_cast<uint32_t>(params_.M_max0);

        // Skip intermediate layers (M slots each)
        for (LayerLevel l = 1; l < layer; ++l) {
            offset += static_cast<uint32_t>(params_.M);
        }

        return offset;
    }
};

// Common graph types
using FlatHNSWGraph8 = FlatHNSWGraph<uint8_t, 16>;
using FlatHNSWGraph16 = FlatHNSWGraph<uint16_t, 16>;

}  // namespace cphnsw
