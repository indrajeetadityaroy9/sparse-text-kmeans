#pragma once

#include "../core/types.hpp"
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <limits>

// For _mm_pause() on x86
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define CPHNSW_HAS_PAUSE 1
#else
#define CPHNSW_HAS_PAUSE 0
#endif

namespace cphnsw {

/**
 * Spinlock: Lightweight lock for fine-grained per-node synchronization.
 *
 * CRITICAL: Uses _mm_pause() hint for Hyperthreading friendliness.
 * Without this, spinning appears as high-priority infinite loop,
 * starving other threads on the same physical core.
 *
 * Size: 1 byte (sizeof(std::atomic_flag))
 * This allows embedding in struct padding without increasing size.
 */
class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin with pause hint for better CPU efficiency
#if CPHNSW_HAS_PAUSE
            _mm_pause();
#endif
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

    // RAII guard for exception safety
    class Guard {
        Spinlock& lock_;
    public:
        explicit Guard(Spinlock& lock) : lock_(lock) { lock_.lock(); }
        ~Guard() { lock_.unlock(); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };
};

// Verify Spinlock is 1 byte to fit in struct padding
static_assert(sizeof(Spinlock) == 1, "Spinlock must be 1 byte to fit in CompactNode padding!");

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
     *
     * Includes a per-node spinlock for thread-safe neighbor updates.
     * The spinlock fits in the struct padding (1 byte).
     */
    struct alignas(8) CompactNode {
        uint32_t link_offset;           // Index into links_pool
        uint8_t max_layer;              // Highest layer this node belongs to
        uint8_t link_counts[16];        // Actual neighbor count per layer (supports MAX_LAYERS=16)
        uint8_t _padding[2];            // Reduced padding (was 3)
        mutable Spinlock lock;          // Per-node lock for thread-safe updates (1 byte)

        CompactNode() : link_offset(0), max_layer(0), _padding{}, lock() {
            std::memset(link_counts, 0, sizeof(link_counts));
        }

        // Allow copy construction (lock starts unlocked in copy)
        CompactNode(const CompactNode& other)
            : link_offset(other.link_offset), max_layer(other.max_layer), _padding{}, lock() {
            std::memcpy(link_counts, other.link_counts, sizeof(link_counts));
        }

        // Allow copy assignment (lock stays unchanged)
        CompactNode& operator=(const CompactNode& other) {
            if (this != &other) {
                link_offset = other.link_offset;
                max_layer = other.max_layer;
                std::memcpy(link_counts, other.link_counts, sizeof(link_counts));
            }
            return *this;
        }
    };
    static_assert(sizeof(CompactNode) == 24, "CompactNode must be 24 bytes for cache efficiency");

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
        dists_pool_.reserve(1024 * links_per_node);
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
        dists_pool_.resize(old_size + total_slots, std::numeric_limits<float>::max());

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
     * Uses atomic exchange for thread safety.
     *
     * NOTE: Using exchange() instead of compare_exchange_weak() because
     * compare_exchange_weak can spuriously fail, causing incorrect "already visited"
     * returns that skip nodes and degrade search recall.
     *
     * @param id        Node ID
     * @param query_id  Current query ID
     * @return          true if already visited, false if newly marked
     */
    bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
        uint64_t old = visited_markers_[id]->exchange(query_id, std::memory_order_relaxed);
        return (old == query_id);  // true if already visited
    }

    /**
     * Reset visited markers (alternative to incrementing query_id).
     */
    void reset_visited() {
        for (auto& marker : visited_markers_) {
            marker->store(0, std::memory_order_relaxed);
        }
    }

    /**
     * Thread-safe: Add a neighbor to a node at a specific layer.
     * Uses per-node spinlock for fine-grained synchronization.
     *
     * @param id        Node ID
     * @param layer     Layer level
     * @param neighbor  Neighbor to add
     * @return          true if added (or already exists), false if layer is full
     */
    bool add_neighbor_safe(NodeId id, LayerLevel layer, NodeId neighbor) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        Spinlock::Guard guard(node.lock);

        size_t max_count = (layer == 0) ? params_.M_max0 : params_.M;
        size_t current_count = node.link_counts[layer];
        uint32_t offset = get_layer_offset(node, layer);

        // Check for duplicate to prevent multiple threads adding same neighbor
        for (size_t i = 0; i < current_count; ++i) {
            if (links_pool_[offset + i] == neighbor) {
                return true;  // Already exists
            }
        }

        if (current_count >= max_count) {
            return false;
        }

        links_pool_[offset + current_count] = neighbor;
        node.link_counts[layer] = static_cast<uint8_t>(current_count + 1);

        return true;
    }

    /**
     * Thread-safe: Add neighbor with cached distance for O(1) pruning.
     *
     * CRITICAL OPTIMIZATION: Stores distance alongside link to avoid
     * recomputing distances inside the critical section during pruning.
     * This reduces lock hold time from ~5µs to ~50ns.
     *
     * @param id        Node ID
     * @param layer     Layer level
     * @param neighbor  Neighbor to add
     * @param dist      Distance to neighbor (lower = better, e.g., negative dot product)
     * @return          true if added or replaced worse neighbor, false if rejected
     */
    bool add_neighbor_with_dist_safe(NodeId id, LayerLevel layer, NodeId neighbor, float dist) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        Spinlock::Guard guard(node.lock);

        size_t max_count = (layer == 0) ? params_.M_max0 : params_.M;
        size_t current_count = node.link_counts[layer];
        uint32_t offset = get_layer_offset(node, layer);

        // Check for duplicate
        for (size_t i = 0; i < current_count; ++i) {
            if (links_pool_[offset + i] == neighbor) {
                return true;  // Already exists
            }
        }

        // Case 1: Not full - just append
        if (current_count < max_count) {
            links_pool_[offset + current_count] = neighbor;
            dists_pool_[offset + current_count] = dist;
            node.link_counts[layer] = static_cast<uint8_t>(current_count + 1);
            return true;
        }

        // Case 2: Full - find worst neighbor using CACHED distances (O(M) scan, no vector fetches!)
        size_t worst_idx = 0;
        float worst_dist = dists_pool_[offset];

        for (size_t i = 1; i < current_count; ++i) {
            if (dists_pool_[offset + i] > worst_dist) {
                worst_dist = dists_pool_[offset + i];
                worst_idx = i;
            }
        }

        // Replace worst if new neighbor is better
        if (dist < worst_dist) {
            links_pool_[offset + worst_idx] = neighbor;
            dists_pool_[offset + worst_idx] = dist;
            return true;
        }

        return false;  // New neighbor is worse than all existing
    }

    /**
     * Thread-safe: Set neighbors for a node at a specific layer.
     * Uses per-node spinlock for fine-grained synchronization.
     *
     * @param id         Node ID
     * @param layer      Layer level
     * @param neighbors  New neighbor list
     */
    void set_neighbors_safe(NodeId id, LayerLevel layer, const std::vector<NodeId>& neighbors) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        Spinlock::Guard guard(node.lock);

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
     * Thread-safe: Update entry point if new node has higher layer.
     *
     * @param node   Node ID
     * @param level  Node's maximum layer
     */
    void set_entry_point_safe(NodeId node, LayerLevel level) {
        Spinlock::Guard guard(entry_lock_);
        if (level > top_layer_ || entry_point_ == INVALID_NODE) {
            entry_point_ = node;
            top_layer_ = level;
        }
    }

    /**
     * Thread-safe: Add a link with automatic pruning if full.
     *
     * This performs the ENTIRE read-modify-write transaction under ONE lock
     * to prevent lost updates. The caller provides a distance function to
     * compute distances for pruning decisions.
     *
     * @param id           Node ID to modify
     * @param layer        Layer level
     * @param new_neighbor Neighbor to add
     * @param new_dist     Distance to new neighbor (lower = better)
     * @param dist_func    Function(NodeId) -> float to compute distance to existing neighbors
     */
    template<typename DistFunc>
    void add_link_with_pruning(NodeId id, LayerLevel layer, NodeId new_neighbor,
                               float new_dist, DistFunc dist_func) {
        auto& node = nodes_[id];
        assert(layer <= node.max_layer);

        Spinlock::Guard guard(node.lock);

        size_t max_count = (layer == 0) ? params_.M_max0 : params_.M;
        size_t current_count = node.link_counts[layer];
        uint32_t offset = get_layer_offset(node, layer);

        // Case 1: Not full - just add
        if (current_count < max_count) {
            links_pool_[offset + current_count] = new_neighbor;
            node.link_counts[layer] = static_cast<uint8_t>(current_count + 1);
            return;
        }

        // Case 2: Full - need to potentially prune
        // Build list of (distance, neighbor) pairs including new candidate
        std::vector<std::pair<float, NodeId>> candidates;
        candidates.reserve(current_count + 1);

        for (size_t i = 0; i < current_count; ++i) {
            NodeId neighbor = links_pool_[offset + i];
            if (neighbor != INVALID_NODE) {
                float dist = dist_func(neighbor);
                candidates.emplace_back(dist, neighbor);
            }
        }
        candidates.emplace_back(new_dist, new_neighbor);

        // Sort by distance (lower is better)
        std::sort(candidates.begin(), candidates.end());

        // Keep best M neighbors
        size_t keep = std::min(candidates.size(), max_count);
        for (size_t i = 0; i < keep; ++i) {
            links_pool_[offset + i] = candidates[i].second;
        }
        for (size_t i = keep; i < max_count; ++i) {
            links_pool_[offset + i] = INVALID_NODE;
        }
        node.link_counts[layer] = static_cast<uint8_t>(keep);
    }

    /**
     * Pre-allocate space for nodes (used in batch insertion).
     * MUST be called from a single thread before parallel insertion.
     *
     * @param count  Total number of nodes after reservation
     */
    void reserve_nodes(size_t count) {
        nodes_.reserve(count);
        codes_.reserve(count);
        visited_markers_.reserve(count);

        // Estimate links per node: M_max0 + M * avg_layers (assume 2)
        size_t links_per_node = params_.M_max0 + params_.M * 2;
        links_pool_.reserve(count * links_per_node);
        dists_pool_.reserve(count * links_per_node);
    }

private:
    CPHNSWParams params_;
    NodeId entry_point_;
    LayerLevel top_layer_;
    mutable Spinlock entry_lock_;  // Protects entry_point_ and top_layer_

    // Flat storage
    std::vector<CompactNode> nodes_;
    std::vector<CPCode<ComponentT, K>> codes_;
    std::vector<NodeId> links_pool_;
    std::vector<float> dists_pool_;  // Parallel array: cached distance for each edge
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
