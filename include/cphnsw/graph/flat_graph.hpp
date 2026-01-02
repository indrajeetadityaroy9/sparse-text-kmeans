#pragma once

#include "../core/types.hpp"
#include <vector>
#include <atomic>
#include <memory>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <limits>
#include <random>
#include <new>  // For std::align_val_t
#include <fstream>
#include <stdexcept>

// For _mm_pause() on x86
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define CPHNSW_HAS_PAUSE 1
#else
#define CPHNSW_HAS_PAUSE 0
#endif

namespace cphnsw {

// ============================================================================
// Flash Memory Layout Constants
// ============================================================================

/// Maximum neighbors per node (compile-time for Flash layout)
constexpr size_t FLASH_MAX_M = 64;

/// Cache line size for alignment
constexpr size_t FLASH_CACHE_LINE = 64;

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

// ============================================================================
// Flash Memory Layout: NeighborBlock (SoA - SIMD Optimized)
// ============================================================================

/**
 * NeighborBlock: Cache-optimized storage for a node's neighbors.
 *
 * FLASH OPTIMIZATION: Stores COPIES of neighbor codes alongside neighbor IDs.
 * This eliminates pointer-chasing during search - all data needed to compute
 * distances to neighbors is contiguous and can be prefetched together.
 *
 * RABITQ AUGMENTATION (SIGMOD 2024):
 * Also stores quantized magnitudes for each neighbor's code components.
 * This enables magnitude-weighted distance computation for better accuracy.
 *
 * CRITICAL: SoA (Struct-of-Arrays) TRANSPOSED LAYOUT
 * ================================================
 * codes_transposed[component_idx][neighbor_idx]
 * magnitudes_transposed[component_idx][neighbor_idx]
 *
 * Why SoA instead of AoS?
 * - Enables SIMD batch processing of multiple neighbors simultaneously
 * - For byte k: codes_transposed[k][0..N] is CONTIGUOUS
 * - Single AVX-512 load fetches byte k for 16 neighbors at once
 * - Gather operation then looks up query.rotated_vecs[k][indices]
 *
 * Example for K=32, M=32:
 *   codes_transposed[0][0..31] = component 0 for all 32 neighbors (contiguous!)
 *   codes_transposed[1][0..31] = component 1 for all 32 neighbors
 *   ...
 *   magnitudes_transposed[0][0..31] = magnitude 0 for all 32 neighbors
 *   magnitudes_transposed[1][0..31] = magnitude 1 for all 32 neighbors
 *   ...
 *
 * Memory layout (for M=32, K=32):
 *   - ids[M]:                       32 * 4  = 128 bytes
 *   - codes_transposed[K][M]:       32 * 32 = 1024 bytes
 *   - magnitudes_transposed[K][M]:  32 * 32 = 1024 bytes (NEW: RaBitQ)
 *   - distances[M]:                 32 * 4  = 128 bytes
 *   - count + lock + padding
 *   Total: ~2306 bytes = 37 cache lines
 *
 * SIMD BATCH PROCESSING:
 *   - Process 16 neighbors per AVX-512 iteration
 *   - Load codes_transposed[k][0..15] → 16 bytes contiguous
 *   - Load magnitudes_transposed[k][0..15] → 16 bytes contiguous
 *   - Expand to 32-bit indices → gather from rotated_vecs[k]
 *   - Multiply by magnitudes and accumulate 16 scores in parallel
 *   - Expected speedup: 3-5x over scalar
 */
template <typename ComponentT, size_t K>
struct alignas(FLASH_CACHE_LINE) NeighborBlock {
    /// Neighbor node IDs (INVALID_NODE if slot unused)
    NodeId ids[FLASH_MAX_M];

    /// TRANSPOSED neighbor codes for SIMD batch processing
    /// Layout: codes_transposed[component_idx][neighbor_idx]
    /// This enables contiguous loads of the same component across all neighbors
    ComponentT codes_transposed[K][FLASH_MAX_M];

    /// TRANSPOSED neighbor magnitudes for RaBitQ-style distance (NEW)
    /// Layout: magnitudes_transposed[component_idx][neighbor_idx]
    uint8_t magnitudes_transposed[K][FLASH_MAX_M];

    /// Cached distances (negative dot product) for O(1) pruning decisions
    float distances[FLASH_MAX_M];

    /// Actual number of neighbors (0 to M)
    /// THREAD SAFETY: Atomic to allow lock-free reads during search.
    /// Writers must use release semantics after updating ids/codes.
    /// Readers should use acquire semantics to see consistent data.
    std::atomic<uint8_t> count;

    /// Per-node spinlock for thread-safe updates
    mutable Spinlock lock;

    /// Padding to cache line boundary (updated for new magnitudes array)
    uint8_t _padding[FLASH_CACHE_LINE - ((sizeof(NodeId) * FLASH_MAX_M +
                                          sizeof(ComponentT) * K * FLASH_MAX_M +
                                          sizeof(uint8_t) * K * FLASH_MAX_M +
                                          sizeof(float) * FLASH_MAX_M +
                                          sizeof(std::atomic<uint8_t>) + sizeof(Spinlock)) % FLASH_CACHE_LINE)];

    NeighborBlock() : count(0), lock() {
        std::fill(std::begin(ids), std::end(ids), INVALID_NODE);
        for (size_t k = 0; k < K; ++k) {
            std::fill(std::begin(codes_transposed[k]), std::end(codes_transposed[k]), ComponentT(0));
            std::fill(std::begin(magnitudes_transposed[k]), std::end(magnitudes_transposed[k]), uint8_t(0));
        }
        std::fill(std::begin(distances), std::end(distances), std::numeric_limits<float>::max());
    }

    // Copy constructor: copy data, create new lock
    NeighborBlock(const NeighborBlock& other) : count(other.count.load(std::memory_order_relaxed)), lock() {
        std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
        for (size_t k = 0; k < K; ++k) {
            std::copy(std::begin(other.codes_transposed[k]), std::end(other.codes_transposed[k]),
                      std::begin(codes_transposed[k]));
            std::copy(std::begin(other.magnitudes_transposed[k]), std::end(other.magnitudes_transposed[k]),
                      std::begin(magnitudes_transposed[k]));
        }
        std::copy(std::begin(other.distances), std::end(other.distances), std::begin(distances));
    }

    // Move constructor: move data, create new lock
    NeighborBlock(NeighborBlock&& other) noexcept : count(other.count.load(std::memory_order_relaxed)), lock() {
        std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
        for (size_t k = 0; k < K; ++k) {
            std::copy(std::begin(other.codes_transposed[k]), std::end(other.codes_transposed[k]),
                      std::begin(codes_transposed[k]));
            std::copy(std::begin(other.magnitudes_transposed[k]), std::end(other.magnitudes_transposed[k]),
                      std::begin(magnitudes_transposed[k]));
        }
        std::copy(std::begin(other.distances), std::end(other.distances), std::begin(distances));
    }

    // Copy assignment: copy data, keep own lock
    NeighborBlock& operator=(const NeighborBlock& other) {
        if (this != &other) {
            count.store(other.count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
            for (size_t k = 0; k < K; ++k) {
                std::copy(std::begin(other.codes_transposed[k]), std::end(other.codes_transposed[k]),
                          std::begin(codes_transposed[k]));
                std::copy(std::begin(other.magnitudes_transposed[k]), std::end(other.magnitudes_transposed[k]),
                          std::begin(magnitudes_transposed[k]));
            }
            std::copy(std::begin(other.distances), std::end(other.distances), std::begin(distances));
        }
        return *this;
    }

    // Move assignment: move data, keep own lock
    NeighborBlock& operator=(NeighborBlock&& other) noexcept {
        if (this != &other) {
            count.store(other.count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
            for (size_t k = 0; k < K; ++k) {
                std::copy(std::begin(other.codes_transposed[k]), std::end(other.codes_transposed[k]),
                          std::begin(codes_transposed[k]));
                std::copy(std::begin(other.magnitudes_transposed[k]), std::end(other.magnitudes_transposed[k]),
                          std::begin(magnitudes_transposed[k]));
            }
            std::copy(std::begin(other.distances), std::end(other.distances), std::begin(distances));
        }
        return *this;
    }

    /// Get component k for neighbor i (SoA access)
    ComponentT get_component(size_t k, size_t neighbor_idx) const {
        return codes_transposed[k][neighbor_idx];
    }

    /// Get magnitude k for neighbor i (SoA access, NEW)
    uint8_t get_magnitude(size_t k, size_t neighbor_idx) const {
        return magnitudes_transposed[k][neighbor_idx];
    }

    /// Get pointer to component k for all neighbors (for SIMD batch load)
    const ComponentT* get_component_row(size_t k) const {
        return codes_transposed[k];
    }

    /// Get pointer to magnitude k for all neighbors (for SIMD batch load, NEW)
    const uint8_t* get_magnitude_row(size_t k) const {
        return magnitudes_transposed[k];
    }

    /// Copy a code into neighbor slot i (scatters to transposed layout)
    /// RABITQ: Also copies magnitudes for magnitude-weighted distance
    void set_neighbor_code(size_t i, const CPCode<ComponentT, K>& code) {
        for (size_t k = 0; k < K; ++k) {
            codes_transposed[k][i] = code.components[k];
            magnitudes_transposed[k][i] = code.magnitudes[k];
        }
    }

    /// Create CPCode from neighbor at index i (gathers from transposed layout)
    /// RABITQ: Also gathers magnitudes
    CPCode<ComponentT, K> get_neighbor_code_copy(size_t i) const {
        CPCode<ComponentT, K> result;
        for (size_t k = 0; k < K; ++k) {
            result.components[k] = codes_transposed[k][i];
            result.magnitudes[k] = magnitudes_transposed[k][i];
        }
        return result;
    }

    /// Legacy API: Get code pointer for neighbor i (DEPRECATED - use batch API)
    /// Returns temporary buffer, caller must copy immediately
    /// This exists only for backward compatibility with scalar code paths
    void get_neighbor_code_to_buffer(size_t i, ComponentT* buffer) const {
        for (size_t k = 0; k < K; ++k) {
            buffer[k] = codes_transposed[k][i];
        }
    }

    /// Get magnitudes for neighbor i to buffer (for compatibility)
    void get_neighbor_magnitudes_to_buffer(size_t i, uint8_t* buffer) const {
        for (size_t k = 0; k < K; ++k) {
            buffer[k] = magnitudes_transposed[k][i];
        }
    }
};

/**
 * FlatNSWGraph: Memory-efficient NSW (single-layer) graph structure.
 *
 * NSW FLATTEN: Removed hierarchy for simpler, faster graph traversal.
 * Per "Down with the Hierarchy" - single layer with random entry points
 * achieves equivalent performance for high-dimensional data.
 *
 * FLASH MEMORY LAYOUT: Optimized for cache locality during search.
 * Each node has a NeighborBlock containing:
 *   - Neighbor IDs
 *   - COPIES of neighbor codes (eliminates pointer chasing)
 *   - Cached distances for O(1) pruning
 *
 * Benefits:
 * - All neighbor data prefetchable in one operation
 * - Codes stored in AoS layout for SIMD-friendly access
 * - 64-byte aligned blocks for cache line efficiency
 * - K=32 codes fit perfectly in AVX-256 (1 neighbor) or AVX-512 (2 neighbors)
 *
 * Memory layout:
 * - codes_: Node's own code (for encoding/construction)
 * - neighbor_blocks_: NeighborBlock per node (Flash layout)
 * - visited_markers_: For thread-safe visited tracking
 *
 * Trade-off: More memory (neighbor codes stored redundantly) for faster search.
 */
template <typename ComponentT, size_t K>
class FlatHNSWGraph {
public:
    /// Flash memory layout: NeighborBlock type for this graph
    using Block = NeighborBlock<ComponentT, K>;

    /**
     * Construct graph with given parameters.
     */
    explicit FlatHNSWGraph(const CPHNSWParams& params)
        : params_(params) {

        // Validate M fits in Flash layout
        assert(params.M <= FLASH_MAX_M && "M exceeds FLASH_MAX_M!");

        // Reserve initial capacity
        codes_.reserve(1024);
        neighbor_blocks_.reserve(1024);
        visited_markers_.reserve(1024);
    }

    /// Get number of nodes
    size_t size() const { return codes_.size(); }

    /// Check if empty
    bool empty() const { return codes_.empty(); }

    /// Get parameters
    const CPHNSWParams& params() const { return params_; }

    /**
     * NSW: Get random entry points for search.
     * Replaces hierarchical entry point with random sampling.
     *
     * @param k     Number of entry points to return
     * @param seed  Random seed (typically query_id for reproducibility)
     * @return      Vector of random node IDs
     */
    std::vector<NodeId> get_random_entry_points(size_t k, uint64_t seed) const {
        if (empty()) return {};
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<NodeId> dist(0, static_cast<NodeId>(size() - 1));
        std::vector<NodeId> entries(std::min(k, size()));
        for (auto& e : entries) e = dist(rng);
        return entries;
        // NOTE: For static build (immutable index), indices 0..N-1 are always valid.
        // No need to check for "holes" since we don't support deletion.
    }

    /**
     * NSW + Flash: Add a new node with given code.
     * Creates a NeighborBlock for Flash memory layout.
     *
     * @param code   CP code for the node
     * @return       Node ID
     */
    NodeId add_node(const CPCode<ComponentT, K>& code) {
        NodeId id = static_cast<NodeId>(codes_.size());

        // Store node's own code
        codes_.push_back(code);

        // Create neighbor block (Flash layout)
        neighbor_blocks_.push_back(Block{});

        // Visited marker for search
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
     * Flash: Get the neighbor block for a node.
     * Returns reference for direct access during search.
     */
    const Block& get_neighbor_block(NodeId id) const {
        assert(id < neighbor_blocks_.size());
        return neighbor_blocks_[id];
    }

    /**
     * Flash: Get mutable neighbor block reference.
     */
    Block& get_neighbor_block_mut(NodeId id) {
        assert(id < neighbor_blocks_.size());
        return neighbor_blocks_[id];
    }

    /**
     * NSW: Get neighbor count for a node.
     * THREAD SAFETY: Uses acquire semantics to ensure visibility of neighbor data.
     */
    size_t get_neighbor_count(NodeId id) const {
        return neighbor_blocks_[id].count.load(std::memory_order_acquire);
    }

    /**
     * NSW: Get neighbors for a node (legacy API).
     * THREAD SAFETY: Uses acquire semantics on count read.
     *
     * @param id     Node ID
     * @return       Pointer to neighbor array and count
     */
    std::pair<const NodeId*, size_t> get_neighbors(NodeId id) const {
        const auto& block = neighbor_blocks_[id];
        return {block.ids, block.count.load(std::memory_order_acquire)};
    }

    /**
     * NSW + Flash: Add a neighbor to a node.
     * Copies neighbor's code into the block for cache locality.
     * NOTE: Not thread-safe. Use add_neighbor_safe for concurrent access.
     *
     * @param id        Node ID
     * @param neighbor  Neighbor to add
     * @return          true if added, false if full
     */
    bool add_neighbor(NodeId id, NodeId neighbor) {
        auto& block = neighbor_blocks_[id];
        size_t current_count = block.count.load(std::memory_order_relaxed);

        if (current_count >= params_.M) {
            return false;
        }

        // Store neighbor ID
        block.ids[current_count] = neighbor;

        // FLASH: Copy neighbor's code for cache locality during search
        block.set_neighbor_code(current_count, codes_[neighbor]);

        // Release semantics: ensure ID and code writes are visible before count update
        block.count.store(static_cast<uint8_t>(current_count + 1), std::memory_order_release);
        return true;
    }

    /**
     * NSW + Flash: Set neighbors for a node (replaces existing).
     * Copies all neighbor codes into the block.
     * NOTE: Not thread-safe. Use set_neighbors_safe for concurrent access.
     *
     * @param id         Node ID
     * @param neighbors  New neighbor list
     */
    void set_neighbors(NodeId id, const std::vector<NodeId>& neighbors) {
        auto& block = neighbor_blocks_[id];
        size_t new_count = std::min(neighbors.size(), params_.M);

        // Copy neighbors and their codes
        for (size_t i = 0; i < new_count; ++i) {
            block.ids[i] = neighbors[i];
            block.set_neighbor_code(i, codes_[neighbors[i]]);
        }

        // Clear remaining slots
        for (size_t i = new_count; i < params_.M; ++i) {
            block.ids[i] = INVALID_NODE;
        }

        // Release semantics: ensure all writes are visible before count update
        block.count.store(static_cast<uint8_t>(new_count), std::memory_order_release);
    }

    /**
     * Check if a node was visited in the current query.
     * Uses atomic exchange for thread safety.
     *
     * NOTE: Using exchange() instead of compare_exchange_weak() because
     * compare_exchange_weak can spuriously fail, causing incorrect "already visited"
     * returns that skip nodes and degrade search recall.
     *
     * WRAP-AROUND HANDLING:
     * Query IDs are 64-bit, so wrap-around after 2^64 queries is theoretically
     * possible but practically never occurs (584 years at 1 billion QPS).
     * If query_id is 0, we skip the visited check to handle the edge case
     * where wrap-around occurs or after reset_visited() is called.
     *
     * @param id        Node ID
     * @param query_id  Current query ID (must be > 0)
     * @return          true if already visited, false if newly marked
     */
    bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
        // Handle query_id = 0 edge case (wrap-around or after reset)
        // Treat as never visited to force fresh traversal
        if (query_id == 0) {
            visited_markers_[id]->store(0, std::memory_order_relaxed);
            return false;
        }
        uint64_t old = visited_markers_[id]->exchange(query_id, std::memory_order_relaxed);
        return (old == query_id);  // true if already visited
    }

    /**
     * Reset visited markers.
     * Call this periodically if query_id counter approaches wrap-around,
     * or before reusing query IDs.
     *
     * After calling reset_visited(), the next query_id should start from 1
     * (not 0) to ensure check_and_mark_visited works correctly.
     */
    void reset_visited() {
        for (auto& marker : visited_markers_) {
            marker->store(0, std::memory_order_relaxed);
        }
    }

    /**
     * Check if query counter needs reset to avoid wrap-around.
     * Returns true if the counter is within 1 billion of UINT64_MAX.
     *
     * Usage:
     *   if (graph.needs_visited_reset(query_counter.load())) {
     *       graph.reset_visited();
     *       query_counter.store(1);  // Start from 1, not 0
     *   }
     */
    static bool needs_visited_reset(uint64_t current_query_id) {
        constexpr uint64_t RESET_THRESHOLD = std::numeric_limits<uint64_t>::max() - 1'000'000'000ULL;
        return current_query_id >= RESET_THRESHOLD;
    }

    /**
     * NSW + Flash Thread-safe: Add a neighbor to a node.
     * Uses per-node spinlock and copies neighbor code.
     *
     * @param id        Node ID
     * @param neighbor  Neighbor to add
     * @return          true if added (or already exists), false if full
     */
    bool add_neighbor_safe(NodeId id, NodeId neighbor) {
        auto& block = neighbor_blocks_[id];

        Spinlock::Guard guard(block.lock);

        // Read with relaxed since we hold the lock
        size_t current_count = block.count.load(std::memory_order_relaxed);

        // Check for duplicate to prevent multiple threads adding same neighbor
        for (size_t i = 0; i < current_count; ++i) {
            if (block.ids[i] == neighbor) {
                return true;  // Already exists
            }
        }

        if (current_count >= params_.M) {
            return false;
        }

        block.ids[current_count] = neighbor;
        block.set_neighbor_code(current_count, codes_[neighbor]);

        // Release semantics: ensure writes visible before count update
        // The spinlock unlock provides release, but we use release here
        // to ensure correct ordering even for lock-free readers
        block.count.store(static_cast<uint8_t>(current_count + 1), std::memory_order_release);

        return true;
    }

    /**
     * NSW + Flash Thread-safe: Add neighbor with cached distance.
     *
     * CRITICAL OPTIMIZATION: Uses cached distances in block for O(1) pruning.
     * Also copies neighbor code for Flash cache locality.
     *
     * THREAD SAFETY: Uses per-node spinlock. Count update uses release
     * semantics to ensure lock-free readers see consistent data.
     *
     * @param id        Node ID
     * @param neighbor  Neighbor to add
     * @param dist      Distance to neighbor (lower = better)
     * @return          true if added or replaced worse neighbor, false if rejected
     */
    bool add_neighbor_with_dist_safe(NodeId id, NodeId neighbor, float dist) {
        auto& block = neighbor_blocks_[id];

        Spinlock::Guard guard(block.lock);

        // Read with relaxed since we hold the lock
        size_t current_count = block.count.load(std::memory_order_relaxed);

        // Check for duplicate
        for (size_t i = 0; i < current_count; ++i) {
            if (block.ids[i] == neighbor) {
                return true;  // Already exists
            }
        }

        // Case 1: Not full - just append
        if (current_count < params_.M) {
            block.ids[current_count] = neighbor;
            block.set_neighbor_code(current_count, codes_[neighbor]);
            block.distances[current_count] = dist;
            // Release semantics for lock-free readers
            block.count.store(static_cast<uint8_t>(current_count + 1), std::memory_order_release);
            return true;
        }

        // Case 2: Full - find worst neighbor using CACHED distances
        size_t worst_idx = 0;
        float worst_dist = block.distances[0];

        for (size_t i = 1; i < current_count; ++i) {
            if (block.distances[i] > worst_dist) {
                worst_dist = block.distances[i];
                worst_idx = i;
            }
        }

        // Replace worst if new neighbor is better
        // Note: No count change, so no release store needed
        if (dist < worst_dist) {
            block.ids[worst_idx] = neighbor;
            block.set_neighbor_code(worst_idx, codes_[neighbor]);
            block.distances[worst_idx] = dist;
            return true;
        }

        return false;  // New neighbor is worse than all existing
    }

    /**
     * NSW + Flash Thread-safe: Set neighbors for a node.
     * Copies all neighbor codes.
     *
     * @param id         Node ID
     * @param neighbors  New neighbor list
     */
    void set_neighbors_safe(NodeId id, const std::vector<NodeId>& neighbors) {
        auto& block = neighbor_blocks_[id];

        Spinlock::Guard guard(block.lock);

        size_t new_count = std::min(neighbors.size(), params_.M);

        // Copy neighbors and their codes
        for (size_t i = 0; i < new_count; ++i) {
            block.ids[i] = neighbors[i];
            block.set_neighbor_code(i, codes_[neighbors[i]]);
        }

        // Clear remaining slots
        for (size_t i = new_count; i < params_.M; ++i) {
            block.ids[i] = INVALID_NODE;
        }

        // Release semantics for lock-free readers
        block.count.store(static_cast<uint8_t>(new_count), std::memory_order_release);
    }

    // NOTE: set_entry_point_safe removed for NSW flatten (no hierarchy)

    /**
     * NSW + Flash Thread-safe: Add a link with automatic pruning if full.
     *
     * This performs the ENTIRE read-modify-write transaction under ONE lock
     * to prevent lost updates. The caller provides a distance function to
     * compute distances for pruning decisions.
     *
     * @param id           Node ID to modify
     * @param new_neighbor Neighbor to add
     * @param new_dist     Distance to new neighbor (lower = better)
     * @param dist_func    Function(NodeId) -> float to compute distance to existing neighbors
     */
    template<typename DistFunc>
    void add_link_with_pruning(NodeId id, NodeId new_neighbor,
                               float new_dist, DistFunc dist_func) {
        auto& block = neighbor_blocks_[id];

        Spinlock::Guard guard(block.lock);

        // Read with relaxed since we hold the lock
        size_t current_count = block.count.load(std::memory_order_relaxed);

        // Case 1: Not full - just add
        if (current_count < params_.M) {
            block.ids[current_count] = new_neighbor;
            block.set_neighbor_code(current_count, codes_[new_neighbor]);
            // Release semantics for lock-free readers
            block.count.store(static_cast<uint8_t>(current_count + 1), std::memory_order_release);
            return;
        }

        // Case 2: Full - need to potentially prune
        // Build list of (distance, neighbor) pairs including new candidate
        std::vector<std::pair<float, NodeId>> candidates;
        candidates.reserve(current_count + 1);

        for (size_t i = 0; i < current_count; ++i) {
            NodeId neighbor = block.ids[i];
            if (neighbor != INVALID_NODE) {
                float dist = dist_func(neighbor);
                candidates.emplace_back(dist, neighbor);
            }
        }
        candidates.emplace_back(new_dist, new_neighbor);

        // Sort by distance (lower is better)
        std::sort(candidates.begin(), candidates.end());

        // Keep best M neighbors (with their codes)
        size_t keep = std::min(candidates.size(), params_.M);
        for (size_t i = 0; i < keep; ++i) {
            block.ids[i] = candidates[i].second;
            block.set_neighbor_code(i, codes_[candidates[i].second]);
        }
        for (size_t i = keep; i < params_.M; ++i) {
            block.ids[i] = INVALID_NODE;
        }
        // Release semantics for lock-free readers
        block.count.store(static_cast<uint8_t>(keep), std::memory_order_release);
    }

    /**
     * Pre-allocate space for nodes (used in batch insertion).
     * MUST be called from a single thread before parallel insertion.
     *
     * @param count  Total number of nodes after reservation
     */
    void reserve_nodes(size_t count) {
        codes_.reserve(count);
        neighbor_blocks_.reserve(count);
        visited_markers_.reserve(count);
    }

    /**
     * Ingest a pre-built k-NN graph (for GPU construction).
     *
     * Clears any existing graph and populates from the given neighbor lists.
     * Used by CAGRA-style GPU construction:
     *   1. GPU builds k-NN graph
     *   2. CPU encodes vectors
     *   3. This method ingests the graph structure
     *
     * @param codes           Pre-encoded CP codes [N]
     * @param neighbor_lists  Neighbor lists per node [N][variable size]
     */
    void ingest_knn_graph(const std::vector<CPCode<ComponentT, K>>& codes,
                          const std::vector<std::vector<NodeId>>& neighbor_lists) {
        size_t N = codes.size();
        assert(neighbor_lists.size() == N);

        // Clear and reserve
        codes_.clear();
        neighbor_blocks_.clear();
        visited_markers_.clear();

        codes_.reserve(N);
        neighbor_blocks_.reserve(N);
        visited_markers_.reserve(N);

        // Add all nodes
        for (size_t i = 0; i < N; ++i) {
            codes_.push_back(codes[i]);
            neighbor_blocks_.push_back(Block{});
            visited_markers_.push_back(std::make_unique<std::atomic<uint64_t>>(0));
        }

        // Set neighbors (with code copies for Flash layout)
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < N; ++i) {
            const auto& neighbors = neighbor_lists[i];
            auto& block = neighbor_blocks_[i];

            size_t new_count = std::min(neighbors.size(), params_.M);
            for (size_t j = 0; j < new_count; ++j) {
                NodeId neighbor = neighbors[j];
                if (neighbor < N) {
                    block.ids[j] = neighbor;
                    block.set_neighbor_code(j, codes_[neighbor]);
                }
            }
            // Release semantics (though this is during single-phase init)
            block.count.store(static_cast<uint8_t>(new_count), std::memory_order_release);
        }
    }

    /**
     * Ingest from matrix format (N x k neighbor matrix).
     *
     * @param codes      Pre-encoded CP codes [N]
     * @param neighbors  Neighbor matrix [N x k], row-major
     * @param N          Number of nodes
     * @param k          Neighbors per node (can be > M, will truncate)
     */
    void ingest_knn_matrix(const std::vector<CPCode<ComponentT, K>>& codes,
                           const uint32_t* neighbors,
                           size_t N, size_t k) {
        assert(codes.size() == N);

        // Clear and reserve
        codes_.clear();
        neighbor_blocks_.clear();
        visited_markers_.clear();

        codes_.reserve(N);
        neighbor_blocks_.reserve(N);
        visited_markers_.reserve(N);

        // Add all nodes
        for (size_t i = 0; i < N; ++i) {
            codes_.push_back(codes[i]);
            neighbor_blocks_.push_back(Block{});
            visited_markers_.push_back(std::make_unique<std::atomic<uint64_t>>(0));
        }

        // Set neighbors
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < N; ++i) {
            auto& block = neighbor_blocks_[i];
            const uint32_t* row = neighbors + i * k;

            size_t new_count = 0;
            for (size_t j = 0; j < k && new_count < params_.M; ++j) {
                uint32_t neighbor = row[j];
                if (neighbor != UINT32_MAX && neighbor < N && neighbor != i) {
                    block.ids[new_count] = static_cast<NodeId>(neighbor);
                    block.set_neighbor_code(new_count, codes_[neighbor]);
                    ++new_count;
                }
            }
            // Release semantics (though this is during single-phase init)
            block.count.store(static_cast<uint8_t>(new_count), std::memory_order_release);
        }
    }

    /**
     * Clear the graph (remove all nodes).
     */
    void clear() {
        codes_.clear();
        neighbor_blocks_.clear();
        visited_markers_.clear();
    }

    /**
     * Flash: Prefetch a neighbor block for upcoming access.
     * Call this before processing a node's neighbors.
     *
     * SoA LAYOUT: codes_transposed is organized as [K][FLASH_MAX_M].
     * We prefetch the IDs, the beginning of codes_transposed (where all component
     * 0 values for neighbors 0-63 are), and the distances array.
     */
    void prefetch_neighbor_block(NodeId id) const {
        if (id < neighbor_blocks_.size()) {
            const auto& block = neighbor_blocks_[id];
            // Prefetch the beginning of the block (IDs)
            __builtin_prefetch(&block, 0, 3);  // Read, high temporal locality
            // Prefetch start of codes_transposed (component 0 for all neighbors)
            __builtin_prefetch(&block.codes_transposed[0][0], 0, 3);
            // Prefetch middle of codes_transposed (around component K/2)
            __builtin_prefetch(&block.codes_transposed[K/2][0], 0, 3);
            // Prefetch distances array
            __builtin_prefetch(&block.distances[0], 0, 3);
        }
    }

    // =========================================================================
    // Serialization API
    // =========================================================================

    /// Magic number for format validation
    static constexpr uint32_t GRAPH_MAGIC = 0x43504857;  // "CPHW"
    static constexpr uint32_t GRAPH_VERSION = 1;

    /**
     * Save graph to binary stream.
     *
     * Format:
     * - Header: magic (4), version (4), N (8), dim (8), M (8), K (8)
     * - Codes: N * K * sizeof(ComponentT) bytes
     * - NeighborBlocks: N * serialized block data
     *
     * @param os  Output stream (binary mode)
     */
    void save(std::ostream& os) const {
        // Write header
        uint32_t magic = GRAPH_MAGIC;
        uint32_t version = GRAPH_VERSION;
        uint64_t n = codes_.size();
        uint64_t dim = params_.dim;
        uint64_t M = params_.M;
        uint64_t k_val = K;

        os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        os.write(reinterpret_cast<const char*>(&version), sizeof(version));
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        os.write(reinterpret_cast<const char*>(&M), sizeof(M));
        os.write(reinterpret_cast<const char*>(&k_val), sizeof(k_val));

        // Write codes
        for (const auto& code : codes_) {
            os.write(reinterpret_cast<const char*>(code.components), sizeof(code.components));
            os.write(reinterpret_cast<const char*>(code.magnitudes), sizeof(code.magnitudes));
        }

        // Write neighbor blocks (only essential data, not atomics/locks)
        for (size_t i = 0; i < neighbor_blocks_.size(); ++i) {
            const auto& block = neighbor_blocks_[i];
            uint8_t count = block.count.load(std::memory_order_relaxed);

            os.write(reinterpret_cast<const char*>(&count), sizeof(count));
            os.write(reinterpret_cast<const char*>(block.ids), sizeof(block.ids));
            os.write(reinterpret_cast<const char*>(block.codes_transposed), sizeof(block.codes_transposed));
            os.write(reinterpret_cast<const char*>(block.magnitudes_transposed), sizeof(block.magnitudes_transposed));
            os.write(reinterpret_cast<const char*>(block.distances), sizeof(block.distances));
        }

        if (!os) {
            throw std::runtime_error("Failed to write graph to stream");
        }
    }

    /**
     * Load graph from binary stream.
     *
     * @param is  Input stream (binary mode)
     */
    void load(std::istream& is) {
        // Read and validate header
        uint32_t magic, version;
        uint64_t n, dim, M, k_val;

        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&M), sizeof(M));
        is.read(reinterpret_cast<char*>(&k_val), sizeof(k_val));

        if (magic != GRAPH_MAGIC) {
            throw std::runtime_error("Invalid graph file: bad magic number");
        }
        if (version != GRAPH_VERSION) {
            throw std::runtime_error("Unsupported graph version: " + std::to_string(version));
        }
        if (k_val != K) {
            throw std::runtime_error("K mismatch: file has K=" + std::to_string(k_val) +
                                     ", expected K=" + std::to_string(K));
        }

        // Update params
        params_.dim = dim;
        params_.M = M;
        params_.k = K;

        // Read codes
        codes_.resize(n);
        for (auto& code : codes_) {
            is.read(reinterpret_cast<char*>(code.components), sizeof(code.components));
            is.read(reinterpret_cast<char*>(code.magnitudes), sizeof(code.magnitudes));
        }

        // Read neighbor blocks
        neighbor_blocks_.resize(n);
        visited_markers_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            auto& block = neighbor_blocks_[i];
            uint8_t count;

            is.read(reinterpret_cast<char*>(&count), sizeof(count));
            is.read(reinterpret_cast<char*>(block.ids), sizeof(block.ids));
            is.read(reinterpret_cast<char*>(block.codes_transposed), sizeof(block.codes_transposed));
            is.read(reinterpret_cast<char*>(block.magnitudes_transposed), sizeof(block.magnitudes_transposed));
            is.read(reinterpret_cast<char*>(block.distances), sizeof(block.distances));

            block.count.store(count, std::memory_order_relaxed);
            visited_markers_[i] = std::make_unique<std::atomic<uint64_t>>(0);
        }

        if (!is) {
            throw std::runtime_error("Failed to read graph from stream");
        }
    }

private:
    CPHNSWParams params_;

    // Flash memory layout storage
    std::vector<CPCode<ComponentT, K>> codes_;              // Node's own code
    std::vector<Block> neighbor_blocks_;                     // NeighborBlock per node
    mutable std::vector<std::unique_ptr<std::atomic<uint64_t>>> visited_markers_;
};

// Common graph types
using FlatHNSWGraph8 = FlatHNSWGraph<uint8_t, 16>;
using FlatHNSWGraph16 = FlatHNSWGraph<uint16_t, 16>;
using FlatHNSWGraph32 = FlatHNSWGraph<uint8_t, 32>;  // K=32 for Flash optimization

}  // namespace cphnsw
