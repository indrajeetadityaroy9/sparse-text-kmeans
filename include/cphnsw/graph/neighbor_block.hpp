#pragma once

#include "../core/codes.hpp"
#include "../core/memory.hpp"
#include <atomic>
#include <algorithm>
#include <limits>
#include <cstring>

// For _mm_pause() on x86
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define CPHNSW_HAS_PAUSE 1
#else
#define CPHNSW_HAS_PAUSE 0
#endif

namespace cphnsw {

// ============================================================================
// Constants
// ============================================================================

/// Maximum neighbors per node (compile-time for memory layout)
constexpr size_t MAX_NEIGHBORS = 64;

// Use existing NodeId and INVALID_NODE from types.hpp if available
#ifndef CPHNSW_NODE_TYPES_DEFINED
#define CPHNSW_NODE_TYPES_DEFINED
/// Node ID type
using NodeId = uint32_t;
/// Invalid node ID sentinel
constexpr NodeId INVALID_NODE = 0xFFFFFFFF;
#endif

// ============================================================================
// Spinlock: Per-node synchronization
// ============================================================================

/**
 * Spinlock: Lightweight lock for fine-grained per-node synchronization.
 *
 * Uses _mm_pause() hint for Hyperthreading friendliness.
 * Size: 1 byte (fits in struct padding).
 */
class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if CPHNSW_HAS_PAUSE
            _mm_pause();
#endif
        }
    }

    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

    // RAII guard
    class Guard {
        Spinlock& lock_;
    public:
        explicit Guard(Spinlock& lock) : lock_(lock) { lock_.lock(); }
        ~Guard() { lock_.unlock(); }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };
};

static_assert(sizeof(Spinlock) == 1, "Spinlock must be 1 byte");

// ============================================================================
// NeighborBlock: Unified cache-local neighbor storage
// ============================================================================

/**
 * NeighborBlock: Cache-optimized storage for a node's neighbors.
 *
 * UNIFIED DESIGN: Templated on CodeT, automatically adapts storage layout
 * based on whether the code has residual bits or not.
 *
 * FLASH OPTIMIZATION: Stores COPIES of neighbor codes alongside neighbor IDs.
 * This eliminates pointer-chasing during search.
 *
 * SoA LAYOUT: Codes are stored in transposed (Structure-of-Arrays) format
 * to enable SIMD batch distance computation.
 *
 * Memory Layout:
 *   - ids[MAX_NEIGHBORS]: Neighbor node IDs
 *   - codes: SoA transposed code storage (adapts to CodeT)
 *   - distances[MAX_NEIGHBORS]: Cached edge distances
 *   - count: Actual neighbor count (0 to MAX_NEIGHBORS)
 *   - lock: Per-node spinlock for thread-safe updates
 *
 * @tparam CodeT The code type (ResidualCode<K, R>)
 */
template <typename CodeT>
struct alignas(CACHE_LINE_SIZE) UnifiedNeighborBlock {
    using CodeType = CodeT;
    using SoALayout = CodeSoALayout<CodeT, MAX_NEIGHBORS>;

    static constexpr size_t PRIMARY_BITS = CodeT::PRIMARY_BITS;
    static constexpr size_t RESIDUAL_BITS = CodeT::RESIDUAL_BITS;
    static constexpr bool HAS_RESIDUAL = CodeT::HAS_RESIDUAL;

    // ========================================================================
    // Storage
    // ========================================================================

    /// Neighbor node IDs (INVALID_NODE if slot unused)
    NodeId ids[MAX_NEIGHBORS];

    /// Transposed code storage for SIMD batch processing
    SoALayout codes;

    /// Cached edge distances (negative dot product for min-heap)
    float distances[MAX_NEIGHBORS];

    /// Actual number of neighbors
    uint8_t count;

    /// Per-node spinlock for thread-safe updates
    mutable Spinlock lock;

    // ========================================================================
    // Initialization
    // ========================================================================

    UnifiedNeighborBlock() : count(0) {
        clear();
    }

    void clear() {
        count = 0;
        std::fill(std::begin(ids), std::end(ids), INVALID_NODE);
        codes.clear();
        std::fill(std::begin(distances), std::end(distances),
                  std::numeric_limits<float>::max());
    }

    // Copy constructor: copy data, create new lock
    UnifiedNeighborBlock(const UnifiedNeighborBlock& other) : count(other.count) {
        std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
        codes = other.codes;
        std::copy(std::begin(other.distances), std::end(other.distances),
                  std::begin(distances));
    }

    UnifiedNeighborBlock& operator=(const UnifiedNeighborBlock& other) {
        if (this != &other) {
            count = other.count;
            std::copy(std::begin(other.ids), std::end(other.ids), std::begin(ids));
            codes = other.codes;
            std::copy(std::begin(other.distances), std::end(other.distances),
                      std::begin(distances));
        }
        return *this;
    }

    // ========================================================================
    // Neighbor Management
    // ========================================================================

    /**
     * Add a neighbor (NOT thread-safe - use add_safe for concurrent access).
     *
     * @param id Neighbor node ID
     * @param code Neighbor's binary code
     * @param distance Distance to neighbor
     * @return true if added, false if block is full
     */
    bool add(NodeId id, const CodeT& code, float distance) {
        if (count >= MAX_NEIGHBORS) {
            return false;
        }

        size_t idx = count;
        ids[idx] = id;
        codes.store(idx, code);
        distances[idx] = distance;
        ++count;

        return true;
    }

    /**
     * Add a neighbor (thread-safe with spinlock).
     */
    bool add_safe(NodeId id, const CodeT& code, float distance) {
        Spinlock::Guard guard(lock);
        return add(id, code, distance);
    }

    /**
     * Try to add a neighbor if it improves the set.
     *
     * If the block is full, replaces the worst (highest distance) neighbor
     * if the new distance is better.
     *
     * @return true if the neighbor was added or replaced an existing one
     */
    bool try_add(NodeId id, const CodeT& code, float distance) {
        if (count < MAX_NEIGHBORS) {
            return add(id, code, distance);
        }

        // Find worst neighbor
        size_t worst_idx = 0;
        float worst_dist = distances[0];
        for (size_t i = 1; i < count; ++i) {
            if (distances[i] > worst_dist) {
                worst_dist = distances[i];
                worst_idx = i;
            }
        }

        // Replace if better
        if (distance < worst_dist) {
            ids[worst_idx] = id;
            codes.store(worst_idx, code);
            distances[worst_idx] = distance;
            return true;
        }

        return false;
    }

    /**
     * Thread-safe try_add.
     */
    bool try_add_safe(NodeId id, const CodeT& code, float distance) {
        Spinlock::Guard guard(lock);
        return try_add(id, code, distance);
    }

    /**
     * Check if a node ID is already a neighbor.
     */
    bool contains(NodeId id) const {
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] == id) return true;
        }
        return false;
    }

    /**
     * Remove a neighbor by ID.
     * @return true if found and removed
     */
    bool remove(NodeId id) {
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] == id) {
                // Swap with last and decrement count
                size_t last = count - 1;
                if (i != last) {
                    ids[i] = ids[last];
                    CodeT temp;
                    codes.load(last, temp);
                    codes.store(i, temp);
                    distances[i] = distances[last];
                }
                ids[last] = INVALID_NODE;
                distances[last] = std::numeric_limits<float>::max();
                --count;
                return true;
            }
        }
        return false;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    size_t size() const { return count; }
    bool empty() const { return count == 0; }
    bool full() const { return count >= MAX_NEIGHBORS; }

    NodeId get_id(size_t idx) const {
        return (idx < count) ? ids[idx] : INVALID_NODE;
    }

    float get_distance(size_t idx) const {
        return (idx < count) ? distances[idx] : std::numeric_limits<float>::max();
    }

    void get_code(size_t idx, CodeT& out_code) const {
        if (idx < count) {
            codes.load(idx, out_code);
        }
    }

    /// Get pointer to SoA layout for batch distance computation
    const SoALayout& get_soa_layout() const { return codes; }
};

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

// Phase 1: Pure binary codes
using UnifiedNeighborBlock32 = UnifiedNeighborBlock<ResidualCode<32, 0>>;
using UnifiedNeighborBlock64 = UnifiedNeighborBlock<ResidualCode<64, 0>>;

// Phase 2: With residual
using UnifiedNeighborBlock64_32 = UnifiedNeighborBlock<ResidualCode<64, 32>>;
using UnifiedNeighborBlock32_16 = UnifiedNeighborBlock<ResidualCode<32, 16>>;

}  // namespace cphnsw
