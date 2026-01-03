#pragma once

#include "../core/memory.hpp"
#include "neighbor_block.hpp"
#include <atomic>
#include <memory>

namespace cphnsw {

// ============================================================================
// VisitationTable: Thread-safe epoch-based visited tracking
// ============================================================================

/**
 * VisitationTable: Efficiently tracks visited nodes during graph traversal.
 *
 * EPOCH-BASED DESIGN: Instead of clearing the entire table between queries,
 * we increment a global epoch. A node is "visited" if its stored epoch matches
 * the current epoch. This makes query start O(1) instead of O(N).
 *
 * THREAD SAFETY: Uses atomic compare-and-swap for concurrent marking.
 * Each query should use a unique query_id (typically: thread_id * max_queries + query_num).
 *
 * NOTE: Uses unique_ptr instead of vector because std::atomic is not
 * copy/move constructible, making vector resize impossible.
 *
 * Usage:
 *   VisitationTable table(num_nodes);
 *   uint64_t query_id = table.new_query();  // Get fresh epoch
 *   if (table.check_and_mark(node_id, query_id)) {
 *       // Already visited
 *   } else {
 *       // First visit - process this node
 *   }
 */
class VisitationTable {
public:
    explicit VisitationTable(size_t capacity)
        : capacity_(capacity), current_epoch_(0) {
        epochs_ = std::make_unique<std::atomic<uint64_t>[]>(capacity);
        for (size_t i = 0; i < capacity_; ++i) {
            epochs_[i].store(0, std::memory_order_relaxed);
        }
    }

    // Non-copyable due to atomics
    VisitationTable(const VisitationTable&) = delete;
    VisitationTable& operator=(const VisitationTable&) = delete;

    // Move constructor
    VisitationTable(VisitationTable&& other) noexcept
        : epochs_(std::move(other.epochs_)),
          capacity_(other.capacity_),
          current_epoch_(other.current_epoch_.load(std::memory_order_relaxed)) {
        other.capacity_ = 0;
    }

    /**
     * Start a new query and return its unique ID.
     * Thread-safe: uses atomic increment.
     */
    uint64_t new_query() {
        return current_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    /**
     * Check if a node was visited in this query, and mark it if not.
     *
     * @param node_id The node to check/mark
     * @param query_id The query epoch (from new_query())
     * @return true if already visited, false if this is the first visit
     */
    bool check_and_mark(NodeId node_id, uint64_t query_id) {
        if (node_id >= capacity_) return true;  // Out of bounds = "visited"

        uint64_t expected = epochs_[node_id].load(std::memory_order_relaxed);

        // Already visited in this query?
        if (expected == query_id) {
            return true;
        }

        // Try to mark as visited
        return !epochs_[node_id].compare_exchange_strong(
            expected, query_id,
            std::memory_order_relaxed,
            std::memory_order_relaxed);
    }

    /**
     * Check without marking (read-only).
     */
    bool is_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= capacity_) return true;
        return epochs_[node_id].load(std::memory_order_relaxed) == query_id;
    }

    /**
     * Resize the table (NOT thread-safe).
     * Allocates a new array and copies existing epochs.
     */
    void resize(size_t new_capacity) {
        if (new_capacity <= capacity_) return;

        auto new_epochs = std::make_unique<std::atomic<uint64_t>[]>(new_capacity);

        // Copy existing epochs
        for (size_t i = 0; i < capacity_; ++i) {
            new_epochs[i].store(epochs_[i].load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        }
        // Initialize new elements to 0
        for (size_t i = capacity_; i < new_capacity; ++i) {
            new_epochs[i].store(0, std::memory_order_relaxed);
        }

        epochs_ = std::move(new_epochs);
        capacity_ = new_capacity;
    }

    /**
     * Get current capacity.
     */
    size_t capacity() const { return capacity_; }

    /**
     * Force reset all epochs (useful for testing, NOT needed in normal use).
     */
    void reset() {
        for (size_t i = 0; i < capacity_; ++i) {
            epochs_[i].store(0, std::memory_order_relaxed);
        }
        current_epoch_.store(0, std::memory_order_relaxed);
    }

private:
    std::unique_ptr<std::atomic<uint64_t>[]> epochs_;
    size_t capacity_;
    std::atomic<uint64_t> current_epoch_;
};

// ============================================================================
// ThreadLocalVisitationTable: Per-thread table for parallel queries
// ============================================================================

/**
 * ThreadLocalVisitationTable: Avoids atomic contention by giving each thread
 * its own visited set. Useful when queries don't share state.
 *
 * This is faster than VisitationTable when queries are fully independent
 * but uses more memory (one table per thread).
 */
class ThreadLocalVisitationTable {
public:
    explicit ThreadLocalVisitationTable(size_t capacity)
        : epochs_(capacity, 0), current_epoch_(0) {}

    /**
     * Start a new query.
     */
    uint64_t new_query() {
        return ++current_epoch_;
    }

    /**
     * Check and mark (non-atomic, single-threaded use only).
     */
    bool check_and_mark(NodeId node_id, uint64_t query_id) {
        if (node_id >= epochs_.size()) return true;

        if (epochs_[node_id] == query_id) {
            return true;  // Already visited
        }

        epochs_[node_id] = query_id;
        return false;  // First visit
    }

    bool is_visited(NodeId node_id, uint64_t query_id) const {
        if (node_id >= epochs_.size()) return true;
        return epochs_[node_id] == query_id;
    }

    void resize(size_t new_capacity) {
        epochs_.resize(new_capacity, 0);
    }

    size_t capacity() const { return epochs_.size(); }

private:
    std::vector<uint64_t> epochs_;
    uint64_t current_epoch_;
};

}  // namespace cphnsw
