#pragma once

#include "../core/types.hpp"
#include <queue>
#include <vector>

namespace cphnsw {

/**
 * MinHeap: Extracts closest elements first.
 * Used for the candidate set (C) in SEARCH-LAYER.
 */
class MinHeap {
public:
    void push(NodeId id, HammingDist dist) {
        heap_.push({id, dist});
    }

    void push(const SearchResult& result) {
        heap_.push(result);
    }

    SearchResult top() const {
        return heap_.top();
    }

    void pop() {
        heap_.pop();
    }

    bool empty() const {
        return heap_.empty();
    }

    size_t size() const {
        return heap_.size();
    }

    void clear() {
        heap_ = std::priority_queue<SearchResult, std::vector<SearchResult>,
                                     std::greater<SearchResult>>();
    }

private:
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        std::greater<SearchResult>> heap_;
};

/**
 * MaxHeap: Extracts furthest elements first.
 * Used for the result set (W) in SEARCH-LAYER with bounded size.
 */
class MaxHeap {
public:
    void push(NodeId id, HammingDist dist) {
        heap_.push({id, dist});
    }

    void push(const SearchResult& result) {
        heap_.push(result);
    }

    SearchResult top() const {
        return heap_.top();
    }

    void pop() {
        heap_.pop();
    }

    bool empty() const {
        return heap_.empty();
    }

    size_t size() const {
        return heap_.size();
    }

    void clear() {
        heap_ = std::priority_queue<SearchResult>();
    }

    /**
     * Bounded insert: only add if closer than current max.
     * Maintains heap size <= max_size.
     *
     * @param id        Node ID
     * @param dist      Distance
     * @param max_size  Maximum heap size
     * @return          true if inserted
     */
    bool try_push(NodeId id, HammingDist dist, size_t max_size) {
        if (heap_.size() < max_size) {
            heap_.push({id, dist});
            return true;
        }
        if (dist < heap_.top().distance) {
            heap_.pop();
            heap_.push({id, dist});
            return true;
        }
        return false;
    }

    /**
     * Extract all elements as a sorted vector (ascending by distance).
     */
    std::vector<SearchResult> extract_sorted() {
        std::vector<SearchResult> results;
        results.reserve(heap_.size());

        while (!heap_.empty()) {
            results.push_back(heap_.top());
            heap_.pop();
        }

        // Results are in descending order, reverse for ascending
        std::reverse(results.begin(), results.end());
        return results;
    }

    /**
     * Convert to vector without clearing (copies elements).
     */
    std::vector<SearchResult> to_vector() const {
        std::vector<SearchResult> results;
        results.reserve(heap_.size());

        // Copy the heap container
        MaxHeap copy = *this;
        while (!copy.empty()) {
            results.push_back(copy.top());
            copy.pop();
        }

        std::reverse(results.begin(), results.end());
        return results;
    }

private:
    std::priority_queue<SearchResult> heap_;
};

}  // namespace cphnsw
