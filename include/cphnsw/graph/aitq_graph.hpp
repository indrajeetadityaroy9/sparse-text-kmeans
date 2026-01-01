#pragma once

#include "../core/types.hpp"
#include "../quantizer/aitq_quantizer.hpp"
#include "priority_queue.hpp"
#include <vector>
#include <array>
#include <atomic>
#include <random>
#include <cstring>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cphnsw {

/// Maximum neighbors per node (matches FLASH_MAX_M)
constexpr size_t AITQ_MAX_M = 64;

// ============================================================================
// A-ITQ NeighborBlock: Cache-optimized storage for A-ITQ codes
// ============================================================================

/**
 * AITQNeighborBlock: Flash-style memory layout for A-ITQ binary codes.
 *
 * Unlike CPCode which stores (index, sign) per rotation, A-ITQ stores
 * K bits per code. The transposed layout enables SIMD batch distance:
 *
 *   bits_transposed[byte_idx][neighbor_idx] contains byte byte_idx of all neighbors
 *
 * This allows loading 16/32/64 bytes at once to process multiple neighbors.
 */
template <size_t K>
struct alignas(64) AITQNeighborBlock {
    static constexpr size_t K_BYTES = (K + 7) / 8;

    /// Neighbor node IDs (INVALID_NODE = unused slot)
    NodeId ids[AITQ_MAX_M];

    /// Transposed binary codes: bits_transposed[byte][neighbor]
    /// This layout enables SIMD: load 16 bytes = byte b for 16 neighbors
    uint8_t bits_transposed[K_BYTES][AITQ_MAX_M];

    /// Cached distances to node (for O(1) replacement decisions)
    float distances[AITQ_MAX_M];

    /// Actual neighbor count (0 to AITQ_MAX_M)
    uint8_t count;

    /// Per-node lock for thread-safe modifications
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;

    /// Padding to cache line
    uint8_t _padding[64 - ((sizeof(NodeId) * AITQ_MAX_M +
                            K_BYTES * AITQ_MAX_M +
                            sizeof(float) * AITQ_MAX_M +
                            sizeof(uint8_t) + sizeof(std::atomic_flag)) % 64)];

    AITQNeighborBlock() : count(0) {
        std::fill(std::begin(ids), std::end(ids), INVALID_NODE);
        std::memset(bits_transposed, 0, sizeof(bits_transposed));
        std::fill(std::begin(distances), std::end(distances),
                  std::numeric_limits<float>::max());
    }

    void lock() {
        while (lock_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }

    void unlock() {
        lock_.clear(std::memory_order_release);
    }

    /// Set neighbor code at index i (scatters to transposed layout)
    void set_neighbor_code(size_t i, const AITQCode<K>& code) {
        for (size_t b = 0; b < K_BYTES; ++b) {
            bits_transposed[b][i] = code.bits[b];
        }
    }

    /// Get neighbor code at index i (gathers from transposed layout)
    AITQCode<K> get_neighbor_code(size_t i) const {
        AITQCode<K> code;
        for (size_t b = 0; b < K_BYTES; ++b) {
            code.bits[b] = bits_transposed[b][i];
        }
        return code;
    }
};

// ============================================================================
// A-ITQ Graph: Specialized graph storage for A-ITQ binary codes
// ============================================================================

/**
 * AITQGraph: NSW graph optimized for A-ITQ binary codes.
 *
 * Key differences from FlatHNSWGraph:
 * - Stores AITQCode<K> instead of CPCode<ComponentT, K>
 * - Bit-packed storage (K/8 bytes per code vs K*2 bytes for CPCode)
 * - Specialized SIMD batch distance using bit operations
 */
template <size_t K>
class AITQGraph {
public:
    using Code = AITQCode<K>;
    using Query = AITQQuery<K>;
    using Block = AITQNeighborBlock<K>;
    static constexpr size_t K_BYTES = (K + 7) / 8;

    explicit AITQGraph(const CPHNSWParams& params)
        : params_(params), size_(0) {}

    /// Get parameters
    const CPHNSWParams& params() const { return params_; }

    /// Get number of nodes
    size_t size() const { return size_; }

    /// Check if empty
    bool empty() const { return size_ == 0; }

    /// Reserve space for N nodes
    void reserve_nodes(size_t N) {
        codes_.reserve(N);
        neighbor_blocks_.reserve(N);
        visited_markers_.reserve(N);
    }

    /**
     * Add a new node with the given code.
     * @return Node ID of the new node
     */
    NodeId add_node(const Code& code) {
        NodeId id = static_cast<NodeId>(size_++);
        codes_.push_back(code);
        neighbor_blocks_.emplace_back();
        visited_markers_.push_back(0);
        return id;
    }

    /// Get code for a node
    const Code& get_code(NodeId id) const {
        return codes_[id];
    }

    /// Get mutable code for a node
    Code& get_code_mut(NodeId id) {
        return codes_[id];
    }

    /// Get neighbor block for a node
    const Block& get_neighbor_block(NodeId id) const {
        return neighbor_blocks_[id];
    }

    /// Get mutable neighbor block
    Block& get_neighbor_block_mut(NodeId id) {
        return neighbor_blocks_[id];
    }

    /// Prefetch neighbor block for upcoming access
    void prefetch_neighbor_block(NodeId id) const {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(&neighbor_blocks_[id], 0, 3);
#endif
    }

    /// Get neighbors (IDs and count)
    std::pair<const NodeId*, size_t> get_neighbors(NodeId id) const {
        const auto& block = neighbor_blocks_[id];
        return {block.ids, block.count};
    }

    /**
     * Add neighbor with thread safety (no distance caching).
     * @return true if added, false if already present or full
     */
    bool add_neighbor_safe(NodeId node, NodeId neighbor) {
        auto& block = neighbor_blocks_[node];
        block.lock();

        // Check if already present
        for (size_t i = 0; i < block.count; ++i) {
            if (block.ids[i] == neighbor) {
                block.unlock();
                return false;
            }
        }

        // Check if full
        if (block.count >= params_.M) {
            block.unlock();
            return false;
        }

        // Add neighbor
        size_t idx = block.count++;
        block.ids[idx] = neighbor;
        block.set_neighbor_code(idx, codes_[neighbor]);
        block.distances[idx] = 0.0f;  // Unknown distance

        block.unlock();
        return true;
    }

    /**
     * Add neighbor with distance, replacing worst if full.
     * Thread-safe with per-node locking.
     */
    bool add_neighbor_with_dist_safe(NodeId node, NodeId neighbor, float dist) {
        auto& block = neighbor_blocks_[node];
        block.lock();

        // Check if already present
        for (size_t i = 0; i < block.count; ++i) {
            if (block.ids[i] == neighbor) {
                // Update distance if better
                if (dist < block.distances[i]) {
                    block.distances[i] = dist;
                }
                block.unlock();
                return false;
            }
        }

        // If not full, just add
        if (block.count < params_.M) {
            size_t idx = block.count++;
            block.ids[idx] = neighbor;
            block.set_neighbor_code(idx, codes_[neighbor]);
            block.distances[idx] = dist;
            block.unlock();
            return true;
        }

        // Full: find worst (maximum distance) and replace if new is better
        size_t worst_idx = 0;
        float worst_dist = block.distances[0];
        for (size_t i = 1; i < block.count; ++i) {
            if (block.distances[i] > worst_dist) {
                worst_dist = block.distances[i];
                worst_idx = i;
            }
        }

        if (dist < worst_dist) {
            block.ids[worst_idx] = neighbor;
            block.set_neighbor_code(worst_idx, codes_[neighbor]);
            block.distances[worst_idx] = dist;
            block.unlock();
            return true;
        }

        block.unlock();
        return false;
    }

    /**
     * Check and mark node as visited for this query.
     * @return true if already visited, false if newly marked
     */
    bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
        uint64_t old = visited_markers_[id].exchange(query_id, std::memory_order_acq_rel);
        return old == query_id;
    }

    /**
     * Get random entry points for NSW search.
     */
    std::vector<NodeId> get_random_entry_points(size_t k, uint64_t seed) const {
        if (size_ == 0) return {};

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<NodeId> dist(0, static_cast<NodeId>(size_ - 1));

        std::vector<NodeId> entries;
        entries.reserve(k);

        for (size_t i = 0; i < k && i < size_; ++i) {
            entries.push_back(dist(rng));
        }

        return entries;
    }

    /**
     * Ingest pre-computed k-NN graph.
     */
    void ingest_knn_graph(
        const std::vector<Code>& codes,
        const std::vector<std::vector<NodeId>>& neighbor_lists) {

        size_t N = codes.size();
        codes_ = codes;
        neighbor_blocks_.resize(N);
        visited_markers_.resize(N);
        size_ = N;

        for (size_t i = 0; i < N; ++i) {
            visited_markers_[i] = 0;

            auto& block = neighbor_blocks_[i];
            block.count = 0;

            const auto& neighbors = neighbor_lists[i];
            for (size_t j = 0; j < std::min(neighbors.size(), params_.M); ++j) {
                NodeId neighbor = neighbors[j];
                if (neighbor < N && neighbor != i) {
                    size_t idx = block.count++;
                    block.ids[idx] = neighbor;
                    block.set_neighbor_code(idx, codes[neighbor]);
                }
            }
        }
    }

private:
    CPHNSWParams params_;
    size_t size_;
    std::vector<Code> codes_;
    std::vector<Block> neighbor_blocks_;
    mutable std::vector<std::atomic<uint64_t>> visited_markers_;
};

// ============================================================================
// A-ITQ Batch Distance Computation
// ============================================================================

/**
 * Compute batch A-ITQ distances using transposed layout.
 *
 * For each neighbor, computes:
 *   distance = -Σᵢ query.proj[i] × (2×bit[i] - 1)
 *
 * @param query            A-ITQ query with projections
 * @param bits_transposed  Transposed bits [K_BYTES][AITQ_MAX_M]
 * @param num_neighbors    Number of neighbors to process
 * @param out_distances    Output distances (must have num_neighbors elements)
 */
template <size_t K>
inline void aitq_batch_distance_soa(
    const AITQQuery<K>& query,
    const uint8_t bits_transposed[(K + 7) / 8][AITQ_MAX_M],
    size_t num_neighbors,
    float* out_distances) {

    constexpr size_t K_BYTES = (K + 7) / 8;

    // Initialize scores to zero
    std::fill(out_distances, out_distances + num_neighbors, 0.0f);

    // Process each bit position
    for (size_t byte_idx = 0; byte_idx < K_BYTES; ++byte_idx) {
        for (size_t bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte) {
            size_t bit_idx = byte_idx * 8 + bit_in_byte;
            if (bit_idx >= K) break;

            float proj_val = query.proj[bit_idx];

            for (size_t n = 0; n < num_neighbors; ++n) {
                uint8_t byte = bits_transposed[byte_idx][n];
                bool bit_set = (byte >> bit_in_byte) & 1;

                // If bit is 1: add proj, else subtract
                out_distances[n] += bit_set ? proj_val : -proj_val;
            }
        }
    }

    // Negate for distance semantics (lower = more similar)
    for (size_t n = 0; n < num_neighbors; ++n) {
        out_distances[n] = -out_distances[n];
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)

/**
 * AVX-512 optimized batch A-ITQ distance for 16 neighbors at once.
 */
template <size_t K>
inline void aitq_batch_distance_avx512_16(
    const AITQQuery<K>& query,
    const uint8_t bits_transposed[(K + 7) / 8][AITQ_MAX_M],
    size_t start_idx,
    float* out_distances) {

    constexpr size_t K_BYTES = (K + 7) / 8;
    __m512 scores = _mm512_setzero_ps();

    for (size_t byte_idx = 0; byte_idx < K_BYTES; ++byte_idx) {
        // Load 16 bytes (one per neighbor)
        __m128i bytes = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&bits_transposed[byte_idx][start_idx]));

        for (size_t bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte) {
            size_t bit_idx = byte_idx * 8 + bit_in_byte;
            if (bit_idx >= K) break;

            // Extract bit for all 16 neighbors
            __m128i bit_mask = _mm_set1_epi8(1 << bit_in_byte);
            __m128i bits = _mm_and_si128(bytes, bit_mask);
            __m128i is_set = _mm_cmpeq_epi8(bits, bit_mask);
            __mmask16 mask = static_cast<__mmask16>(_mm_movemask_epi8(is_set));

            // Load projection value
            float proj_val = query.proj[bit_idx];
            __m512 proj = _mm512_set1_ps(proj_val);
            __m512 neg_proj = _mm512_set1_ps(-proj_val);

            // Blend: +proj where bit=1, -proj where bit=0
            __m512 contributions = _mm512_mask_blend_ps(mask, neg_proj, proj);
            scores = _mm512_add_ps(scores, contributions);
        }
    }

    // Negate and store
    __m512 neg_scores = _mm512_sub_ps(_mm512_setzero_ps(), scores);
    _mm512_storeu_ps(out_distances, neg_scores);
}

#endif  // AVX-512

}  // namespace cphnsw
