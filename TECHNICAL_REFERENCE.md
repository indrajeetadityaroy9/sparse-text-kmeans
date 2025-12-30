# CP-HNSW Technical Reference

A comprehensive technical reference for understanding the architecture, algorithms, and design decisions in the Cross-Polytope HNSW library.

## Table of Contents

1. [Overview and Mathematical Foundations](#1-overview-and-mathematical-foundations)
2. [Core Type System](#2-core-type-system)
3. [Cross-Polytope Encoding Pipeline](#3-cross-polytope-encoding-pipeline)
4. [HNSW Graph Structure](#4-hnsw-graph-structure)
5. [Search Algorithms](#5-search-algorithms)
6. [Insert Algorithms](#6-insert-algorithms)
7. [SIMD Optimizations](#7-simd-optimizations)
8. [CUDA GPU Acceleration](#8-cuda-gpu-acceleration)
9. [Performance Characteristics](#9-performance-characteristics)

---

## 1. Overview and Mathematical Foundations

### 1.1 Core Concept

CP-HNSW combines two state-of-the-art techniques for approximate nearest neighbor search:

- **Cross-Polytope LSH** (Andoni et al. 2015): Asymptotically optimal locality-sensitive hashing for angular distance
- **HNSW** (Malkov & Yashunin 2018): Hierarchical graph structure for logarithmic-time navigation

### 1.2 Cross-Polytope Hash Function

The cross-polytope hash maps a vector to the nearest vertex of a hypercross (unit vectors along coordinate axes):

```
h(x) = argmax_i |Rx[i]| × sign(Rx[i])
```

Where R is a random rotation matrix. The hash identifies:
1. Which coordinate axis the rotated vector aligns with most strongly
2. Whether the alignment is positive or negative

### 1.3 Pseudo-Random Rotation

Instead of storing a full d×d rotation matrix, CP-HNSW uses the rotation chain:

```
Ψ(x) = H D₃ H D₂ H D₁ x
```

Where:
- **H**: Hadamard matrix (computed via Fast Hadamard Transform in O(d log d))
- **D_i**: Diagonal matrix with random ±1 entries

This approximates a true Gaussian random rotation in O(d log d) time instead of O(d²).

### 1.4 Multiple Rotations for Robustness

CP-HNSW applies K independent rotations (typically K=16 or K=32), creating a K-byte code:

```
CPCode = [h₁(x), h₂(x), ..., h_K(x)]
```

Each component encodes both the dominant axis index and its sign.

### 1.5 Asymmetric Distance: The Key Innovation

Standard Hamming distance between codes has only K+1 discrete values, creating flat plateaus that prevent HNSW's greedy descent. CP-HNSW uses **asymmetric distance**:

```
AsymmetricDist(query, node) = Σᵣ sign_r × query.rotated_vecs[r][node.index_r]
```

The query stores full rotated vectors (expensive), while nodes store only codes (cheap). This provides continuous gradients for navigation while maintaining memory efficiency.

---

## 2. Core Type System

### 2.1 Fundamental Types

| Type | Definition | Purpose |
|------|------------|---------|
| `NodeId` | `uint32_t` | Node identifier (supports 4B nodes) |
| `LayerLevel` | `uint8_t` | HNSW layer level (0-255) |
| `HammingDist` | `uint16_t` | Discrete Hamming distance (0 to K) |
| `AsymmetricDist` | `float` | Continuous distance for navigation |
| `Float` | `float` | Vector component type |

### 2.2 CPCode Structure

```cpp
template <typename ComponentT, size_t K>
struct CPCode {
    std::array<ComponentT, K> components;

    // Encoding: (index << 1) | (is_negative ? 1 : 0)
    static constexpr ComponentT encode(size_t index, bool is_negative);
    static constexpr size_t decode_index(ComponentT c);
    static constexpr bool decode_sign_negative(ComponentT c);
};
```

**Memory Layout:**
- Each component packs index and sign into a single word
- Upper bits: dimension index (argmax in rotated space)
- LSB: sign bit (0=positive, 1=negative)

**Type Aliases:**

| Alias | ComponentT | K | Bytes/Code | Use Case |
|-------|-----------|---|------------|----------|
| `CPCode8` | `uint8_t` | 16 | 16 | d ≤ 128 |
| `CPCode16` | `uint16_t` | 16 | 32 | d > 128 |
| `CPCode32` | `uint8_t` | 32 | 32 | Higher precision |

### 2.3 CPQuery Structure

```cpp
template <typename ComponentT, size_t K>
struct CPQuery {
    CPCode<ComponentT, K> primary_code;
    std::array<std::vector<Float>, K> rotated_vecs;  // Full K rotated vectors
    std::array<Float, K> magnitudes;                  // |argmax| per rotation
    std::array<uint32_t, K> original_indices;         // Argmax indices
};
```

**Asymmetric Storage Design:**

| Component | Index (stored) | Query (computed) |
|-----------|----------------|------------------|
| Code | K bytes | K bytes |
| Rotated vectors | Not stored | K × padded_dim × 4B |
| Magnitudes | Not stored | K × 4B |

This asymmetry enables continuous gradient navigation while keeping index memory minimal.

### 2.4 Index Parameters

```cpp
struct CPHNSWParams {
    size_t dim;              // Original dimension
    size_t padded_dim;       // Next power of 2 ≥ dim (for FHT)
    size_t k = 16;           // Number of rotations
    size_t M = 16;           // Max edges per node (layers > 0)
    size_t M_max0 = 32;      // Max edges at layer 0 (≈2×M)
    size_t ef_construction = 100;  // Search width during construction
    double m_L = 0.0;        // Level multiplier: 1/ln(M)
    bool keep_pruned = true; // Keep discarded edges for robustness
    uint64_t seed = 42;      // RNG seed
};
```

**Key Formulas:**

1. **Level multiplier:** `m_L = 1/ln(M)` (controls layer distribution)
2. **Level probability:** `P(level=L) ∝ exp(-L/m_L)`
3. **Padded dimension:** smallest power-of-2 ≥ dim

---

## 3. Cross-Polytope Encoding Pipeline

### 3.1 Fast Hadamard Transform (FHT)

The Hadamard matrix is defined recursively:
```
H₁ = [1]
H₂ₙ = [[Hₙ,  Hₙ],
       [Hₙ, -Hₙ]]
```

**Butterfly Algorithm:**

```cpp
void fht_scalar(Float* vec, size_t len) {
    for (size_t h = 1; h < len; h *= 2) {           // log(len) stages
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; ++j) {
                Float x = vec[j];
                Float y = vec[j + h];
                vec[j] = x + y;      // Butterfly operation
                vec[j + h] = x - y;
            }
        }
    }
}
```

**Complexity:** O(n log n) time, O(1) auxiliary space

### 3.2 Rotation Chain

```cpp
void RotationChain::apply(Float* vec, size_t rotation_index) const {
    apply_diagonal(vec, signs_[rotation_index][0]);  // D₁
    fht(vec, padded_dim_);                           // H
    apply_diagonal(vec, signs_[rotation_index][1]);  // D₂
    fht(vec, padded_dim_);                           // H
    apply_diagonal(vec, signs_[rotation_index][2]);  // D₃
    fht(vec, padded_dim_);                           // H
}
```

**Sign Matrix Generation:**
- Uses MT19937-64 seeded deterministically
- Each rotation gets 3 independent sign vectors
- Memory: K × 3 × padded_dim bytes

### 3.3 Argmax Extraction

```cpp
void find_argmax_abs(const Float* buffer, size_t size,
                     size_t& out_idx, Float& out_abs) {
    // SIMD-optimized two-pass algorithm:
    // Pass 1: Find global maximum using SIMD reduction
    // Pass 2: Find index using SIMD comparison
}
```

### 3.4 Complete Encoding Pipeline

```cpp
CPCode<ComponentT, K> CPEncoder::encode(const Float* vec) const {
    CPCode<ComponentT, K> code;

    for (size_t r = 0; r < K; ++r) {
        rotation_chain_.apply_copy(vec, buffer_.data(), r);

        size_t max_idx;
        Float max_abs;
        find_argmax_abs(buffer_.data(), padded_dim_, max_idx, max_abs);

        bool is_negative = (buffer_[max_idx] < 0);
        code.components[r] = CPCode<ComponentT, K>::encode(max_idx, is_negative);
    }

    return code;
}
```

### 3.5 Multiprobe Sequence Generation

Multiprobe expands search by generating alternative codes ranked by collision probability:

```
log P(collision) ∝ -(|x_max| - |x_alternative|)²
```

**Algorithm (Dijkstra-like exploration):**

1. Initialize with primary code (log_prob = 0)
2. For each rotation, create alternative using rank-1 coordinate
3. Greedy extraction: always pop highest probability probe
4. Expand by incrementing modification rank at each rotation

---

## 4. HNSW Graph Structure

### 4.1 Flat Memory Layout

Standard HNSW uses `vector<vector<NodeId>>` which has high overhead (~384 bytes/node for 16 layers). CP-HNSW uses contiguous arrays:

```
links_pool: [Node0_L0 | Node0_L1 | Node1_L0 | Node1_L1 | ...]
nodes:      [CompactNode0, CompactNode1, ...]
codes:      [CPCode0, CPCode1, ...]
```

### 4.2 CompactNode Structure

```cpp
struct alignas(8) CompactNode {
    uint32_t link_offset;       // Byte offset into links_pool
    uint8_t max_layer;          // Highest layer for this node
    uint8_t link_counts[16];    // Actual neighbor count per layer
    uint8_t _padding[3];        // Align to 24 bytes
};
static_assert(sizeof(CompactNode) == 24);
```

**Memory Savings:** 24 bytes vs 384+ bytes per node

### 4.3 Neighbor Access

```cpp
std::pair<const NodeId*, size_t> get_neighbors(NodeId id, LayerLevel layer) const {
    const auto& node = nodes_[id];
    uint32_t offset = get_layer_offset(node, layer);
    size_t count = node.link_counts[layer];
    return {links_pool_.data() + offset, count};
}
```

### 4.4 Thread-Safe Visited Tracking

```cpp
bool check_and_mark_visited(NodeId id, uint64_t query_id) const {
    uint64_t expected = visited_markers_[id]->load(std::memory_order_relaxed);
    if (expected == query_id) return true;  // Already visited
    return !visited_markers_[id]->compare_exchange_weak(
        expected, query_id, std::memory_order_relaxed);
}
```

- Each query gets unique ID (avoids O(N) clearing)
- Atomic compare-exchange prevents race conditions
- Memory order RELAXED sufficient for search traversal

---

## 5. Search Algorithms

### 5.1 SEARCH-LAYER (Algorithm 2)

Greedy search within a single HNSW layer:

```
SEARCH-LAYER(query, entry_points, ef, layer, graph, query_id)

C = MinHeap()    // Candidates to explore (closest first)
W = MaxHeap()    // Results (bounded to ef, furthest first)

Initialize C and W with entry_points

While C is not empty:
    c = C.pop()                    // Get closest candidate
    f = W.top()                    // Get furthest result

    If c.distance > f.distance:
        break                       // No improvement possible

    For each neighbor of c at layer:
        If not visited[neighbor]:
            d = asymmetric_distance(query, neighbor.code)

            If d < f.distance OR |W| < ef:
                C.push(neighbor, d)
                W.try_push(neighbor, d, ef)

Return W as sorted vector
```

**Complexity:** O(ef × M × K)

### 5.2 Asymmetric Search Distance Computation

```cpp
AsymmetricDist asymmetric_search_distance(const CPQuery& query, const CPCode& code) {
    float score = 0.0f;

    for (size_t r = 0; r < K; ++r) {
        size_t idx = code.decode_index(code.components[r]);
        bool is_negative = code.decode_sign_negative(code.components[r]);

        float val = query.rotated_vecs[r][idx];
        score += is_negative ? -val : val;
    }

    return -score;  // Negative for min-heap semantics
}
```

### 5.3 K-NN Search

```
K-NN-SEARCH(query, k, ef, graph)

Phase 1: Descend to Layer 0
    entry = graph.entry_point()
    For layer = top_layer down to 1:
        results = SEARCH-LAYER(query, {entry}, ef=1, layer)
        entry = results[0].id

Phase 2: Search Layer 0
    results = SEARCH-LAYER(query, {entry}, max(ef, k), layer=0)
    Return top-k results
```

**Complexity:** O(log N + ef × M × K)

### 5.4 Multiprobe Search

```cpp
std::vector<SearchResult> search_multiprobe(query, k, ef, num_probes, graph) {
    auto probes = multiprobe_generator.generate(query, num_probes);

    std::unordered_set<NodeId> seen;
    std::vector<SearchResult> all_results;

    for (const auto& probe : probes) {
        auto results = search(probe.code, k, ef, graph);
        for (const auto& r : results) {
            if (seen.insert(r.id).second) {
                all_results.push_back(r);
            }
        }
    }

    // Re-rank with primary query distance
    for (auto& r : all_results) {
        r.distance = asymmetric_distance(query, graph.get_code(r.id));
    }

    std::sort(all_results.begin(), all_results.end());
    return top_k(all_results, k);
}
```

---

## 6. Insert Algorithms

### 6.1 Standard INSERT (Algorithm 3)

```
INSERT(new_node_id, query, graph, params)

Phase 0: Generate random level
    level = floor(-log(uniform(0,1)) × m_L)

Phase 1: Zoom (descend to insertion level)
    For l = top_layer down to level+1:
        entry = SEARCH-LAYER(query, {entry}, ef=1, l)[0]

Phase 2: Construct (insert at all layers)
    For l = min(top_layer, level) down to 0:
        candidates = SEARCH-LAYER(query, entries, ef_construction, l)
        neighbors = SELECT-NEIGHBORS(query, candidates, M_l)

        Add forward edges: new_node → neighbors
        Add reverse edges: neighbors → new_node (with overflow handling)

        entries = candidates  // For next layer

Phase 3: Update entry point
    If level > top_layer:
        graph.set_entry_point(new_node, level)
```

### 6.2 SELECT-NEIGHBORS Heuristic (Algorithm 4)

```
SELECT-NEIGHBORS(base_query, candidates, M, keep_pruned)

Sort candidates by distance to base (ascending)

selected = []
discarded = []

For each candidate in sorted order:
    If |selected| >= M: break

    is_good = true
    For each already_selected:
        dist_to_selected = hamming_distance(candidate, selected)
        If dist_to_selected < candidate.distance:
            is_good = false
            break

    If is_good:
        selected.append(candidate)
    Else:
        discarded.append(candidate)

If keep_pruned AND |selected| < M:
    Fill remaining slots from discarded

Return selected
```

**Purpose:** Ensures spatial diversity by rejecting candidates too close to already-selected neighbors.

### 6.3 Parallel Construction Strategy

CP-HNSW uses a parallel batch insertion strategy for high-throughput construction:

**Phase 1: Bootstrap (first 1000 nodes, sequential)**
```cpp
for (size_t i = 0; i < bootstrap_count; ++i) {
    Insert::insert_hybrid(id, query, vec, all_vectors, dim, graph, params);
}
```
- Uses sequential hybrid insert for guaranteed connectivity
- Builds a well-connected core graph as entry point

**Phase 2: Wave-based Parallel Linking (remaining nodes)**
```cpp
for (size_t w = 0; w < waves; ++w) {
    #pragma omp parallel for
    for (size_t i = wave_start; i < wave_end; ++i) {
        Insert::insert_hybrid_parallel(id, query, vec, all_vectors, dim,
                                       graph, params, query_counter);
    }
}
```
- Uses atomic query_id per search to prevent stale visited markers
- Per-node spinlocks protect concurrent edge modifications
- Wave synchronization reduces blind spot issues

**Phase 3: Connectivity Repair**
```cpp
repair_connectivity(start_id, count, queries, vecs);
```
- BFS to find disconnected nodes
- Force-connects isolated nodes with true distance search
- Guarantees 100% connectivity

**Performance:** ~4x faster than sequential on multi-core systems

**Edge Selection with True Distance:**
```cpp
auto neighbors = SelectNeighbors::select_with_true_distance(
    new_vec, all_vectors, dim, candidates, M, keep_pruned);
```

This recomputes all candidate distances using exact float dot products, eliminating quantization error in edge selection.

### 6.4 Overflow Handling

When a neighbor's edge list is full:

```cpp
if (!graph.add_neighbor(neighbor_id, layer, new_node_id)) {
    // Collect all existing links + new node
    std::vector<SearchResult> all_candidates;
    for (auto link : neighbor_links) {
        float dist = dot_product(neighbor_vec, link_vec);
        all_candidates.push_back({link, -dist});
    }
    all_candidates.push_back({new_node_id, -dot_new});

    // Re-select best M neighbors
    auto new_neighbors = SelectNeighbors::select_simple(all_candidates, M);
    graph.set_neighbors(neighbor_id, layer, new_neighbors);
}
```

---

## 7. SIMD Optimizations

### 7.1 Fast Hadamard Transform

**AVX2 Implementation (8-way SIMD):**

```cpp
void fht_avx2(Float* vec, size_t len) {
    // Phase 1: Intra-register (h = 1, 2, 4)
    for (size_t i = 0; i < len; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);

        // h=1: Swap adjacent pairs
        __m256 v_swap = _mm256_permute_ps(v, 0xB1);
        v = _mm256_blend_ps(
            _mm256_add_ps(v, v_swap),
            _mm256_sub_ps(v, v_swap), 0xAA);

        // h=2, h=4 similar...
        _mm256_storeu_ps(&vec[i], v);
    }

    // Phase 2: Cross-register (h >= 8)
    for (size_t h = 8; h < len; h *= 2) {
        for (size_t i = 0; i < len; i += h * 2) {
            for (size_t j = i; j < i + h; j += 8) {
                __m256 x = _mm256_loadu_ps(&vec[j]);
                __m256 y = _mm256_loadu_ps(&vec[j + h]);
                _mm256_storeu_ps(&vec[j], _mm256_add_ps(x, y));
                _mm256_storeu_ps(&vec[j + h], _mm256_sub_ps(x, y));
            }
        }
    }
}
```

**Performance:** ~3-4x speedup vs scalar

### 7.2 Hamming Distance

**AVX2 for K=16, uint8_t:**

```cpp
HammingDist hamming_avx2_u8_16(const uint8_t* a, const uint8_t* b) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

    __m128i eq = _mm_cmpeq_epi8(va, vb);        // 0xFF if equal
    uint32_t mask = _mm_movemask_epi8(eq);      // 1 bit per byte

    return 16 - _mm_popcnt_u32(mask & 0xFFFF);  // Count mismatches
}
```

**Critical Note for uint16_t:**
`_mm256_movemask_epi8` returns 2 bits per 16-bit match. Must divide popcount by 2:

```cpp
uint32_t matches = _mm_popcnt_u32(mask) >> 1;  // Divide by 2!
```

### 7.3 Argmax with AVX-512

```cpp
void find_argmax_abs_avx512(const Float* buffer, size_t size,
                            size_t& out_idx, Float& out_abs) {
    const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __m512 max_vals = _mm512_setzero_ps();

    // Pass 1: Find global max
    for (size_t i = 0; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&buffer[i]);
        __m512 abs_v = _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), abs_mask));
        max_vals = _mm512_max_ps(max_vals, abs_v);
    }
    Float global_max = _mm512_reduce_max_ps(max_vals);

    // Pass 2: Find index
    const __m512 target = _mm512_set1_ps(global_max);
    for (size_t i = 0; i + 16 <= size; i += 16) {
        __m512 abs_v = /* compute absolute value */;
        __mmask16 eq_mask = _mm512_cmp_ps_mask(abs_v, target, _CMP_EQ_OQ);
        if (eq_mask) {
            out_idx = i + __builtin_ctz(eq_mask);
            out_abs = global_max;
            return;
        }
    }
}
```

### 7.4 Dispatcher Pattern

```cpp
inline void fht(Float* vec, size_t len) {
#if CPHNSW_HAS_AVX512
    fht_avx512(vec, len);
#elif CPHNSW_HAS_AVX2
    fht_avx2(vec, len);
#else
    fht_scalar(vec, len);
#endif
}
```

Compile-time selection based on available instruction sets.

---

## 8. CUDA GPU Acceleration

### 8.1 Architecture Overview

GPU acceleration targets batch operations where parallelism amortizes transfer overhead:

```
Host                          Device
─────                         ──────
vectors[] ──H2D transfer──→  d_vectors[]
                              ↓
                              rotation_chain_kernel (K times)
                              ↓
                              d_rotated[]
                              ↓
                              argmax_encode_kernel
                              ↓
codes[] ←──D2H transfer────  d_codes[]
```

### 8.2 CUDA Kernels

**Fast Hadamard Transform Kernel:**
```cpp
__global__ void fht_kernel(Float* data, size_t dim, size_t num_vecs) {
    extern __shared__ Float shared[];
    size_t vec_idx = blockIdx.x;

    // Load vector into shared memory
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        shared[i] = data[vec_idx * dim + i];
    }
    __syncthreads();

    // Butterfly iterations
    for (size_t len = 1; len < dim; len *= 2) {
        for (size_t i = threadIdx.x; i < dim / 2; i += blockDim.x) {
            size_t j = (i / len) * 2 * len + (i % len);
            Float a = shared[j], b = shared[j + len];
            shared[j] = a + b;
            shared[j + len] = a - b;
        }
        __syncthreads();
    }

    // Store result
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        data[vec_idx * dim + i] = shared[i];
    }
}
```

**Argmax Encoding with Warp Reduction:**
```cpp
template<typename ComponentT>
__global__ void argmax_encode_kernel(const Float* rotated, ComponentT* codes,
                                     size_t padded_dim, size_t num_vecs, size_t K) {
    size_t vec_idx = blockIdx.x / K;
    size_t rot_idx = blockIdx.x % K;

    // Thread-local maximum
    Float thread_max_abs = 0.0f;
    size_t thread_max_idx = 0;
    Float thread_max_val = 0.0f;

    for (size_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        Float val = rotated[vec_idx * K * padded_dim + rot_idx * padded_dim + i];
        Float abs_val = fabsf(val);
        if (abs_val > thread_max_abs) {
            thread_max_abs = abs_val;
            thread_max_idx = i;
            thread_max_val = val;
        }
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        Float other_abs = __shfl_down_sync(0xffffffff, thread_max_abs, offset);
        size_t other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        Float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
        if (other_abs > thread_max_abs) {
            thread_max_abs = other_abs;
            thread_max_idx = other_idx;
            thread_max_val = other_val;
        }
    }

    // Block-level reduction via shared memory
    // ... (first thread of each warp writes to shared, thread 0 reduces)

    // Thread 0 encodes and writes result
    if (threadIdx.x == 0) {
        bool is_negative = (final_val < 0);
        codes[vec_idx * K + rot_idx] = (final_idx << 1) | (is_negative ? 1 : 0);
    }
}
```

### 8.3 cuBLAS Integration

For brute-force similarity computation:

```cpp
cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_base, num_queries, dim,
            &alpha, d_base, dim,      // A = base vectors
            d_queries, dim,            // B = query vectors
            &beta, d_similarities, num_base);  // C = similarity matrix
```

This computes all pairwise dot products in a single highly-optimized operation.

### 8.4 Memory Management

```cpp
class GPUEncoder {
    Float* d_signs_;           // Persistent: K × 3 × padded_dim bytes
    Float* d_vectors_padded_;  // Per-batch: batch_size × padded_dim × 4B
    Float* d_rotated_;         // Per-batch: batch_size × K × padded_dim × 4B
    ComponentT* d_codes_;      // Per-batch: batch_size × K bytes
    cudaStream_t stream_;      // Asynchronous execution
};
```

**Allocation Strategy:**
- Signs are persistent (generated once)
- Work buffers reallocated only when batch size increases
- Stream-based asynchronous execution overlaps H2D, compute, D2H

### 8.5 Hybrid CPU+GPU Pipeline

Optimal architecture for production:

1. **CPU: Index Building** - Sequential HNSW insertion (graph connectivity critical)
2. **CPU: Candidate Generation** - HNSW search returns top-N candidates
3. **GPU: Re-ranking** - Exact similarity computation on candidates

```cpp
// Step 1: CPU generates candidates
auto candidates = index.search(query, rerank_k, ef);

// Step 2: GPU re-ranks with exact similarity
gpu_bf.rerank(queries, candidate_ids, num_queries, rerank_k, k,
              final_indices, final_distances);
```

---

## 9. Performance Characteristics

### 9.1 Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Encode vector | O(K × d log d) | O(d) buffer |
| Hamming distance | O(K) SIMD | O(1) |
| Asymmetric distance | O(K) | O(1) |
| Search layer | O(ef × M × K) | O(ef) |
| K-NN search | O(log N + ef × M × K) | O(ef) |
| Insert node | O(log N × ef × M × K) | O(1) |

### 9.2 Memory Usage

**Per Vector:**
- CPCode: K bytes (16-32)
- Graph metadata: ~24 bytes
- Graph edges: ~(M + M_max0) × 4 bytes ≈ 200 bytes
- Original vector (if stored): dim × 4 bytes

**Example (SIFT-1M: N=1M, dim=128, K=16, M=16):**
- Codes: 16 MB
- Graph: ~250 MB
- Original vectors: 512 MB
- **Total: ~780 MB** (vs 512 MB for vectors alone)

### 9.3 Expected Performance

**CPU (single-threaded, modern x86):**
| Metric | Value |
|--------|-------|
| Encoding throughput | 50-100K vectors/sec |
| Search latency (k=10, ef=50) | 0.5-2 ms |
| Build throughput | 5-10K insertions/sec |

**GPU (H100):**
| Metric | Speedup vs CPU |
|--------|----------------|
| Batch encoding | 10-50x |
| Similarity matrix | 100x+ |
| Full re-ranking | 20-50x |

### 9.4 Recall vs Speed Tradeoffs

| Parameter | Higher Value Effect |
|-----------|---------------------|
| `ef` | ↑ recall, ↓ speed |
| `K` | ↑ precision, ↑ memory, ↓ speed |
| `M` | ↑ recall, ↑ memory, ↓ build speed |
| `num_probes` | ↑ recall, ↓ speed (linear) |

**Typical configurations:**

| Use Case | K | M | ef | Expected Recall@10 |
|----------|---|---|----|--------------------|
| Fast search | 16 | 16 | 50 | 85-90% |
| Balanced | 16 | 32 | 100 | 92-95% |
| High recall | 32 | 32 | 200 | 97-99% |

---

## References

1. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., & Schmidt, L. (2015). *Practical and Optimal LSH for Angular Distance*. NeurIPS.

2. Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. IEEE TPAMI.
