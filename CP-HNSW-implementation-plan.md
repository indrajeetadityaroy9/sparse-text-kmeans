# CP-HNSW Implementation Plan

## Overview
C++ implementation of CP-HNSW (Cross-Polytope Hierarchical Navigable Small World) with SIMD optimizations, multiprobe support, and evaluation framework.

---

## Project Structure

```
cp-hnsw/
├── CMakeLists.txt
├── include/cphnsw/
│   ├── core/
│   │   ├── types.hpp              # NodeId, CPCode<K>, CPHNSWParams, SearchResult
│   │   ├── simd_utils.hpp         # SIMD detection
│   │   └── bit_utils.hpp          # Popcount wrappers
│   ├── quantizer/
│   │   ├── hadamard.hpp           # FHT interface + scalar/AVX2/AVX512 impls
│   │   ├── rotation_chain.hpp     # Ψ(x) = H D₃ H D₂ H D₁ x
│   │   ├── cp_encoder.hpp         # Vector → CPCode<K>
│   │   └── multiprobe.hpp         # Probe sequence generator
│   ├── distance/
│   │   └── hamming.hpp            # Hamming distance (scalar + SIMD)
│   ├── graph/
│   │   ├── flat_graph.hpp         # FlatHNSWGraph with contiguous memory layout
│   │   └── priority_queue.hpp     # MinHeap/MaxHeap for search
│   ├── algorithms/
│   │   ├── search_layer.hpp       # SEARCH-LAYER (Algorithm 2)
│   │   ├── select_neighbors.hpp   # SELECT-NEIGHBORS-HEURISTIC (Algorithm 4)
│   │   ├── insert.hpp             # INSERT (Algorithm 3)
│   │   └── knn_search.hpp         # K-NN-SEARCH + multiprobe variant
│   └── index/
│       └── cp_hnsw_index.hpp      # Main public API
├── src/quantizer/
│   ├── hadamard_scalar.cpp
│   ├── hadamard_avx2.cpp
│   └── hadamard_avx512.cpp
├── tests/unit/                    # GTest unit tests
├── benchmarks/                    # Google Benchmark microbenchmarks
└── evaluation/
    ├── datasets/                  # SIFT-1M, GloVe-100, GIST-1M loaders
    ├── metrics/                   # Recall, QPS measurement
    └── eval_main.cpp              # Main evaluation driver
```

---

## Implementation Steps

### Step 1: Core Types and Infrastructure
**Files to create:** `include/cphnsw/core/types.hpp`

- `NodeId` (uint32_t), `LayerLevel` (uint8_t), `HammingDist` (uint16_t)
- `CPCode<ComponentT, K>`: template for dimension-appropriate component type
  - `uint8_t` for d ≤ 128 (SIFT, GloVe)
  - `uint16_t` for d > 128 (GIST-1M, d=960)
- `CPQuery<ComponentT, K>`: query-only struct with magnitudes for multiprobe (NOT stored in index)
- `CPHNSWParams`: dim, padded_dim, k, M, M_max0, ef_construction, m_L, seed
- `SearchResult`: {NodeId, HammingDist} with comparison operators

### Step 2: Fast Hadamard Transform
**Files to create:** `include/cphnsw/quantizer/hadamard.hpp`, `src/quantizer/hadamard_*.cpp`

Scalar implementation (reference):
```cpp
for (size_t h = 1; h < len; h *= 2) {
    for (size_t i = 0; i < len; i += h * 2) {
        for (size_t j = i; j < i + h; ++j) {
            Float x = vec[j], y = vec[j + h];
            vec[j] = x + y;
            vec[j + h] = x - y;
        }
    }
}
```

AVX2 implementation:
- Phase 1: Intra-register butterfly (h=1,2,4) using `_mm256_permute_ps`, `_mm256_blend_ps`
- Phase 2: Cross-register butterfly (h≥8) using load/store with add/sub

**Complexity:** O(d log d) time, O(1) auxiliary space

**Note on Normalization:** FHT increases magnitude by √d, but since we use `argmax` (scale-invariant), no normalization needed. Just ensure float doesn't overflow for very large d (sanity check: d < 2^20 is safe).

### Step 3: Pseudo-Random Rotation Chain
**Files to create:** `include/cphnsw/quantizer/rotation_chain.hpp`

- Pre-generate 3*k sign vectors (D₁, D₂, D₃ for each of k rotations)
- `apply(vec, rotation_idx)`: D₁→FHT→D₂→FHT→D₃→FHT

### Step 4: CP Encoder
**Files to create:** `include/cphnsw/quantizer/cp_encoder.hpp`

- Zero-pad input to power of 2
- For each rotation r ∈ [0,k): apply rotation, find argmax|y_i|, encode (index, sign)
- Store magnitudes for multiprobe tie-breaking
- `encode_with_probe_data()`: return sorted indices by magnitude for each rotation

### Step 5: Multiprobe Generator
**Files to create:** `include/cphnsw/quantizer/multiprobe.hpp`

Based on Andoni et al. Section 5:
- Primary probe = original code
- Generate alternatives by flipping to 2nd, 3rd best indices
- Rank by log probability: log(Pr) ∝ -(|x_max| - |x_alt|)²
- Use priority queue for incremental generation

### Step 6: Hamming Distance
**Files to create:** `include/cphnsw/distance/hamming.hpp`

**CRITICAL:** Distance = K - matches (not XOR-based, use equality comparison)

Scalar: count mismatches in O(K)

SIMD for uint8_t components (K≤32, AVX2):
```cpp
inline uint32_t dist_avx2_byte(const uint8_t* a, const uint8_t* b, size_t k) {
    __m256i va = _mm256_loadu_si256((const __m256i*)a);
    __m256i vb = _mm256_loadu_si256((const __m256i*)b);

    // Compare bytes for equality (0xFF if equal, 0x00 if not)
    __m256i eq = _mm256_cmpeq_epi8(va, vb);

    // Create bitmask (1 bit per byte)
    uint32_t mask = _mm256_movemask_epi8(eq);

    // Distance = K - matches
    return k - _mm_popcnt_u32(mask);
}
```

SIMD for uint16_t components (GIST-1M):
```cpp
inline uint32_t dist_avx2_short(const uint16_t* a, const uint16_t* b, size_t k) {
    __m256i va = _mm256_loadu_si256((const __m256i*)a);
    __m256i vb = _mm256_loadu_si256((const __m256i*)b);

    // cmpeq_epi16: match = 0xFFFF (two bytes of 1s)
    __m256i eq = _mm256_cmpeq_epi16(va, vb);

    // movemask_epi8 returns 2 bits per 16-bit match!
    uint32_t mask = _mm256_movemask_epi8(eq);

    // CRITICAL: Divide popcount by 2 to get actual match count
    return k - (_mm_popcnt_u32(mask) >> 1);
}
```

### Step 7: HNSW Graph Structure (Flat Layout)
**Files to create:** `include/cphnsw/graph/flat_graph.hpp`, `priority_queue.hpp`

**CRITICAL:** Use flat memory layout, NOT vector<vector>

- `FlatHNSWGraph`: single `links_pool` array + `CompactNode` metadata (see Key Data Structures)
- Codes stored in separate parallel array for cache-friendly access
- `visited_markers`: parallel array of atomic<uint64_t> compared to query_id
- `MinHeap`/`MaxHeap`: std::priority_queue wrappers for search

**Flat Indexing Strategy (Option B - Dense Packing):**
Store all layers contiguously with cumulative counts:
```
links_pool[offset]             = Layer 0 links (up to M_max0)
links_pool[offset + count_L0]  = Layer 1 links (up to M)
links_pool[offset + count_L0 + count_L1] = Layer 2 links...
```

```cpp
struct CompactNode {
    uint32_t link_offset;   // Start in links_pool
    uint8_t max_layer;
    uint8_t link_counts[8]; // Actual count per layer (supports max_layer ≤ 7)
    // Total: 13 bytes, pad to 16 for alignment
};

// Helper to find layer L start offset
uint32_t get_layer_offset(const CompactNode& n, uint8_t layer) {
    uint32_t off = n.link_offset;
    for (uint8_t l = 0; l < layer; ++l) {
        off += n.link_counts[l];
    }
    return off;
}
```

**Note:** High-layer nodes (layer > 2) are exponentially rare (~1% of nodes), so the small overhead of walking counts is acceptable.

### Step 8: SEARCH-LAYER Algorithm
**Files to create:** `include/cphnsw/algorithms/search_layer.hpp`

From `arXiv-1603.09320/technical_details_1.tex` Algorithm 1:
- W (max-heap): found neighbors, bounded to ef
- C (min-heap): candidates to explore
- Terminate when closest candidate is worse than furthest found
- Use atomic visited_marker with query_id for thread-safe tracking

**Complexity:** O(ef × M × K)

### Step 9: SELECT-NEIGHBORS-HEURISTIC
**Files to create:** `include/cphnsw/algorithms/select_neighbors.hpp`

From Algorithm 4:
- Sort candidates by distance to base
- Accept candidate if closer to base than to any already-selected neighbor
- Optionally keep pruned connections for robustness

### Step 10: INSERT Algorithm
**Files to create:** `include/cphnsw/algorithms/insert.hpp`

From Algorithm 3:
1. Generate level: l = floor(-ln(uniform) × m_L)
2. Phase 1 (zoom): descend from top layer to l+1 with ef=1
3. Phase 2 (construct): for each layer l down to 0:
   - Search for neighbors with ef_construction
   - Select neighbors using heuristic
   - Add bidirectional edges
   - Shrink overflowing neighbor lists
4. Update entry point if new level > top_layer

**Complexity:** O(log N × ef_c × M × K)

### Step 11: K-NN-SEARCH
**Files to create:** `include/cphnsw/algorithms/knn_search.hpp`

1. Descend layers L→1 with ef=1
2. Search layer 0 with full ef
3. Return top-k results

Multiprobe variant:
- Generate probe sequence
- Union results from all probes
- Deduplicate and return top-k

### Step 12: Main Index API
**Files to create:** `include/cphnsw/index/cp_hnsw_index.hpp`

```cpp
template<size_t K = 16>
class CPHNSWIndex {
    void add(const Float* vec, NodeId id);
    void add_batch(const Float* vecs, size_t count);
    vector<SearchResult> search(const Float* query, size_t k, size_t ef);
    vector<SearchResult> search_multiprobe(const Float* query, size_t k, size_t ef, size_t num_probes);
};
```

### Step 13: Evaluation Framework
**Files to create:** `evaluation/datasets/*.cpp`, `evaluation/metrics/*.hpp`, `evaluation/eval_main.cpp`

Dataset loaders:
- SIFT-1M: 128-dim, fvecs/ivecs format → use `CPCode<uint8_t, K>`
- GloVe-100: 100-dim, text format → use `CPCode<uint8_t, K>`
- GIST-1M: 960-dim, fvecs format → use `CPCode<uint16_t, K>`

Metrics:
- Recall@k = |retrieved ∩ ground_truth| / k
- QPS = 1e6 / mean_latency_us
- Memory footprint

**CRITICAL: Connectivity Analysis**
Since quantized metric is coarse, graph may become disconnected:
```cpp
// After build, verify layer 0 is connected
size_t visited = BFS_from_entry_point(graph, layer=0);
assert(visited == graph.size());  // Fail if disconnected
```
If connectivity fails frequently → increase K (more rotations) or use float precision for edge selection heuristic

### Step 14: Build System
**Files to create:** `CMakeLists.txt`

- C++17, -O3, -march=native
- AVX2 by default, optional AVX-512
- Link GTest for tests, Google Benchmark for benchmarks

---

## Key Data Structures

### CPCode (Index-Stored, Compact)
```cpp
// Component type depends on dimensionality:
// - uint8_t for d ≤ 128 (7-bit index + 1-bit sign = 8 bits)
// - uint16_t for d > 128 (e.g., GIST-1M d=960 needs 10+1 = 11 bits)
template <typename ComponentT, size_t K>
struct CPCode {
    std::array<ComponentT, K> components;  // ONLY discretized data, NO floats
};

// Type aliases
using CPCode8 = CPCode<uint8_t, 16>;    // For SIFT-1M (d=128)
using CPCode16 = CPCode<uint16_t, 16>;  // For GIST-1M (d=960)
```

### CPQuery (Query-Time Only, NOT stored in index)
```cpp
// Separate struct for query - holds data needed for multiprobe
template <typename ComponentT, size_t K>
struct CPQuery {
    CPCode<ComponentT, K> primary_code;
    std::array<float, K> magnitudes;      // For probability ranking
    std::array<uint32_t, K> original_indices;  // For probe generation
};
```

### Flat Graph Layout (Memory-Efficient)
```cpp
// CRITICAL: Do NOT use vector<vector> - it destroys cache locality
// and adds 24 bytes overhead per vector header

class FlatHNSWGraph {
    // All links in one contiguous array (dense packing)
    // Layout: [Node0_L0_links | Node0_L1_links | Node1_L0_links | ...]
    std::vector<NodeId> links_pool;

    // Compact node metadata (16 bytes per node, aligned)
    struct alignas(16) CompactNode {
        uint32_t link_offset;   // Index into links_pool
        uint8_t max_layer;
        uint8_t link_counts[8]; // Actual count per layer (supports max_layer ≤ 7)
        uint8_t _padding[3];    // Pad to 16 bytes
    };
    std::vector<CompactNode> nodes;

    // Codes stored separately for cache-friendly SIMD scanning
    std::vector<CPCode8> codes;  // Parallel array, codes[i] = node i's code

    // Visited tracking (one uint64 per node, compared to global query_id)
    std::vector<std::atomic<uint64_t>> visited_markers;

    // Helper to find layer L start offset in links_pool
    uint32_t get_layer_offset(NodeId id, uint8_t layer) const {
        const auto& n = nodes[id];
        uint32_t off = n.link_offset;
        for (uint8_t l = 0; l < layer; ++l) off += n.link_counts[l];
        return off;
    }
};
```

**Memory comparison per node (K=16, M=16, L=2):**
| Design | Bytes |
|--------|-------|
| vector<vector> | 96 (headers) + 16 (code) + floats = 176+ |
| Flat layout | 16 (CompactNode) + 16 (code) = 32 |

---

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| FHT | O(d log d) | O(1) |
| CP encoding | O(k × d log d) | O(k) |
| Hamming distance | O(1) SIMD | O(1) |
| SEARCH-LAYER | O(ef × M × K) | O(ef) |
| INSERT | O(log N × ef_c × M × K) | O(M × L) |
| K-NN-SEARCH | O(log N + ef × M × K) | O(ef) |

**Memory:** N × K bytes (codes) + N × M × L × 4 bytes (graph)
vs Standard HNSW: N × d × 4 bytes + graph

---

## Reference Files

- `CP-HNSW.md`: Mathematical foundation (Ψ(x), CP codes, distance metric)
- `arXiv-1603.09320/technical_details_1.tex`: HNSW algorithms (direct pseudocode translation)
- `arXiv-1509.02897v1/multiprobe.tex`: Multiprobe scheme
- `arXiv-1509.02897v1/cp_analysis.tex`: Collision probability analysis
