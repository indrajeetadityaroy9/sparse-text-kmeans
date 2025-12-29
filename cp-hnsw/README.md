# CP-HNSW

Cross-Polytope Hierarchical Navigable Small World - a memory-efficient approximate nearest neighbor search algorithm optimized for angular/cosine similarity.

## Overview

CP-HNSW combines two state-of-the-art techniques:
- **Cross-Polytope LSH** (Andoni et al. 2015): Asymptotically optimal hashing for angular distance
- **HNSW** (Malkov & Yashunin 2018): Hierarchical graph for logarithmic-time navigation

Key benefits:
- **Memory efficient**: ~32x compression vs standard HNSW (K bytes per vector vs 4d bytes)
- **Fast distance computation**: O(1) SIMD Hamming distance vs O(d) float dot product
- **Optimal hashing**: Cross-Polytope LSH provides theoretical optimality guarantees

## Quick Start

```cpp
#include <cphnsw/index/cp_hnsw_index.hpp>

using namespace cphnsw;

// Create index for 128-dimensional vectors
CPHNSWIndex8 index(128, /*M=*/16, /*ef_construction=*/100);

// Add vectors
std::vector<float> vectors = ...;  // N x 128
for (size_t i = 0; i < N; ++i) {
    index.add(vectors.data() + i * 128);
}

// Search
std::vector<float> query(128);
auto results = index.search(query.data(), /*k=*/10, /*ef=*/50);

for (const auto& r : results) {
    std::cout << "ID: " << r.id << " Distance: " << r.distance << "\n";
}
```

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Run evaluation
./eval_cphnsw
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CPHNSW_USE_AVX512` | OFF | Enable AVX-512 SIMD |
| `CPHNSW_BUILD_TESTS` | ON | Build unit tests |
| `CPHNSW_BUILD_EVAL` | ON | Build evaluation framework |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Build benchmarks |

## API Reference

### `CPHNSWIndex<ComponentT, K>`

Template parameters:
- `ComponentT`: `uint8_t` for d ≤ 128, `uint16_t` for d > 128
- `K`: Code width (number of rotations, default 16)

#### Constructor

```cpp
CPHNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 100);
```

#### Methods

| Method | Description |
|--------|-------------|
| `add(const Float* vec)` | Add single vector |
| `add_batch(const Float* vecs, size_t count)` | Add multiple vectors |
| `search(const Float* query, size_t k, size_t ef)` | K-NN search |
| `search_multiprobe(...)` | Search with multiprobe for higher recall |
| `size()` | Number of indexed vectors |
| `verify_connectivity()` | Check graph connectivity |

## Architecture

```
include/cphnsw/
├── core/types.hpp           # Type definitions, CPCode, CPQuery
├── quantizer/
│   ├── hadamard.hpp         # Fast Hadamard Transform (FHT)
│   ├── rotation_chain.hpp   # Ψ(x) = H D₃ H D₂ H D₁ x
│   ├── cp_encoder.hpp       # Vector → CPCode encoding
│   └── multiprobe.hpp       # Multiprobe sequence generator
├── distance/hamming.hpp     # SIMD Hamming distance
├── graph/
│   ├── flat_graph.hpp       # Memory-efficient graph structure
│   └── priority_queue.hpp   # Min/Max heaps for search
├── algorithms/
│   ├── search_layer.hpp     # SEARCH-LAYER (Algorithm 2)
│   ├── select_neighbors.hpp # SELECT-NEIGHBORS-HEURISTIC (Algorithm 4)
│   ├── insert.hpp           # INSERT (Algorithm 3)
│   └── knn_search.hpp       # K-NN-SEARCH
└── index/cp_hnsw_index.hpp  # Main public API
```

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Add (single) | O(log N × ef_c × M × K) | O(M × L) |
| Search | O(log N + ef × M × K) | O(ef) |
| FHT | O(d log d) | O(1) |
| Hamming distance | O(1) SIMD | O(1) |

**Memory per vector**: K bytes (code) + ~16 bytes (graph metadata)

## References

- Andoni et al. (2015): "Practical and Optimal LSH for Angular Distance"
- Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
