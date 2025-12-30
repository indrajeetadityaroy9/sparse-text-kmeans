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

// Create index for 128-dimensional vectors (optimized defaults)
CPHNSWIndex8 index(128, /*M=*/32, /*ef_construction=*/200);

// Add vectors (guaranteed 100% connectivity)
std::vector<float> vectors = ...;  // N x 128
index.add_batch(vectors.data(), N);

// Or add with parallel construction (~4x faster, 100% connectivity)
index.add_batch_parallel(vectors.data(), N);

// Search with re-ranking for high recall
std::vector<float> query(128);
auto results = index.search_and_rerank(query.data(), /*k=*/10, /*ef=*/100, /*rerank_k=*/500);

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
| `CPHNSW_USE_AVX512` | ON | Enable AVX-512 SIMD |
| `CPHNSW_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `CPHNSW_USE_CUDA` | ON | Enable CUDA GPU acceleration |
| `CPHNSW_BUILD_TESTS` | ON | Build unit tests |
| `CPHNSW_BUILD_EVAL` | ON | Build evaluation framework |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Build benchmarks |

## API Reference

### `CPHNSWIndex<ComponentT, K>`

Template parameters:
- `ComponentT`: `uint8_t` for d ≤ 128, `uint16_t` for d > 128
- `K`: Code width (number of rotations, default 32)

#### Constructor

```cpp
CPHNSWIndex(size_t dim, size_t M = 32, size_t ef_construction = 200);
```

#### Methods

| Method | Description |
|--------|-------------|
| `add(const Float* vec)` | Add single vector (sequential) |
| `add_batch(const Float* vecs, size_t count)` | Add multiple vectors (sequential, 100% connectivity) |
| `add_batch_parallel(const Float* vecs, size_t count)` | Add multiple vectors (~4x faster, 100% connectivity) |
| `search(const Float* query, size_t k, size_t ef)` | K-NN search with CP distance |
| `search_and_rerank(const Float* query, size_t k, size_t ef, size_t rerank_k)` | Search + exact re-ranking (recommended) |
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
