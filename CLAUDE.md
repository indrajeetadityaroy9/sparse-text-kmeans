# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build (from repo root)
cd cp-hnsw && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Build with AVX-512
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPHNSW_USE_AVX512=ON

# Run evaluation
./eval_cphnsw           # Random sphere datasets
./eval_comprehensive    # Multi-dataset evaluation
./eval_sift             # SIFT-1M evaluation (needs OpenMP for fast ground truth)

# Run tests (requires GTest)
ctest
# or
./cphnsw_tests

# Run single test
./cphnsw_tests --gtest_filter=HadamardTest.ScalarImplementation
```

### CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `CPHNSW_USE_AVX512` | OFF | Enable AVX-512 SIMD |
| `CPHNSW_BUILD_TESTS` | ON | Build GTest unit tests |
| `CPHNSW_BUILD_EVAL` | ON | Build evaluation executables |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Build Google Benchmark microbenchmarks |

## Architecture

CP-HNSW is a memory-efficient approximate nearest neighbor search library combining Cross-Polytope LSH with HNSW graphs. The implementation is a C++17 header-only library.

### Core Components

```
cp-hnsw/include/cphnsw/
├── core/types.hpp           # CPCode, CPQuery, CPHNSWParams
├── quantizer/
│   ├── hadamard.hpp         # Fast Hadamard Transform (scalar/AVX2/AVX512)
│   ├── rotation_chain.hpp   # Pseudo-random rotation Ψ(x) = H D₃ H D₂ H D₁ x
│   ├── cp_encoder.hpp       # Vector → CPCode encoding pipeline
│   └── multiprobe.hpp       # Multiprobe sequence generator
├── distance/hamming.hpp     # SIMD Hamming distance
├── graph/
│   ├── flat_graph.hpp       # Contiguous memory HNSW graph
│   └── priority_queue.hpp   # Min/Max heaps for search traversal
├── algorithms/
│   ├── search_layer.hpp     # SEARCH-LAYER algorithm
│   ├── select_neighbors.hpp # SELECT-NEIGHBORS-HEURISTIC
│   ├── insert.hpp           # INSERT with tiered construction
│   └── knn_search.hpp       # K-NN search + multiprobe variant
└── index/cp_hnsw_index.hpp  # Main public API (CPHNSWIndex)
```

### Key Design Decisions

1. **Tiered Construction**: First 10K nodes use full float search (backbone phase) for guaranteed connectivity; remaining nodes use CP search with float edge selection (hybrid phase).

2. **Asymmetric Distance**: CPQuery stores full rotated vectors + magnitudes; CPCode stores only K bytes. This enables proper gradient-based navigation despite discrete codes.

3. **Flat Memory Layout**: FlatHNSWGraph uses contiguous arrays instead of vector<vector> for cache locality.

4. **Template Parameters**: `CPHNSWIndex<ComponentT, K>` where ComponentT is `uint8_t` (d ≤ 128) or `uint16_t` (d > 128), K is code width (default 16).

### Type Aliases

```cpp
using CPHNSWIndex8 = CPHNSWIndex<uint8_t, 16>;    // d ≤ 128
using CPHNSWIndex16 = CPHNSWIndex<uint16_t, 16>;  // d > 128
using CPHNSWIndex32 = CPHNSWIndex<uint8_t, 32>;   // Higher precision
```

## Reference Papers

- `arXiv-1509.02897v1/`: "Practical and Optimal LSH for Angular Distance" (Cross-Polytope LSH)
- `arXiv-1603.09320/`: "Efficient and robust approximate nearest neighbor search using HNSW"
