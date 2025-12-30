# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build (from repo root)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Build without AVX-512 (for older CPUs)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPHNSW_USE_AVX512=OFF

# Build without CUDA (for CPU-only systems)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPHNSW_USE_CUDA=OFF

# Run evaluation (from build directory)
./eval_cphnsw           # Random sphere datasets
./eval_comprehensive    # Multi-dataset evaluation
./eval_sift             # SIFT-1M evaluation
./eval_sift_gpu         # SIFT-1M with GPU acceleration (requires CUDA)

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
| `CPHNSW_USE_AVX512` | ON | Enable AVX-512 SIMD |
| `CPHNSW_USE_OPENMP` | ON | Enable OpenMP parallelization |
| `CPHNSW_USE_CUDA` | ON | Enable CUDA GPU acceleration |
| `CPHNSW_BUILD_TESTS` | ON | Build GTest unit tests |
| `CPHNSW_BUILD_EVAL` | ON | Build evaluation executables |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Build Google Benchmark microbenchmarks |

## Architecture

CP-HNSW is a memory-efficient approximate nearest neighbor search library combining Cross-Polytope LSH with HNSW graphs. The implementation is a C++17 header-only library.

### Core Components

```
include/cphnsw/
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
│   ├── insert.hpp           # INSERT with hybrid construction + parallel linking
│   └── knn_search.hpp       # K-NN search + multiprobe variant
└── index/cp_hnsw_index.hpp  # Main public API (CPHNSWIndex)
```

### Key Design Decisions

1. **Hybrid Construction**: CP search for fast candidate generation + TRUE cosine distance for accurate edge selection. Achieves 100% graph connectivity without backbone phase.

2. **Asymmetric Search Distance**: `asymmetric_search_distance()` returns negative dot product for min-heap compatibility. Enables proper gradient-based navigation despite discrete codes.

3. **Fine-Grained Locking**: Per-node spinlocks with `_mm_pause()` hint for hyperthreading-friendly parallel builds. Four-phase batch insertion prevents vector reallocation races.

4. **Flat Memory Layout**: FlatHNSWGraph uses contiguous arrays instead of vector<vector> for cache locality.

5. **Template Parameters**: `CPHNSWIndex<ComponentT, K>` where ComponentT is `uint8_t` (d ≤ 128) or `uint16_t` (d > 128), K is code width (default 32).

### Default Parameters (optimized for high recall)

```cpp
k = 32               // Code width (rotations)
M = 32               // Max connections per node
M_max0 = 64          // Max connections at layer 0
ef_construction = 200 // Search width during construction
```

### Type Aliases

```cpp
using CPHNSWIndex8 = CPHNSWIndex<uint8_t, 16>;    // d ≤ 128
using CPHNSWIndex16 = CPHNSWIndex<uint16_t, 16>;  // d > 128
using CPHNSWIndex32 = CPHNSWIndex<uint8_t, 32>;   // Higher precision
```

## Reference Papers

- `resources/cross-polytope-lsh/`: "Practical and Optimal LSH for Angular Distance" (Andoni et al. 2015)
- `resources/hnsw-paper/`: "Efficient and robust approximate nearest neighbor search using HNSW" (Malkov & Yashunin 2018)
