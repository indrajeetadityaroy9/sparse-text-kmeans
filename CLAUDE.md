# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CP-HNSW (Cross-Polytope Hierarchical Navigable Small World) is a memory-efficient approximate nearest neighbor search algorithm optimized for angular/cosine similarity. It combines Cross-Polytope LSH (Andoni et al. 2015) with HNSW graphs (Malkov & Yashunin 2018) to achieve ~32x memory compression vs standard HNSW.

## Build Commands

```bash
# Full build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Build with specific options
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCPHNSW_USE_AVX512=ON \
    -DCPHNSW_USE_OPENMP=ON \
    -DCPHNSW_USE_CUDA=ON

# Build without CUDA (CPU only)
cmake .. -DCPHNSW_USE_CUDA=OFF
```

## Running Tests

```bash
# Build and run unit tests (requires GTest)
make cphnsw_tests && ctest

# Run specific test executable
./cphnsw_tests
```

## Evaluation Executables

```bash
./eval_cphnsw           # Basic evaluation
./eval_sift             # SIFT-1M benchmark
./eval_sift_gpu         # GPU-accelerated SIFT benchmark
./eval_master           # Full evaluation protocol (6 experiments)
./test_tiered           # Tiered construction test
./test_correlation      # Estimator correlation test
```

## Architecture

The codebase is a **header-only C++17 library** (except CUDA) with this structure:

```
include/cphnsw/
├── core/
│   ├── types.hpp          # CPCode, CPQuery, CPHNSWParams, SearchResult
│   └── debug.hpp          # Debug macros and validation utilities
├── quantizer/
│   ├── hadamard.hpp       # Fast Hadamard Transform (scalar, AVX2, AVX-512)
│   ├── rotation_chain.hpp # Ψ(x) = H D₃ H D₂ H D₁ x rotation encoding
│   ├── cp_encoder.hpp     # Vector → CPCode quantization pipeline
│   └── multiprobe.hpp     # Multiprobe sequence generation
├── distance/
│   └── hamming.hpp        # SIMD Hamming distance + asymmetric search distance
├── graph/
│   ├── flat_graph.hpp     # FlatHNSWGraph: contiguous memory layout graph
│   └── priority_queue.hpp # Min/Max heaps for search algorithms
├── algorithms/
│   ├── search_layer.hpp   # SEARCH-LAYER (Algorithm 2)
│   ├── select_neighbors.hpp # SELECT-NEIGHBORS-HEURISTIC (Algorithm 4)
│   ├── insert.hpp         # INSERT (Algorithm 3) + hybrid/parallel variants
│   └── knn_search.hpp     # K-NN-SEARCH with multiprobe support
└── index/
    └── cp_hnsw_index.hpp  # Main public API: CPHNSWIndex<ComponentT, K>
```

**CUDA code** (compiled, not header-only):
- `src/cuda/gpu_encoder.cu` - GPU kernels for batch encoding

**Key type aliases**:
- `CPHNSWIndex8` = `CPHNSWIndex<uint8_t, 16>` for dim ≤ 128
- `CPHNSWIndex16` = `CPHNSWIndex<uint16_t, 16>` for dim > 128
- `CPHNSWIndex32` = `CPHNSWIndex<uint8_t, 32>` for higher precision

## Key Implementation Details

**Parallel Construction**: `add_batch_parallel()` uses shuffled insertion order to break lock contention on clustered data. This prevents adjacent vectors from fighting over the same hub node locks.

**Hybrid Insert**: CP search for candidate generation + float edge selection for high-quality graph construction. This achieves 100% graph connectivity.

**Connectivity Repair**: Optional pass (enable with `-DCPHNSW_ENABLE_CONNECTIVITY_REPAIR`) that connects isolated nodes after parallel construction.

**Thread Safety**: Uses per-node spinlocks and atomic query counters. Each search gets a fresh query_id to prevent stale visited marker bugs.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CPHNSW_USE_AVX512` | ON | AVX-512 SIMD (-mavx512f -mavx512bw -mavx512vl) |
| `CPHNSW_USE_OPENMP` | ON | OpenMP parallelization |
| `CPHNSW_USE_CUDA` | ON | CUDA GPU acceleration (sm_70, sm_80, sm_90) |
| `CPHNSW_BUILD_TESTS` | ON | Build unit tests |
| `CPHNSW_BUILD_EVAL` | ON | Build evaluation framework |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Build Google Benchmark targets |

## Data Files

Evaluation datasets go in `evaluation/data/` (gitignored). Use `evaluation/scripts/download_datasets.sh` to fetch SIFT-1M and GIST-1M.
