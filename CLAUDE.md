# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CP-HNSW (Cross-Polytope Hierarchical Navigable Small World) is a memory-efficient approximate nearest neighbor search algorithm optimized for angular/cosine similarity. It combines Cross-Polytope LSH (asymptotically optimal hashing) with HNSW graph navigation.

**Key innovation**: Replace 4d-byte float vectors with K-bit hash codes (2.8x memory compression) while maintaining competitive recall through hybrid search with reranking.

## Build Commands

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Disable CUDA if not available
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPHNSW_USE_CUDA=OFF

# Disable AVX-512 (e.g., on Apple Silicon or older CPUs)
cmake .. -DCPHNSW_USE_AVX512=OFF

# Enable evaluation framework (requires porting to unified API)
cmake .. -DCPHNSW_BUILD_EVAL=ON
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CPHNSW_USE_AVX512` | ON | AVX-512 SIMD (fallback to AVX2) |
| `CPHNSW_USE_OPENMP` | ON | OpenMP parallelization |
| `CPHNSW_USE_CUDA` | ON | CUDA GPU acceleration |
| `CPHNSW_BUILD_TESTS` | ON | Unit tests |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Google Benchmark microbenchmarks |
| `CPHNSW_BUILD_EVAL` | OFF | Evaluation framework (needs porting) |

### Running Tests

```bash
# GTest-based unit tests (if GTest installed)
./cphnsw_tests

# Standalone unified API test
./test_unified_distance    # Policy-based design validation (6 tests)

# Run single GTest case
./cphnsw_tests --gtest_filter=HadamardTest.SelfInverse
```

## Architecture

The codebase uses a **Unified Policy-Based Design** where a single `CPHNSWIndex<K, R, Shift>` template handles both Phase 1 (pure RaBitQ) and Phase 2 (with residual quantization). When `R=0`, all residual operations compile away for zero overhead.

```
include/cphnsw/
├── api/                        # PUBLIC API (Entry Point)
│   ├── config.hpp              #   IndexParams, BuildParams, SearchParams
│   └── index.hpp               #   CPHNSWIndex<K, R, Shift> main interface
│
├── core/                       # Core Types & Utilities
│   ├── codes.hpp               #   ResidualCode<K, R> unified code type
│   │                           #     - Zero-overhead: R=0 collapses to Phase 1
│   │                           #     - CodeSoALayout<CodeT, N> for SIMD batching
│   ├── debug.hpp               #   Debug macros and logging
│   ├── memory.hpp              #   Cache alignment, prefetch utilities
│   └── types.hpp               #   Legacy types (CPCode, BinaryCode, CPHNSWParams)
│
├── distance/                   # Unified Distance Computation
│   ├── metric_policy.hpp       #   UnifiedMetricPolicy<K, R, Shift>
│   │                           #     - Phase 1: R=0 → pure Hamming
│   │                           #     - Phase 2: R>0 → (primary << Shift) + residual
│   └── detail/                 #   SIMD Kernel Implementations
│       ├── scalar_kernels.hpp  #     Portable C++ fallback
│       ├── avx2_kernels.hpp    #     AVX2 optimized (256-bit)
│       └── avx512_kernels.hpp  #     AVX-512 optimized (512-bit)
│
├── encoder/                    # Vector Encoding
│   ├── cp_encoder.hpp          #   CPEncoder<K, R> Cross-Polytope encoder
│   ├── rotation.hpp            #   RotationChain Ψ(x) = H·D₃·H·D₂·H·D₁·x
│   └── transform/
│       └── fht.hpp             #   Fast Hadamard Transform O(d log d)
│
├── graph/                      # Graph Storage
│   ├── flat_graph.hpp          #   FlatGraph<CodeT> NSW graph
│   ├── neighbor_block.hpp      #   UnifiedNeighborBlock<CodeT> SoA storage
│   ├── priority_queue.hpp      #   Min/Max heaps for search
│   └── visitation_table.hpp    #   Epoch-based visited tracking
│
├── search/                     # Search Algorithms
│   └── search_engine.hpp       #   SearchEngine<Policy> greedy beam search
│
└── cuda/                       # CUDA Headers
    ├── gpu_encoder.cuh         #   GPU encoding interface
    └── gpu_knn_graph.cuh       #   GPU k-NN graph interface

src/cuda/                       # CUDA Source (compiled library)
├── gpu_encoder.cu
└── gpu_knn_graph.cu

tests/unit/
├── test_unified_distance.cpp   # Unified API validation (6 tests)
└── test_hadamard.cpp           # FHT correctness (GTest)

evaluation/                     # Benchmarking (needs porting to unified API)
├── eval_sift.cpp               # SIFT-1M benchmark
├── eval_master.cpp             # PhD 6-experiment protocol
└── ...
```

## Key Technical Details

### Template Parameters (Unified API)

`CPHNSWIndex<K, R, Shift>`:
- `K`: Primary code bits (32, 64, 128, 256)
- `R`: Residual code bits (0 = Phase 1, 16/32 = Phase 2)
- `Shift`: Bit-shift for residual weighting (default 2 = 4:1 ratio)

```cpp
// Phase 1: Pure RaBitQ (64-bit codes, no residual)
using Index64 = cphnsw::CPHNSWIndex<64, 0>;

// Phase 2: With residual (64-bit primary + 32-bit residual)
using Index64_32 = cphnsw::CPHNSWIndex<64, 32, 2>;
```

### Usage Example

```cpp
#include "cphnsw/api/index.hpp"

// Create index
cphnsw::IndexParams params;
params.dim = 128;
params.M = 32;
params.ef_construction = 200;

cphnsw::CPHNSWIndex<64, 0> index(params);  // Phase 1

// Add vectors
index.add_batch(vectors, num_vectors);

// Search
cphnsw::SearchParams search_params;
search_params.k = 10;
search_params.ef = 100;
search_params.rerank = true;

auto results = index.search(query, search_params);
```

### Performance Characteristics

- **Memory**: K/8 bytes/vector + ~16 bytes graph metadata
- **Graph-only recall ceiling**: K=32→23%, K=64→37% (limited by quantization)
- **Hybrid recall** (with reranking): K=64 achieves 82.6% recall@10
- **Parallel scaling**: Near-linear to 16 threads (84% efficiency)

### Thread Safety

- `add()` and `add_batch()`: Thread-safe (per-node spinlocks)
- `search()` and `search_batch()`: Thread-safe (read-only)

## Python Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dependencies
pip install -e .

# Install with optional FAISS CPU support
pip install -e ".[faiss]"

# Run evaluation scripts
python evaluation/scripts/plot_results.py --input results/ --output results/plots/
python evaluation/baselines/run_faiss.py --sift ~/datasets/sift1m/sift --output results/
```

## Datasets

SIFT-1M datasets are tracked via Git LFS. Download evaluation datasets:
```bash
bash evaluation/scripts/download_datasets.sh
```

---

## PhD Optimization Phases

### Phase 1: RaBitQ Bitwise Distance ✅ COMPLETE
XOR + PopCount distance for ~3x speedup over SIMD gather.
- Key type: `ResidualCode<K, 0>` (R=0 collapses residual storage)
- Distance: Pure Hamming via `UnifiedMetricPolicy<K, 0>`

### Phase 2: Residual Quantization ✅ COMPLETE
Adds residual codes to recover quantization precision.
- Key type: `ResidualCode<K, R>` where R > 0
- Distance formula: `(primary_hamming << Shift) + residual_hamming`
- Default Shift=2 gives 4:1 primary:residual weighting

**Note**: Phase 2 underperforms on random synthetic data due to concentration of measure. This is expected behavior for LSH methods; real datasets (SIFT-1M) with diverse pairwise distances show R² = 0.84 correlation.

### Phase 3: Learned Rotations ⏳ NOT STARTED
Replace random diagonal matrices with learned ones via PyTorch + Gumbel-Softmax.

### Remaining Work

1. **Port evaluation harness**: Update `evaluation/*.cpp` to use unified `CPHNSWIndex<K, R>` API
2. **SIFT-1M baseline**: Run unified index on SIFT-1M to establish Phase 2 baseline
3. **Phase 3 implementation**: After confirming Phase 2 improvement on real data

---

## Migration Notes

The codebase was refactored from a legacy dual-API design to a unified Policy-Based Design. Key changes:

| Legacy API | Unified API |
|------------|-------------|
| `CPHNSWIndex<ComponentT, K>` | `CPHNSWIndex<K, R, Shift>` |
| `FlatHNSWGraph<ComponentT, K>` | `FlatGraph<ResidualCode<K, R>>` |
| `NeighborBlock<ComponentT, K>` | `UnifiedNeighborBlock<CodeT>` |
| `hamming.hpp` (62KB monolith) | `metric_policy.hpp` + `detail/*.hpp` |

The evaluation framework (`evaluation/*.cpp`) still references the legacy API and needs to be updated before enabling `CPHNSW_BUILD_EVAL=ON`.
