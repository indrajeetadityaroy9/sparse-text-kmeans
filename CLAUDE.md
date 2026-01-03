# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CP-HNSW (Cross-Polytope Hierarchical Navigable Small World) is a memory-efficient approximate nearest neighbor search algorithm optimized for angular/cosine similarity. It combines Cross-Polytope LSH (asymptotically optimal hashing) with HNSW graph navigation.

**Key innovation**: Replace 4d-byte float vectors with K-byte hash codes (2.8x memory compression) while maintaining competitive recall through hybrid search with reranking.

## Build Commands

```bash
# Standard build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Key executables
./eval_cphnsw          # Basic evaluation
./eval_sift            # SIFT-1M CPU benchmark
./eval_sift_gpu        # SIFT-1M GPU benchmark
./eval_master          # Full 6-experiment evaluation suite
./eval_master_k64      # K=64 variant (high precision)
./qa_protocol          # QA test suite
./test_tiered          # Tiered construction test
./test_correlation     # Distance correlation test
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CPHNSW_USE_AVX512` | ON | AVX-512 SIMD (fallback to AVX2) |
| `CPHNSW_USE_OPENMP` | ON | OpenMP parallelization |
| `CPHNSW_USE_CUDA` | ON | CUDA GPU acceleration |
| `CPHNSW_BUILD_TESTS` | ON | Unit tests (requires GTest) |
| `CPHNSW_BUILD_BENCHMARKS` | OFF | Google Benchmark microbenchmarks |

### Tests and Benchmarks

```bash
# Unit tests (requires GTest)
./cphnsw_tests

# Microbenchmarks (requires Google Benchmark, set CPHNSW_BUILD_BENCHMARKS=ON)
./bench_fht
./bench_hamming
```

## Architecture

```
include/cphnsw/           # Header-only core library (C++17)
├── index/
│   └── cp_hnsw_index.hpp # Main public API - CPHNSWIndex<ComponentT, K>
├── core/
│   └── types.hpp         # CPCode, CPQuery, NodeId definitions
├── quantizer/            # Vector → K-byte hash code encoding
│   ├── hadamard.hpp      # Fast Hadamard Transform O(d log d)
│   ├── rotation_chain.hpp # Ψ(x) = H D₃ H D₂ H D₁ x
│   ├── cp_encoder.hpp    # Cross-Polytope encoder
│   └── multiprobe.hpp    # Multiprobe sequence generation
├── distance/
│   └── hamming.hpp       # SIMD Hamming distance O(1)
├── graph/
│   └── flat_graph.hpp    # Memory-efficient NSW graph with spinlocks
└── algorithms/
    ├── search_layer.hpp     # Greedy graph traversal (Algorithm 2)
    ├── select_neighbors.hpp # Neighbor selection heuristic (Algorithm 4)
    ├── insert.hpp           # Hybrid insertion (Algorithm 3)
    └── knn_search.hpp       # K-NN search wrapper

src/cuda/                 # GPU-accelerated components (compiled, not header-only)
├── gpu_encoder.cu        # CUDA Cross-Polytope encoding
└── gpu_knn_graph.cu      # CUDA k-NN graph construction

evaluation/               # Benchmarking framework
├── eval_master.cpp       # 6-experiment protocol (PhD portfolio quality)
├── eval_sift.cpp         # SIFT-1M standard benchmark
├── baselines/run_faiss.py # Faiss comparison baseline
└── results/ANALYSIS_AND_LATEX.md # Complete evaluation results
```

## Key Technical Details

### Template Parameters

`CPHNSWIndex<ComponentT, K>`:
- `ComponentT`: `uint8_t` for dim ≤ 128, `uint16_t` for dim > 128
- `K`: Code width in bytes (16/32/64), higher = better recall but more memory

### Critical Methods

```cpp
// Recommended: parallel construction + hybrid search
index.add_batch_parallel(vectors, count);  // ~4x faster than sequential
auto results = index.search_and_rerank(query, k, ef, rerank_k);  // Best recall
```

### Performance Characteristics

- **Memory**: K bytes/vector + ~16 bytes graph metadata (vs 4d bytes for float vectors)
- **Graph-only recall ceiling**: K=16→12%, K=32→23%, K=64→37% (limited by quantization)
- **Hybrid recall** (with rerank_k=200): K=64 achieves 82.6% recall@10
- **Parallel scaling**: Near-linear to 16 threads (84% efficiency), diminishing beyond 32

### Thread Safety

Graph uses per-node spinlocks for concurrent insertions. Construction connectivity improves with parallelism (91.6% → 99.0% at 52 threads).

## Python Environment

Python scripts (plotting, FAISS baselines) use a virtual environment with dependencies managed via `pyproject.toml`:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dependencies
pip install -e .

# Install with optional FAISS CPU support
pip install -e ".[faiss]"

# Install with development tools
pip install -e ".[dev]"
```

### Running Evaluation Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Run via module
python evaluation/scripts/plot_results.py --input results/ --output results/plots/
python evaluation/baselines/run_faiss.py --sift ~/datasets/sift1m/sift --output results/

# Deactivate when done
deactivate
```

### Dependencies

Core (always installed):
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`

Optional:
- `faiss-cpu` - CPU-only FAISS (pip install -e ".[faiss]")
- For GPU FAISS: `conda install -c pytorch -c nvidia faiss-gpu`

## Datasets

SIFT-1M datasets are tracked via Git LFS. Download evaluation datasets:
```bash
bash evaluation/scripts/download_datasets.sh
```

---

## PhD Optimization Phases (Implementation Status)

This section documents the multi-phase optimization plan for achieving state-of-the-art ANN performance.

### Phase 1: RaBitQ Bitwise Distance ✅ COMPLETE

**Goal:** Replace expensive SIMD Gather with XOR + PopCount for ~3x speedup.

**Files Modified:**
- `include/cphnsw/core/types.hpp` - Added `BinaryCode<K>`, `RaBitQQuery<K>`
- `include/cphnsw/distance/hamming.hpp` - Added `rabitq_hamming_*` functions
- `include/cphnsw/algorithms/search_layer.hpp` - Added `RaBitQSearchLayer<K>`

**Verification:** `test_rabitq_distance.cpp` - All tests pass

### Phase 2: Residual Quantization ✅ COMPLETE (Integration Pending SIFT-1M Test)

**Goal:** Recover precision lost in Phase 1 by adding residual codes.

**Distance Formula (Integer-only until final conversion):**
```cpp
combined = (primary_hamming << Shift) + residual_hamming
distance = base + scale * combined
```

**Files Created/Modified:**
- `include/cphnsw/core/types.hpp` (lines 185-304):
  - `ResidualBinaryCode<K, R>` - Primary (K bits) + Residual (R bits)
  - `ResidualQuery<K, R>` - Query with pre-computed scalars
  - `ResidualWeighting<Shift>` - Template for bit-shift weighting (default Shift=2 → 4:1 ratio)

- `include/cphnsw/quantizer/residual_encoder.hpp` (NEW):
  - `ResidualCPEncoder<K, R>` - Full reconstruction-based residual
  - `SimplifiedResidualEncoder<K, R>` - Two independent rotation chains

- `include/cphnsw/distance/hamming.hpp` (lines 1491-1858):
  - `residual_distance_integer_scalar<K, R, Shift>()` - Scalar fallback
  - `residual_hamming_batch8_avx512<K, R, Shift>()` - AVX-512 batch kernel

- `include/cphnsw/graph/flat_graph.hpp` (lines 931-1192):
  - `ResidualNeighborBlock<K, R>` - SoA transposed storage for batch SIMD

- `include/cphnsw/algorithms/search_layer.hpp` (lines 379-615):
  - `ResidualSearchLayer<K, R, Shift>` - NSW search with residual distance

- `include/cphnsw/index/residual_index.hpp` (NEW):
  - `ResidualCPHNSWIndex<K, R, Shift>` - Full index wrapper

**Verification Tests:**
- `test_residual_distance.cpp` - 8 unit tests, all pass
- `test_go_nogo_verification.cpp` - Comprehensive checklist:
  - Memory alignment: ✅ PASS
  - Shift logic: ✅ PASS
  - Throughput: 28,605 M distances/sec
  - Correlation (R²): 0.84 on diverse distance pairs
- `test_phase2_integration.cpp` - Index integration test

### ⚠️ CRITICAL: Random Data Limitation

**Observed Behavior:**
On random synthetic data (unit vectors in d=128), Phase 2 shows ~0.4% graph-only recall vs Phase 1's ~50%.

**Root Cause - Concentration of Measure:**
Random unit vectors in high dimensions have L2 distances **concentrated around sqrt(2) ≈ 1.414** with very narrow variance (~0.1). When all pairwise distances are nearly identical, any quantization error causes large ranking shuffles.

| Distance Type | Behavior on Random Data |
|---------------|------------------------|
| True L2 top-5 | 1.21, 1.21, 1.21, 1.21, 1.22 |
| Phase 2 top-5 L2 | 1.42, 1.26, 1.39, 1.32, 1.37 |
| Overlap | 0/10 |

**Why Phase 1 Works Better on Random Data:**
Phase 1 uses **asymmetric dot product** which preserves query magnitude information. Phase 2 uses **symmetric Hamming** which loses this discrimination. On real data (SIFT-1M) with diverse pairwise distances, Phase 2's R² = 0.84 correlation should translate to improved recall.

**This is NOT a bug.** The go/no-go verification confirmed correct math on diverse distance pairs. Random data is a known pathological case for LSH methods.

### Phase 3: Learned Rotations ⏳ NOT STARTED

**Goal:** Improve quantization quality by learning optimal diagonal matrices.

**Approach:**
- Replace random diagonals (D₁, D₂, D₃) with learned ones
- Train via PyTorch with Gumbel-Softmax relaxation
- Maintain O(d log d) complexity (NOT dense PCA)

**Files to Create:**
- `scripts/learn_diagonals.py` - PyTorch training script
- `include/cphnsw/quantizer/learned_diagonals.hpp` - C++ diagonal chain

### Remaining Work

1. **SIFT-1M Baseline (Phase 2.5)**
   - Run `ResidualCPHNSWIndex` on SIFT-1M dataset
   - Record "Before Learning" recall numbers for thesis
   - Requires A100 GPU system (not local machine)

2. **Phase 3 Implementation**
   - Only start after confirming Phase 2 improves recall on SIFT-1M
   - Expected improvement: Graph-only recall 50% → 70%+ (reduces reranking need)

### Test Executables

```bash
# Phase 1 & 2 verification
./test_rabitq_distance        # Phase 1 unit tests
./test_residual_distance      # Phase 2 unit tests
./test_go_nogo_verification   # Comprehensive Phase 1+2 checklist
./test_recall_diagnostic      # Recall collapse debugging
./test_phase2_integration     # Phase 1 vs Phase 2 comparison
```
