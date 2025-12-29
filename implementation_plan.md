This is a comprehensive research implementation and evaluation plan for **CP-HNSW (Cross-Polytope Hierarchical Navigable Small World)**.

This research aims to validate the hypothesis that combining the **optimal angular hashing** of Cross-Polytope LSH with the **navigable graph structure** of HNSW yields a superior index for high-dimensional cosine similarity search, specifically in memory-constrained environments.

---

# 1. Architecture Design
The system consists of three distinct modules to ensure modularity and ease of ablation testing.

### Module A: The Quantizer (CP-LSH Core)
*   **Responsibility:** Maps $\mathbb{R}^d \to \mathbb{Z}^k$.
*   **Input:** Raw float vector $x$.
*   **Process:**
    1.  Zero-pad $x$ to next power of 2 ($d'$).
    2.  Apply **Fast Hadamard Transform (FHT)** based rotation: $x' = H D_3 H D_2 H D_1 x$.
    3.  **argmax Encoding:** For each of $k$ rotations, find the index of the maximum absolute component and its sign.
*   **Output:** A "Sketch" (compact byte array representing the code).

### Module B: The Graph Index (HNSW Core)
*   **Responsibility:** Stores topology and executes navigation.
*   **Storage:**
    *   **Nodes:** Store *only* the Quantized Sketches from Module A (no floats).
    *   **Edges:** Adjacency lists for layers $l=0 \dots L$.
*   **Metric:** Hamming Distance on Sketches (XOR + Popcount).

### Module C: The Hybrid Builder
*   **Responsibility:** Constructs the graph quality.
*   **Strategy:** "Quantized Search, Precise Linkage".
    *   During insertion, use the fast Sketch metric to traverse to the neighborhood.
    *   (Optional Control) When selecting final neighbors to connect, momentarily load full float vectors (or use a higher-precision sketch) to ensure the HNSW "diversity heuristic" (Algorithm 4) works correctly.

---

# 2. Implementation Steps

## Phase 1: Mathematical Primitives (Python Prototype)
*Goal: Validate the Andoni et al. collision probabilities and correctness of FHT.*

1.  **FHT Implementation:** Implement the non-recursive Fast Hadamard Transform in Python (`numpy`).
    *   *Verification:* Compare $H \cdot x$ against a naive matrix multiplication of the Hadamard matrix.
2.  **Rotation Chain:** Implement $x \mapsto H D_3 H D_2 H D_1 x$ where $D_i$ are fixed random diagonal sign matrices.
3.  **Collision Test:**
    *   Generate random unit vectors with fixed angular distance $\theta$.
    *   Compute their CP-codes.
    *   Plot **Hamming Distance vs. Angular Distance**.
    *   *Success Criterion:* The curve must match the theoretical bounds derived in Andoni's paper (Theorem 1).

## Phase 2: High-Performance Core (C++)
*Goal: Build the engine capable of handling 1M+ vectors.*

1.  **SIMD FHT:** Implement the Hadamard transform using AVX2 or AVX-512 intrinsics.
    *   *Note:* FHT is addition/subtraction heavy. AVX allows processing 8 or 16 floats simultaneously.
2.  **Bit-Packed Storage:** Create a `struct Node` that holds the CP-codes.
    *   If $d=128$, the code index is 7 bits + 1 sign bit = 8 bits (1 byte).
    *   For $k=3$ rotations, storage is just 3 bytes per node.
3.  **Distance Kernel:** Implement `dist_cp(Node a, Node b)`.
    *   Use `_mm_xor_si128` and `_mm_popcnt_u64` to compute distance in single-digit CPU cycles.

## Phase 3: Graph Construction (Hybrid Logic)
*Goal: Port Malkov’s HNSW logic to use the CP kernel.*

1.  **Level Generation:** Implement $l = \lfloor -\ln(unif(0,1)) \cdot m_L \rfloor$.
2.  **Greedy Search (`SEARCH-LAYER`):**
    *   Standard HNSW logic, but replacing `L2` distance with `dist_cp`.
3.  **Neighbor Selection (`SELECT-NEIGHBORS-HEURISTIC`):**
    *   *Critical Detail:* The heuristic relies on the triangle inequality. The Hamming distance on CP codes satisfies this.
    *   Implement the "keepPrunedConnections" logic from HNSW to ensure connectivity.

---

# 3. Evaluation Plan

## A. Datasets
Select datasets that stress different aspects of the index (angular distribution, dimensionality).

| Dataset | Dimensions | Type | Why? |
| :--- | :--- | :--- | :--- |
| **SIFT-1M** | 128 | Vector (Visual) | Standard HNSW benchmark. Dense, clustered. |
| **GloVe-100** | 100 | Cosine (NLP) | **Target Use Case.** Pure angular distance. |
| **GIST-1M** | 960 | High-Dim | Tests effectiveness of LSH rotation in high $d$. |
| **Random-S** | 1024 | Synthetic Sphere | Validates the "Curse of Dimensionality" resilience. |

## B. Baselines
1.  **HNSW (Float32):** The original implementation (nmslib/faiss). Represents the "Speed/Recall Ceiling" (fastest, best recall, high RAM).
2.  **Faiss IVFPQ:** The standard for compressed search. Represents the "Memory Baseline".
3.  **CP-LSH (Stand-alone):** Implementation of Andoni’s paper without the graph. Represents the "hashing baseline".

## C. Metrics
1.  **Recall@N:** (Intersection of returned neighbors with ground truth) / N.
2.  **Queries Per Second (QPS):** Throughput on a single thread.
3.  **Memory Footprint (GB):** Total RAM usage excluding raw data.
4.  **Construction Time:** Wall-clock time to build the index.

## D. Experiments

### Experiment
