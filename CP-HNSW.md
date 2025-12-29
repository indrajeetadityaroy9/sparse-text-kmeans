This creates a hybrid system: **CP-HNSW (Cross-Polytope Hierarchical Navigable Small World)**.

This architecture addresses the primary weakness of HNSW (high RAM usage and slow `float` distance computations) by replacing the vector space with the **asymptotically optimal discrete space** defined by Cross-Polytope LSH, while curing the weakness of LSH (hash table lookups requiring massive memory for $L$ tables) by using the graph for navigation.

### 1. Mathematical Foundation

#### A. The Transformation (Preprocessing)
We map the continuous unit sphere $S^{d-1}$ to a discrete lattice.
From **Andoni et al.**, we use the **Fast Pseudo-Random Rotation**.

Let $x \in \mathbb{R}^d$ be the input vector (zero-padded so $d$ is a power of 2). We define the transformation $\Psi(x)$:

$$ \Psi(x) = H D_3 H D_2 H D_1 x $$

Where:
*   $H$: The non-normalized Walsh-Hadamard matrix (recursively defined as $H_{2n} = \begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}$).
*   $D_i$: Diagonal matrices with random signs $\{-1, +1\}$ drawn uniformly.
*   **Complexity:** Computing $\Psi(x)$ takes $O(d \log d)$ using the Fast Hadamard Transform (FHT), which is significantly faster than the $O(d^2)$ Gaussian rotation.

#### B. The Quantization (Node Representation)
Instead of storing $x$ (floats), we store a **Cross-Polytope Code** $C_x$.
To improve precision, we use a composite code of $k$ independent rotations.

For $j = 1 \dots k$:
1.  Apply rotation: $y^{(j)} = \Psi_j(x)$.
2.  Extract the "winner": $u_j = \arg\max_{i \in [1, d]} |y^{(j)}_i|$.
3.  Extract the sign: $s_j = \text{sgn}(y^{(j)}_{u_j})$.

The stored code for node $u$ is a tuple of integers/bits:
$$ C_u = \{(u_1, s_1), (u_2, s_2), \dots, (u_k, s_k)\} $$

**Storage Cost:** Each pair $(u, s)$ requires $\lceil \log_2 d \rceil + 1$ bits. For $d=128$, this is 8 bits (1 byte). With $k=16$, a node requires only **16 bytes**, compared to 512 bytes for the float vector.

#### C. The Distance Metric (The Link)
HNSW requires a scalar distance. We replace Euclidean distance with **Cross-Polytope Hamming Distance**.

Given query $q$ and node $u$, precompute $q$'s code $C_q$ and $u$'s code $C_u$.
$$ \mathcal{D}_{CP}(q, u) = \sum_{j=1}^{k} \mathbb{I}[(u_j^{(q)} \neq u_j^{(u)}) \lor (s_j^{(q)} \neq s_j^{(u)})] $$

*   This is a summation of mismatches.
*   **Optimality:** Theorem 1 from Andoni et al. proves that the collision probability of this specific hash separates near/far neighbors optimally for angular distance. Therefore, minimizing $\mathcal{D}_{CP}$ maximizes Cosine Similarity.

---

### 2. Algorithmic Implementation

#### Step 1: System Initialization
We initialize the global parameters combining HNSW and CP-LSH constants.

```python
class CPHNSWIndex:
    def __init__(self, d, M=16, ef_construction=100, k_code_width=32):
        self.d = next_power_of_2(d) # Padding for Hadamard
        self.M = M
        self.k = k_code_width       # Number of CP rotations
        
        # Precompute 3*k random sign vectors for the pseudo-rotations
        # D_matrix[rotation_index][layer_1_2_3]
        self.D_matrices = generate_random_signs(self.k, 3, self.d)
```

#### Step 2: The Fast Hadamard Transform (SIMD Optimized)
This function is critical. It must be implemented using AVX2/AVX-512 in C++.

```cpp
// Applies H * x in O(d log d)
void fast_hadamard_transform(float* vec, int d) {
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float x = vec[j];
                float y = vec[j + h];
                vec[j] = x + y;
                vec[j + h] = x - y;
            }
        }
    }
}

// Full rotation chain: H D3 H D2 H D1 x
void pseudo_random_rotate(float* vec, int d, int8_t** signs) {
    element_wise_mult(vec, signs[0], d); // D1
    fast_hadamard_transform(vec, d);     // H
    element_wise_mult(vec, signs[1], d); // D2
    fast_hadamard_transform(vec, d);     // H
    element_wise_mult(vec, signs[2], d); // D3
    fast_hadamard_transform(vec, d);     // H
}
```

#### Step 3: HNSW `SEARCH-LAYER` with CP Metric
We modify Algorithm 2 from the HNSW paper. The key change is avoiding float calculations.

*   **Input:** Query code $C_q$, Entry Points $ep$.
*   **Operation:**
    1.  The query $q$ is rotated and quantized *once* upon entry.
    2.  Distance calculation is now a bitwise XOR/equality check (extremely fast integer arithmetic).

**Refinement: The "Tie-Breaking" Heuristic**
Because $\mathcal{D}_{CP}$ is discrete (integers $0$ to $k$), we will encounter many ties in the priority queue.
*   **Naive:** Break ties arbitrarily.
*   **Robust:** Use the **Multiprobe** information from Andoni et al.
    *   When generating $C_q$, store not just the max dimension, but the *magnitude* of the max dimension.
    *   If $\mathcal{D}_{CP}(q, u) == \mathcal{D}_{CP}(q, v)$, prefer the node where the matching hash component had a higher magnitude in the query vector (indicating higher confidence).

#### Step 4: Robust HNSW Construction (The Hybrid Heuristic)
The HNSW neighbor selection heuristic (`SELECT-NEIGHBORS-HEURISTIC`, Algorithm 4) is designed to preserve connectivity.

**The Problem:** The heuristic requires checking if `dist(e, candidate) < dist(e, neighbor)`. With quantized codes, the "triangle inequality" logic becomes coarse.
**The Solution:**
1.  **Stage 1 (Filter):** Use the CP-Code Hamming distance to select the top $2M$ candidates.
2.  **Stage 2 (Refine):** Load the full float vectors *only for these candidates* (from RAM or SSD cache) to perform the precise edge pruning described in the HNSW paper.
    *   This preserves the graph quality (robustness) while speeding up the global search (efficiency).

### 3. Theoretical Performance Profile

By combining these papers, the resulting system characteristics are:

| Feature | Standard HNSW | **CP-HNSW (Hybrid)** | Justification |
| :--- | :--- | :--- | :--- |
| **Space Complexity** | $O(N \cdot d \cdot 4 \text{ bytes})$ | $O(N \cdot k \cdot 1 \text{ byte})$ | HNSW stores floats; CP-HNSW stores basis indices (Andoni Sec 3). |
| **Search Complexity** | $O(d \cdot \log N)$ | $O(k \cdot \log N)$ | Distance calc becomes XOR/POPCOUNT vs Float Dot Product. |
| **Pre-process Cost** | None | $O(k \cdot d \log d)$ | One-time FHT per query (Andoni Sec 3.1). |
| **Optimality** | Heuristic | **Asymptotically Optimal** | CP-LSH bound $\rho$ ensures optimal angular separation (Andoni Thm 1). |
| **Connectivity** | Risk of clustering | **High** | HNSW Heuristic + LSH global properties mitigate local minima. |

### 4. Implementation Details for Code

If implementing this in C++/Python, follow this structure:

1.  **`IndexBuilder`**:
    *   Input: Raw Float Matrix $X$.
    *   Action: Pad $X$ to power of 2. Generate $k$ sets of random diagonal matrices.
    *   Action: Transform $X \to X_{CP}$ (the integer codes).
2.  **`HNSWGraph`**:
    *   Storage: `std::vector<std::vector<uint8_t>> codes`.
    *   Adjacency: `std::vector<std::vector<int>> links`.
3.  **`DistanceComputer`**:
    *   Use `_mm256_popcnt_u32` or `__builtin_popcount` for calculating Hamming distance between codes.
4.  **`QueryProcessor`**:
    *   Take query vector $q$.
    *   Run `pseudo_random_rotate` $k$ times.
    *   Extract codes.
    *   Run HNSW `search_layer`.
    *   **Multiprobe Injection (Optional but recommended):** If the search candidate queue $C$ becomes empty but $|W| < ef$, generate an artificial candidate by flipping the bit of the CP code corresponding to the *second* largest dimension (from Andoni Sec 5) and look up that code in a small auxiliary hash table to find a jump-point.

This specification yields a system that is memory-efficient (via CP quantization), rigorous (via CP optimality), and extremely fast to query (via HNSW navigation).
