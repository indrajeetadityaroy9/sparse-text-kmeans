# Custom K-Means for Text
A from-scratch K-Means implementation specialized for text vectors (sparse and dense), plus a lightweight evaluation suite.

## Technical Components

- Core algorithms
  - Sparse-aware K-Means tuned for high-dimensional text vectors.
  - k-means++ or random initialization
  - Multiple restarts (`n_init`) with best-inertia selection
  - Farthest-point fallback for empty clusters
  - Convergence by centroid shift and label stability
  - Squared Euclidean distances and optional distance batching for memory control

- Evaluation
  - `data.py`: Vectorization helpers (TF‑IDF/Count) and optional dimensionality reduction (TruncatedSVD/PCA). Returns a `DatasetBundle` with matrix, labels, and metadata. Swap in your own loader to use a different corpus.
  - `metrics.py`: Homogeneity, Completeness, V‑measure, Adjusted Rand Index, Silhouette; plus aggregation across runs.
  - `evaluation.py`: Orchestrates experiments across seeds and model specs; writes JSON artifacts.
  - `significance.py`: Bootstrap confidence intervals and paired significance tests between models.
  - `experiment_configs.py`: Canonical vectorizer/model configurations for baseline runs.
  - `run_experiments.py`: CLI entrypoint for end-to-end comparison and summary rendering.


- Squared Euclidean distances are used throughout for performance and numerical stability.
- Optional `distance_batch_size` avoids materializing full distance matrices for large `n`.
- Dimensionality reduction outputs are kept dense for speed; upstream data can remain sparse.
