import time
import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.metrics import euclidean_distances


class FitResult:
    def __init__(self, iterations, converged, runtime_sec):
        self.iterations = iterations
        self.converged = converged
        self.runtime_sec = runtime_sec


class CustomKMeans:
    def __init__(
        self,
        n_clusters,
        max_iterations=300,
        tolerance=1e-4,
        random_state=None,
        init="k-means++",
        n_init=10,
        distance_batch_size=None,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if init not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'")
        if n_init <= 0:
            raise ValueError("n_init must be positive")

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.distance_batch_size = distance_batch_size

        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.fit_result_ = None

        self._rng = np.random.RandomState(self.random_state)

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter {key}")
            setattr(self, key, value)
        if "random_state" in params:
            self._rng = np.random.RandomState(self.random_state)
        return self

    def _sample_indices(self, n_samples, size=1):
        return self._rng.choice(n_samples, size=size, replace=False)

    def _init_centroids(self, X):
        if self.init == "random":
            indices = self._sample_indices(X.shape[0], size=self.n_clusters)
            centroids = X[indices]
        else:
            centroids = self._init_centroids_kmeanspp(X)
        if not issparse(centroids):
            centroids = csr_matrix(centroids)
        return centroids

    def _init_centroids_kmeanspp(self, X):
        n_samples = X.shape[0]
        first_idx = int(self._sample_indices(n_samples)[0])
        if issparse(X):
            centroids = [X[first_idx]]
        else:
            centroids = [X[first_idx, :][None, :]]

        sq_dist = euclidean_distances(X, centroids[0], squared=True).reshape(-1)

        for _ in range(1, self.n_clusters):
            prob = sq_dist / sq_dist.sum()
            next_idx = int(self._rng.choice(n_samples, p=prob))
            if issparse(X):
                centroids.append(X[next_idx])
            else:
                centroids.append(X[next_idx, :][None, :])
            new_sq = euclidean_distances(X, centroids[-1], squared=True).reshape(-1)
            sq_dist = np.minimum(sq_dist, new_sq)

        if issparse(X):
            return vstack(centroids)
        dense_centroids = np.vstack([
            c.toarray().ravel() if issparse(c) else np.asarray(c).ravel()
            for c in centroids
        ])
        return csr_matrix(dense_centroids)

    def fit(self, X):
        overall_start = time.perf_counter()
        base_rng = np.random.RandomState(self.random_state)
        seeds = base_rng.randint(0, np.iinfo(np.int32).max, size=self.n_init)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_iterations = None
        best_converged = None
        best_runtime = None

        for seed in seeds:
            rng = np.random.RandomState(int(seed))
            centroids, labels, inertia, iterations, converged, runtime = self._fit_single(X, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_iterations = iterations
                best_converged = converged
                best_runtime = runtime

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.fit_result_ = FitResult(
            iterations=int(best_iterations) if best_iterations is not None else None,
            converged=bool(best_converged) if best_converged is not None else None,
            runtime_sec=float(best_runtime) if best_runtime is not None else float(time.perf_counter() - overall_start),
        )
        return self

    def _fit_single(self, X, rng):
        start_time = time.perf_counter()
        # Initialization
        if self.init == "random":
            # Temporarily use rng for random init
            prev_rng = self._rng
            self._rng = rng
            centroids = self._init_centroids(X)
            self._rng = prev_rng
        else:
            # Use k-means++ with rng by temporarily overriding
            prev_rng = self._rng
            self._rng = rng
            centroids = self._init_centroids_kmeanspp(X)
            self._rng = prev_rng

        prev_labels = None
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            labels, closest_dist_sq = self._labels_and_closest_sq(X, centroids)

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                converged = True
                break

            # Build new centroids and handle empty clusters via farthest-point reassignment
            n_samples = X.shape[0]
            available = np.argsort(closest_dist_sq)[::-1]
            used = set()
            new_centroids = []
            for cluster_idx in range(self.n_clusters):
                mask = labels == cluster_idx
                if mask.any():
                    cluster_points = X[mask]
                    centroid = cluster_points.mean(axis=0)
                    if not issparse(centroid):
                        centroid = csr_matrix(centroid)
                    new_centroids.append(centroid)
                else:
                    reassigned_idx = self._fallback_index(iter(available), used, rng, n_samples)
                    row = X[reassigned_idx]
                    if not issparse(row):
                        row = csr_matrix(row)
                    new_centroids.append(row)

            new_centroids = vstack(new_centroids)
            shift = new_centroids - centroids
            if issparse(shift):
                centroid_shift = float(np.sqrt(shift.multiply(shift).sum()))
            else:
                centroid_shift = float(np.linalg.norm(shift))
            centroids = new_centroids

            if centroid_shift <= self.tolerance:
                converged = True
                break

            prev_labels = labels
        else:
            iteration = self.max_iterations

        inertia = float(self._inertia_from_labels(X, centroids, labels))
        runtime = time.perf_counter() - start_time
        return centroids, labels, inertia, iteration, converged, runtime

    def _labels_and_closest_sq(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)
        closest = np.empty(n_samples, dtype=float)
        batch = self.distance_batch_size
        if not batch:
            d2 = euclidean_distances(X, centroids, squared=True)
            labels[:] = np.argmin(d2, axis=1)
            closest[:] = d2[np.arange(n_samples), labels]
            return labels, closest
        for start in range(0, n_samples, batch):
            end = min(start + batch, n_samples)
            d2 = euclidean_distances(X[start:end], centroids, squared=True)
            local_labels = np.argmin(d2, axis=1)
            labels[start:end] = local_labels
            closest[start:end] = d2[np.arange(end - start), local_labels]
        return labels, closest

    def _inertia_from_labels(self, X, centroids, labels):
        n_samples = X.shape[0]
        batch = self.distance_batch_size
        total = 0.0
        if not batch:
            d2 = euclidean_distances(X, centroids, squared=True)
            total = float(np.sum(d2[np.arange(n_samples), labels]))
            return total
        for start in range(0, n_samples, batch):
            end = min(start + batch, n_samples)
            d2 = euclidean_distances(X[start:end], centroids, squared=True)
            local_labels = labels[start:end]
            total += float(np.sum(d2[np.arange(end - start), local_labels]))
        return total

    def _compute_inertia(self, X):
        labels, closest = self._labels_and_closest_sq(X, self.centroids)
        return float(np.sum(closest))

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted")
        d2 = euclidean_distances(X, self.centroids, squared=True)
        return np.argmin(d2, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _fallback_index(self, iterator, used, rng, n_samples):
        for idx in iterator:
            idx = int(idx)
            if idx not in used:
                used.add(idx)
                return idx
        rand = int(rng.randint(n_samples))
        used.add(rand)
        return rand

