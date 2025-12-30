#!/usr/bin/env python3
"""
Faiss Baseline Benchmarks for CP-HNSW Comparison

Runs Faiss HNSW (Float32) and IVF-PQ (GPU) baselines on SIFT-1M.
Results are saved as CSV for merging with CP-HNSW results.

Usage:
    conda activate faiss_bench
    python run_faiss.py --sift ~/datasets/sift1m/sift --output results/

Requirements:
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0
    pip install numpy pandas
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

try:
    import faiss
    HAS_GPU = faiss.get_num_gpus() > 0
except ImportError:
    print("Error: faiss not found. Install with:")
    print("  conda install -c pytorch -c nvidia faiss-gpu=1.8.0")
    exit(1)


def load_fvecs(path):
    """Load fvecs format (SIFT, GIST)."""
    with open(path, 'rb') as f:
        # Read dimension from first record
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        # Calculate number of vectors
        f.seek(0, 2)
        file_size = f.tell()
        record_size = 4 + dim * 4  # int32 dim + float32 * dim
        n = file_size // record_size
        f.seek(0)

        # Read all vectors
        data = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
            data[i] = np.fromfile(f, dtype=np.float32, count=dim)

    return data


def load_ivecs(path):
    """Load ivecs format (ground truth)."""
    with open(path, 'rb') as f:
        k = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        f.seek(0, 2)
        file_size = f.tell()
        record_size = 4 + k * 4
        n = file_size // record_size
        f.seek(0)

        data = np.zeros((n, k), dtype=np.int32)
        for i in range(n):
            kk = np.fromfile(f, dtype=np.int32, count=1)[0]
            data[i] = np.fromfile(f, dtype=np.int32, count=k)

    return data


def normalize_vectors(vecs):
    """L2 normalize vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return vecs / norms


def compute_recall(results, ground_truth, k):
    """Compute Recall@k."""
    total = 0
    for i in range(len(results)):
        retrieved = set(results[i][:k])
        gt = set(ground_truth[i][:k])
        total += len(retrieved & gt) / k
    return total / len(results)


def benchmark_faiss_hnsw(base, queries, gt, M=32, output_csv=None):
    """Benchmark Faiss HNSW (Float32, CPU)."""
    print(f"\n=== Faiss HNSW (M={M}) ===")

    n, dim = base.shape
    nq = len(queries)
    k = 10

    # Build index
    print(f"Building HNSW index (n={n}, dim={dim})...")
    t0 = time.time()

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(base)

    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s ({n/build_time:.0f} vec/s)")

    # Benchmark search
    ef_values = [10, 20, 40, 80, 100, 200, 400]
    results = []

    print(f"\n{'ef':>10} {'Recall@10':>12} {'QPS':>12} {'Latency(us)':>14}")
    print("-" * 50)

    for ef in ef_values:
        index.hnsw.efSearch = ef

        # Warmup
        _, _ = index.search(queries[:10], k)

        # Timed search
        t0 = time.time()
        D, I = index.search(queries, k)
        elapsed = time.time() - t0

        recall = compute_recall(I, gt, k)
        qps = nq / elapsed
        latency_us = elapsed * 1e6 / nq

        print(f"{ef:>10} {recall:>12.4f} {qps:>12.0f} {latency_us:>14.1f}")

        results.append({
            'system': 'Faiss_HNSW',
            'config': f'M={M}',
            'ef_search': ef,
            'recall_10': recall,
            'qps': qps,
            'latency_mean_us': latency_us,
        })

    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return results


def benchmark_faiss_ivfpq_gpu(base, queries, gt, nlist=1024, M_pq=16, output_csv=None):
    """Benchmark Faiss IVF-PQ (GPU)."""
    if not HAS_GPU:
        print("\n=== Faiss IVF-PQ (GPU) - SKIPPED (no GPU) ===")
        return []

    print(f"\n=== Faiss IVF-PQ GPU (nlist={nlist}, M_pq={M_pq}) ===")

    n, dim = base.shape
    nq = len(queries)
    k = 10

    # Build index
    print(f"Building IVF-PQ index...")
    t0 = time.time()

    # Create CPU index first
    quantizer = faiss.IndexFlatIP(dim)
    index_cpu = faiss.IndexIVFPQ(quantizer, dim, nlist, M_pq, 8, faiss.METRIC_INNER_PRODUCT)

    # Train
    index_cpu.train(base)
    index_cpu.add(base)

    # Move to GPU
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")

    # Benchmark search
    nprobe_values = [1, 4, 16, 32, 64, 128]
    results = []

    print(f"\n{'nprobe':>10} {'Recall@10':>12} {'QPS':>12} {'Latency(us)':>14}")
    print("-" * 50)

    for nprobe in nprobe_values:
        index.nprobe = nprobe

        # Warmup
        _, _ = index.search(queries[:100], k)

        # Timed search (batch)
        t0 = time.time()
        D, I = index.search(queries, k)
        elapsed = time.time() - t0

        recall = compute_recall(I, gt, k)
        qps = nq / elapsed
        latency_us = elapsed * 1e6 / nq

        print(f"{nprobe:>10} {recall:>12.4f} {qps:>12.0f} {latency_us:>14.1f}")

        results.append({
            'system': 'Faiss_IVFPQ_GPU',
            'config': f'nlist={nlist}_M={M_pq}',
            'ef_search': nprobe,  # Use ef_search column for nprobe
            'recall_10': recall,
            'qps': qps,
            'latency_mean_us': latency_us,
        })

    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))
        print(f"\nResults appended to: {output_csv}")

    return results


def benchmark_faiss_flat_gpu(base, queries, gt, output_csv=None):
    """Benchmark Faiss Flat (brute force, GPU) for baseline."""
    if not HAS_GPU:
        print("\n=== Faiss Flat (GPU) - SKIPPED (no GPU) ===")
        return []

    print("\n=== Faiss Flat (Brute Force GPU) ===")

    n, dim = base.shape
    nq = len(queries)
    k = 10

    # Build index
    print(f"Building Flat index...")
    t0 = time.time()

    index_cpu = faiss.IndexFlatIP(dim)
    index_cpu.add(base)

    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")

    # Warmup
    _, _ = index.search(queries[:100], k)

    # Timed search
    t0 = time.time()
    D, I = index.search(queries, k)
    elapsed = time.time() - t0

    recall = compute_recall(I, gt, k)
    qps = nq / elapsed
    latency_us = elapsed * 1e6 / nq

    print(f"Recall@{k}: {recall:.4f}")
    print(f"QPS: {qps:.0f}")
    print(f"Latency: {latency_us:.1f} us")

    results = [{
        'system': 'Faiss_Flat_GPU',
        'config': 'brute_force',
        'ef_search': 0,
        'recall_10': recall,
        'qps': qps,
        'latency_mean_us': latency_us,
    }]

    return results


def main():
    parser = argparse.ArgumentParser(description='Faiss Baseline Benchmarks')
    parser.add_argument('--sift', type=str, required=True, help='Path to SIFT-1M directory')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--normalize', action='store_true', help='Normalize vectors to unit length')
    args = parser.parse_args()

    print("=== Faiss Baseline Benchmarks ===\n")
    print(f"Faiss version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU:
        print(f"Number of GPUs: {faiss.get_num_gpus()}")

    # Load dataset
    print(f"\nLoading SIFT-1M from: {args.sift}")
    base = load_fvecs(os.path.join(args.sift, 'sift_base.fvecs'))
    queries = load_fvecs(os.path.join(args.sift, 'sift_query.fvecs'))
    gt = load_ivecs(os.path.join(args.sift, 'sift_groundtruth.ivecs'))

    print(f"  Base: {base.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  Ground truth: {gt.shape}")

    if args.normalize:
        print("\nNormalizing vectors...")
        base = normalize_vectors(base)
        queries = normalize_vectors(queries)

    # Create output directory
    output_dir = os.path.join(args.output, 'exp1_recall_qps')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'faiss_results.csv')

    # Remove old results
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Run benchmarks
    all_results = []

    # HNSW (Float32)
    results = benchmark_faiss_hnsw(base, queries, gt, M=32)
    all_results.extend(results)

    # IVF-PQ (GPU)
    results = benchmark_faiss_ivfpq_gpu(base, queries, gt, nlist=1024, M_pq=16)
    all_results.extend(results)

    # Flat (GPU brute force baseline)
    results = benchmark_faiss_flat_gpu(base, queries, gt)
    all_results.extend(results)

    # Save all results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"\n=== All results saved to: {output_csv} ===")

    print("\n=== Benchmark Complete ===")


if __name__ == '__main__':
    main()
