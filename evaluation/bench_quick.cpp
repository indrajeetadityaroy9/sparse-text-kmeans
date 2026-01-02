/**
 * Quick Benchmark: Minimal test to measure actual build performance
 */
#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <chrono>

using namespace cphnsw;
using namespace cphnsw::eval;

int main(int argc, char** argv) {
    std::cout << "=== Quick Build Benchmark ===\n" << std::flush;

    size_t limit = (argc > 1) ? std::stoull(argv[1]) : 10000;
    std::cout << "Limit: " << limit << " vectors\n" << std::flush;

    // Load SIFT
    std::cout << "Loading SIFT data..." << std::flush;
    size_t dim, count;
    auto vectors = load_fvecs("../evaluation/data/sift/sift_base.fvecs", dim, count);
    count = std::min(count, limit);
    std::cout << " loaded " << count << " x " << dim << "\n" << std::flush;

    // Build index
    std::cout << "Building index (M=32, K=32, ef_c=200)...\n" << std::flush;

    auto start = std::chrono::high_resolution_clock::now();

    CPHNSWIndex<uint8_t, 32> index(dim, 32, 200);
    index.add_batch_parallel(vectors.data(), count);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Build time: " << elapsed << "s\n";
    std::cout << "Throughput: " << (count / elapsed) << " vec/s\n";
    std::cout << "Connected: " << index.verify_connectivity() << "/" << count << "\n";

    return 0;
}
