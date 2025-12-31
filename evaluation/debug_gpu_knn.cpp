/**
 * Debug GPU k-NN Output
 *
 * Checks if the GPU k-NN kernel is producing valid neighbor indices.
 */

#include <algorithm>  // Must be first for partial_sort
#include "../include/cphnsw/cuda/gpu_knn_graph.cuh"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <unordered_set>

using namespace cphnsw;
using namespace cphnsw::eval;

int main() {
    std::cout << "GPU k-NN Debug\n";
    std::cout << "==============\n\n";

    // Load small subset for debugging
    std::cout << "Loading SIFT-1M...\n";
    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");

    // Use first 10K vectors for quick test
    size_t N = 10000;
    size_t dim = dataset.dim;
    size_t k = 32;

    std::cout << "Testing with N=" << N << ", k=" << k << "\n\n";

    // Build k-NN graph
    std::vector<uint32_t> neighbors(N * k);
    std::vector<float> distances(N * k);

    cuda::GPUKNNGraphBuilder builder(dim, k);
    builder.build(dataset.base_vectors.data(), N, neighbors.data(), distances.data());

    // Analyze output
    std::cout << "Analyzing k-NN output:\n";

    // Count valid/invalid entries
    size_t valid_count = 0;
    size_t invalid_count = 0;
    size_t self_loops = 0;
    size_t out_of_range = 0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < k; ++j) {
            uint32_t neighbor = neighbors[i * k + j];
            if (neighbor == UINT32_MAX) {
                invalid_count++;
            } else if (neighbor >= N) {
                out_of_range++;
            } else if (neighbor == i) {
                self_loops++;
            } else {
                valid_count++;
            }
        }
    }

    size_t total = N * k;
    std::cout << "  Valid neighbors:   " << valid_count << " / " << total
              << " (" << (100.0 * valid_count / total) << "%)\n";
    std::cout << "  Invalid (MAX):     " << invalid_count << "\n";
    std::cout << "  Out of range:      " << out_of_range << "\n";
    std::cout << "  Self-loops:        " << self_loops << "\n\n";

    // Show first few nodes
    std::cout << "Sample nodes (first 5):\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "  Node " << i << ": neighbors = [";
        for (size_t j = 0; j < std::min(k, size_t(8)); ++j) {
            if (j > 0) std::cout << ", ";
            uint32_t n = neighbors[i * k + j];
            if (n == UINT32_MAX) {
                std::cout << "INV";
            } else {
                std::cout << n;
            }
        }
        if (k > 8) std::cout << ", ...";
        std::cout << "]\n";

        std::cout << "         distances = [";
        for (size_t j = 0; j < std::min(k, size_t(8)); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(3) << distances[i * k + j];
        }
        if (k > 8) std::cout << ", ...";
        std::cout << "]\n";
    }

    // Check connectivity via simple BFS
    std::cout << "\nConnectivity check (BFS from node 0):\n";
    std::unordered_set<uint32_t> visited;
    std::vector<uint32_t> queue;
    queue.push_back(0);
    visited.insert(0);

    size_t qi = 0;
    while (qi < queue.size() && visited.size() < N) {
        uint32_t node = queue[qi++];
        for (size_t j = 0; j < k; ++j) {
            uint32_t neighbor = neighbors[node * k + j];
            if (neighbor < N && visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    std::cout << "  Reachable from node 0: " << visited.size() << " / " << N
              << " (" << (100.0 * visited.size() / N) << "%)\n";

    // Verify distances make sense (should be negative for cosine similarity)
    float min_dist = 1e30f, max_dist = -1e30f;
    for (size_t i = 0; i < N * k; ++i) {
        if (neighbors[i] != UINT32_MAX && neighbors[i] < N) {
            min_dist = std::min(min_dist, distances[i]);
            max_dist = std::max(max_dist, distances[i]);
        }
    }
    std::cout << "\nDistance range: [" << min_dist << ", " << max_dist << "]\n";
    std::cout << "(Negative = high similarity, Positive = low similarity)\n";

    return 0;
}
