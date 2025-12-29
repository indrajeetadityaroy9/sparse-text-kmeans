#include "../include/cphnsw/index/cp_hnsw_index.hpp"
#include "datasets/dataset_loader.hpp"
#include "metrics/recall.hpp"
#include <iostream>
#include <iomanip>

using namespace cphnsw;
using namespace cphnsw::eval;

int main() {
    std::cout << "=== DIAGNOSTIC TEST ===\n\n";
    
    size_t n = 1000;
    size_t dim = 128;
    size_t k = 10;
    
    // Generate dataset
    Dataset dataset = generate_random_sphere(n, dim, 10, k);
    
    // Build index
    CPHNSWIndex8 index(dim, 16, 100);
    index.add_batch(dataset.base_vectors.data(), dataset.num_base);
    
    std::cout << "Index built. Connectivity: " << index.verify_connectivity() << "/" << n << "\n\n";
    
    // Test first query
    const Float* query = dataset.get_query(0);
    
    // Ground truth for first query
    std::cout << "Ground truth (first 5 of k=" << k << "):\n";
    for (size_t i = 0; i < std::min(size_t(5), dataset.ground_truth[0].size()); ++i) {
        NodeId id = dataset.ground_truth[0][i];
        const Float* base = dataset.get_base(id);
        Float dot = 0;
        for (size_t j = 0; j < dim; ++j) dot += query[j] * base[j];
        std::cout << "  ID=" << id << " dot=" << std::fixed << std::setprecision(6) << dot << "\n";
    }
    
    // HNSW search result
    std::cout << "\nHNSW search results (ef=100):\n";
    auto results = index.search(query, k, 100);
    for (size_t i = 0; i < std::min(size_t(5), results.size()); ++i) {
        NodeId id = results[i].id;
        const Float* base = dataset.get_base(id);
        Float dot = 0;
        for (size_t j = 0; j < dim; ++j) dot += query[j] * base[j];
        std::cout << "  ID=" << id << " result.dist=" << std::fixed << std::setprecision(6) 
                  << results[i].distance << " true_dot=" << dot << "\n";
    }
    
    // Brute force search for comparison
    std::cout << "\nBrute-force CP search results:\n";
    auto bf_results = index.brute_force_search(query, k);
    for (size_t i = 0; i < std::min(size_t(5), bf_results.size()); ++i) {
        NodeId id = bf_results[i].id;
        const Float* base = dataset.get_base(id);
        Float dot = 0;
        for (size_t j = 0; j < dim; ++j) dot += query[j] * base[j];
        std::cout << "  ID=" << id << " result.dist=" << std::fixed << std::setprecision(6)
                  << bf_results[i].distance << " true_dot=" << dot << "\n";
    }
    
    // Compute recalls
    double recall_hnsw = compute_recall(results, dataset.ground_truth[0], k);
    double recall_bf = compute_recall(bf_results, dataset.ground_truth[0], k);
    
    std::cout << "\n=== RECALL ===\n";
    std::cout << "HNSW Recall@" << k << ": " << recall_hnsw << "\n";
    std::cout << "BruteForce CP Recall@" << k << ": " << recall_bf << "\n";
    
    return 0;
}
