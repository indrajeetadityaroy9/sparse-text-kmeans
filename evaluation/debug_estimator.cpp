/**
 * Debug CP Estimator
 *
 * Isolates the encoding/distance computation from graph/GPU to identify
 * the source of R² ≈ 0.04 (encoding producing random noise).
 */

#include "../include/cphnsw/legacy/quantizer/cp_encoder.hpp"
#include "../include/cphnsw/legacy/distance/hamming.hpp"
#include "../include/cphnsw/graph/flat_graph.hpp"
#include "datasets/dataset_loader.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace cphnsw;
using namespace cphnsw::eval;

constexpr size_t K = 32;
using Encoder = CPEncoder<uint8_t, K>;
using Code = CPCode<uint8_t, K>;
using Query = CPQuery<uint8_t, K>;

// Compute true cosine similarity (dot product for normalized vectors)
float true_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

void test_identity(Encoder& encoder, const float* vec, size_t dim) {
    std::cout << "\n=== Test 1: Identity Verification ===\n";
    std::cout << "Encoding vector X, then computing Score(X, Code_X)\n";

    // Encode vector as query (with rotated vectors)
    Query query = encoder.encode_query(vec);

    // Encode same vector as code
    Code code = encoder.encode(vec);

    // Compute asymmetric distance
    float asym_dist = asymmetric_search_distance(query, code);

    // Also compute with scalar explicitly
    float scalar_score = 0.0f;
    for (size_t r = 0; r < K; ++r) {
        uint8_t raw = code.components[r];
        size_t idx = raw >> 1;
        bool is_neg = raw & 1;
        float val = query.rotated_vecs[r][idx];
        scalar_score += is_neg ? -val : val;
    }

    std::cout << "  Asymmetric distance (library): " << asym_dist << "\n";
    std::cout << "  Scalar score (manual):         " << -scalar_score << " (negated)\n";
    std::cout << "  Raw score (before negate):     " << scalar_score << "\n";

    // Check query encoding details
    std::cout << "\n  Query encoding analysis:\n";
    float sum_magnitudes = 0.0f;
    for (size_t r = 0; r < K; ++r) {
        sum_magnitudes += query.magnitudes[r];
    }
    std::cout << "  Sum of magnitudes: " << sum_magnitudes << "\n";
    std::cout << "  First 5 magnitudes: ";
    for (size_t r = 0; r < 5; ++r) {
        std::cout << query.magnitudes[r] << " ";
    }
    std::cout << "\n";

    // Expected: For identity, score should be sum of magnitudes (max possible)
    std::cout << "\n  Expected: score ≈ sum_magnitudes = " << sum_magnitudes << "\n";
    std::cout << "  Actual:   score = " << scalar_score << "\n";

    if (std::abs(scalar_score - sum_magnitudes) < 0.01f * sum_magnitudes) {
        std::cout << "  [PASS] Identity score matches expected\n";
    } else {
        std::cout << "  [FAIL] Identity score mismatch! Encoding or distance is broken.\n";
    }
}

void test_neighbor_discrimination(Encoder& encoder, const float* vectors,
                                   size_t num_vecs, size_t dim) {
    std::cout << "\n=== Test 2: Neighbor Discrimination ===\n";
    std::cout << "Compare Score(0, close_neighbor) vs Score(0, distant_vector)\n";

    const float* vec0 = vectors;

    // Find closest and farthest vectors to vec0
    float max_sim = -1e30f, min_sim = 1e30f;
    size_t closest_idx = 1, farthest_idx = 1;

    for (size_t i = 1; i < std::min(num_vecs, size_t(10000)); ++i) {
        float sim = true_similarity(vec0, vectors + i * dim, dim);
        if (sim > max_sim) {
            max_sim = sim;
            closest_idx = i;
        }
        if (sim < min_sim) {
            min_sim = sim;
            farthest_idx = i;
        }
    }

    std::cout << "  Closest neighbor: idx=" << closest_idx << ", sim=" << max_sim << "\n";
    std::cout << "  Farthest vector:  idx=" << farthest_idx << ", sim=" << min_sim << "\n";

    // Encode vec0 as query
    Query query0 = encoder.encode_query(vec0);

    // Encode neighbors as codes
    Code code_close = encoder.encode(vectors + closest_idx * dim);
    Code code_far = encoder.encode(vectors + farthest_idx * dim);

    float dist_close = asymmetric_search_distance(query0, code_close);
    float dist_far = asymmetric_search_distance(query0, code_far);

    std::cout << "\n  Asymmetric distance to closest:  " << dist_close << "\n";
    std::cout << "  Asymmetric distance to farthest: " << dist_far << "\n";

    // For MIPS with negation: lower distance = more similar
    if (dist_close < dist_far) {
        std::cout << "  [PASS] Correctly distinguishes close vs far\n";
    } else {
        std::cout << "  [FAIL] Cannot distinguish close from far! R² will be ~0.\n";
    }
}

void test_code_diversity(Encoder& encoder, const float* vectors,
                         size_t num_vecs, size_t dim) {
    std::cout << "\n=== Test 3: Code Diversity ===\n";
    std::cout << "Check if different vectors produce different codes\n";

    // Encode first 100 vectors
    std::vector<Code> codes;
    for (size_t i = 0; i < std::min(num_vecs, size_t(100)); ++i) {
        codes.push_back(encoder.encode(vectors + i * dim));
    }

    // Check how many unique codes we have (using Hamming distance)
    size_t identical_pairs = 0;
    size_t total_pairs = 0;

    for (size_t i = 0; i < codes.size(); ++i) {
        for (size_t j = i + 1; j < codes.size(); ++j) {
            int ham = hamming_distance(codes[i], codes[j]);
            if (ham == 0) identical_pairs++;
            total_pairs++;
        }
    }

    std::cout << "  Total pairs: " << total_pairs << "\n";
    std::cout << "  Identical pairs: " << identical_pairs << "\n";

    // Check component distribution for code[0]
    std::cout << "\n  Code[0] components (first 8): ";
    for (size_t r = 0; r < 8; ++r) {
        std::cout << (int)codes[0].components[r] << " ";
    }
    std::cout << "\n  Code[1] components (first 8): ";
    for (size_t r = 0; r < 8; ++r) {
        std::cout << (int)codes[1].components[r] << " ";
    }
    std::cout << "\n";

    // Check if all codes are identical (Bug #4: RNG reset)
    bool all_same = true;
    for (size_t i = 1; i < codes.size(); ++i) {
        if (hamming_distance(codes[0], codes[i]) > 0) {
            all_same = false;
            break;
        }
    }

    if (all_same) {
        std::cout << "  [FAIL] All codes are IDENTICAL! RNG is probably being reset.\n";
    } else if (identical_pairs > total_pairs / 100) {
        std::cout << "  [WARN] Too many identical pairs - encoding too coarse.\n";
    } else {
        std::cout << "  [PASS] Codes show diversity.\n";
    }
}

void test_rotation_magnitudes(Encoder& encoder, const float* vec, size_t dim) {
    std::cout << "\n=== Test 4: FHT Scaling Check ===\n";
    std::cout << "Check if rotated vector magnitudes are reasonable\n";

    Query query = encoder.encode_query(vec);

    std::cout << "  Rotated vector stats per rotation:\n";
    std::cout << "  Rot | MaxAbs  | Mean   | Query[argmax]\n";
    std::cout << "  ----|---------|--------|-------------\n";

    for (size_t r = 0; r < std::min(K, size_t(8)); ++r) {
        const auto& rv = query.rotated_vecs[r];
        float max_abs = 0.0f, sum = 0.0f;
        for (size_t i = 0; i < rv.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(rv[i]));
            sum += rv[i];
        }
        float mean = sum / rv.size();

        // Get the value at the argmax position
        size_t argmax_idx = query.original_indices[r];
        float argmax_val = rv[argmax_idx];

        std::cout << "  " << std::setw(3) << r << " | "
                  << std::setw(7) << std::fixed << std::setprecision(4) << max_abs << " | "
                  << std::setw(6) << mean << " | "
                  << std::setw(7) << argmax_val << "\n";
    }

    // Check for overflow/underflow
    float total_magnitude = 0.0f;
    for (size_t r = 0; r < K; ++r) {
        total_magnitude += query.magnitudes[r];
    }

    std::cout << "\n  Total magnitude sum: " << total_magnitude << "\n";

    if (total_magnitude > 1e6) {
        std::cout << "  [FAIL] Magnitudes exploding! FHT needs 1/sqrt(D) normalization.\n";
    } else if (total_magnitude < 1e-3) {
        std::cout << "  [FAIL] Magnitudes vanishing! Precision loss in FHT.\n";
    } else {
        std::cout << "  [PASS] Magnitude range looks reasonable.\n";
    }
}

void test_soa_transposition() {
    std::cout << "\n=== Test 5: SoA Transposition Verification ===\n";
    std::cout << "Check if NeighborBlock.set_neighbor_code matches get_neighbor_code\n";

    using Block = NeighborBlock<uint8_t, K>;
    Block block;

    // Create test codes with known patterns
    std::vector<Code> test_codes(8);
    for (size_t n = 0; n < 8; ++n) {
        for (size_t k = 0; k < K; ++k) {
            // Unique pattern: component[k] = n * 10 + k
            test_codes[n].components[k] = static_cast<uint8_t>((n * 10 + k) % 256);
        }
    }

    // Store using set_neighbor_code
    for (size_t n = 0; n < 8; ++n) {
        block.set_neighbor_code(n, test_codes[n]);
    }
    block.count = 8;

    // Retrieve using get_neighbor_code_copy and verify
    bool all_match = true;
    for (size_t n = 0; n < 8; ++n) {
        Code retrieved = block.get_neighbor_code_copy(n);
        for (size_t k = 0; k < K; ++k) {
            if (retrieved.components[k] != test_codes[n].components[k]) {
                std::cout << "  [FAIL] Mismatch at n=" << n << " k=" << k
                          << ": expected " << (int)test_codes[n].components[k]
                          << " got " << (int)retrieved.components[k] << "\n";
                all_match = false;
            }
        }
    }

    if (all_match) {
        std::cout << "  [PASS] set_neighbor_code <-> get_neighbor_code_copy matches\n";
    }

    // Now check if codes_transposed has the expected layout
    std::cout << "\n  Checking codes_transposed[k][n] layout:\n";
    std::cout << "  Expected: codes_transposed[k][n] = n*10 + k\n";
    bool layout_correct = true;
    for (size_t k = 0; k < 4; ++k) {
        std::cout << "  k=" << k << ": ";
        for (size_t n = 0; n < 8; ++n) {
            uint8_t expected = static_cast<uint8_t>((n * 10 + k) % 256);
            uint8_t actual = block.codes_transposed[k][n];
            std::cout << (int)actual << " ";
            if (actual != expected) {
                layout_correct = false;
            }
        }
        std::cout << "\n";
    }

    if (layout_correct) {
        std::cout << "  [PASS] SoA transposed layout is correct\n";
    } else {
        std::cout << "  [FAIL] SoA transposed layout is WRONG!\n";
    }
}

void test_correlation(Encoder& encoder, const float* vectors, size_t num_vecs, size_t dim) {
    std::cout << "\n=== Test 6: R² Correlation Analysis ===\n";
    std::cout << "Sample 1000 random pairs and compute correlation\n";

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, std::min(num_vecs, size_t(10000)) - 1);

    std::vector<float> X, Y;  // X = asymm dist, Y = true dist

    for (int i = 0; i < 1000; ++i) {
        size_t u = dist(rng);
        size_t v = dist(rng);
        if (u == v) continue;

        const float* vec_u = vectors + u * dim;
        const float* vec_v = vectors + v * dim;

        // True distance (negative dot product)
        float true_dist = -true_similarity(vec_u, vec_v, dim);

        // Asymmetric distance
        Query query_u = encoder.encode_query(vec_u);
        Code code_v = encoder.encode(vec_v);
        float asymm_dist = asymmetric_search_distance(query_u, code_v);

        X.push_back(asymm_dist);
        Y.push_back(true_dist);
    }

    // Compute means
    float mean_x = 0, mean_y = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        mean_x += X[i];
        mean_y += Y[i];
    }
    mean_x /= X.size();
    mean_y /= Y.size();

    // Compute variances and covariance
    float var_x = 0, var_y = 0, cov_xy = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        float dx = X[i] - mean_x;
        float dy = Y[i] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }

    float r = cov_xy / std::sqrt(var_x * var_y);
    float r_squared = r * r;

    std::cout << "  Sample size: " << X.size() << "\n";
    std::cout << "  Mean(X): " << mean_x << ", Mean(Y): " << mean_y << "\n";
    std::cout << "  Var(X):  " << var_x / X.size() << ", Var(Y): " << var_y / X.size() << "\n";
    std::cout << "  Correlation r: " << r << "\n";
    std::cout << "  R-squared: " << r_squared << "\n";

    if (r_squared > 0.5) {
        std::cout << "  [PASS] R² > 0.5 - encoding is meaningful\n";
    } else if (r_squared > 0.2) {
        std::cout << "  [WARN] R² in 0.2-0.5 range - encoding has issues\n";
    } else {
        std::cout << "  [FAIL] R² < 0.2 - encoding is producing noise!\n";
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "CP Estimator Debug Suite\n";
    std::cout << "===========================================\n";
    std::cout << "K (rotations): " << K << "\n";

    // Load a small subset of SIFT
    std::cout << "\nLoading SIFT-1M (first 10K vectors)...\n";
    Dataset dataset = load_sift1m_normalized("../evaluation/data/sift");

    size_t test_size = std::min(dataset.num_base, size_t(10000));
    size_t dim = dataset.dim;

    std::cout << "Using " << test_size << " vectors, dim=" << dim << "\n";

    // Create encoder
    Encoder encoder(dim, 42);
    std::cout << "Encoder created: dim=" << encoder.dim()
              << ", padded_dim=" << encoder.padded_dim() << "\n";

    // Run tests
    test_identity(encoder, dataset.base_vectors.data(), dim);
    test_neighbor_discrimination(encoder, dataset.base_vectors.data(), test_size, dim);
    test_code_diversity(encoder, dataset.base_vectors.data(), test_size, dim);
    test_rotation_magnitudes(encoder, dataset.base_vectors.data(), dim);
    test_soa_transposition();
    test_correlation(encoder, dataset.base_vectors.data(), test_size, dim);

    std::cout << "\n===========================================\n";
    std::cout << "Debug Complete\n";
    std::cout << "===========================================\n";

    return 0;
}
