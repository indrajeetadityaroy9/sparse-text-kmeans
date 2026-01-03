#pragma once

#include <cstddef>
#include <cstdint>

namespace cphnsw {

// ============================================================================
// Index Configuration
// ============================================================================

/**
 * IndexParams: Configuration for index construction.
 *
 * These parameters affect memory usage and search quality.
 */
struct IndexParams {
    /// Vector dimension (required)
    size_t dim = 0;

    /// Maximum neighbors per node (M parameter)
    /// Higher M = better recall but more memory
    /// Typical range: 16-64
    size_t M = 32;

    /// Search width during construction (ef_construction)
    /// Higher ef = better graph quality but slower build
    /// Typical range: 100-500
    size_t ef_construction = 200;

    /// Random seed for reproducibility
    uint64_t seed = 42;

    /// Initial capacity (number of vectors to pre-allocate)
    size_t initial_capacity = 1024;

    // Builder pattern
    IndexParams& set_dim(size_t d) { dim = d; return *this; }
    IndexParams& set_M(size_t m) { M = m; return *this; }
    IndexParams& set_ef_construction(size_t ef) { ef_construction = ef; return *this; }
    IndexParams& set_seed(uint64_t s) { seed = s; return *this; }
    IndexParams& set_capacity(size_t c) { initial_capacity = c; return *this; }
};

/**
 * BuildParams: Configuration for batch construction.
 */
struct BuildParams {
    /// Number of threads for parallel construction (0 = auto-detect)
    size_t num_threads = 0;

    /// Use GPU for k-NN graph construction (if CUDA available)
    bool use_gpu = false;

    /// GPU device ID (for multi-GPU systems)
    int gpu_device = 0;

    /// Tile size for GPU k-NN (larger = more memory, potentially faster)
    size_t gpu_tile_size = 16384;

    /// Enable verbose progress output
    bool verbose = false;

    // Builder pattern
    BuildParams& set_threads(size_t n) { num_threads = n; return *this; }
    BuildParams& set_gpu(bool use, int device = 0) { use_gpu = use; gpu_device = device; return *this; }
    BuildParams& set_verbose(bool v) { verbose = v; return *this; }
};

// ============================================================================
// Search Configuration
// ============================================================================

/**
 * SearchParams: Configuration for k-NN search.
 */
struct SearchParams {
    /// Number of nearest neighbors to return
    size_t k = 10;

    /// Search width (ef)
    /// Higher ef = better recall but slower search
    /// Must be >= k
    size_t ef = 100;

    /// Number of entry points for search (for multi-start search)
    size_t num_entry_points = 1;

    /// Enable reranking with exact distances
    bool rerank = true;

    /// Number of candidates to rerank (if rerank enabled)
    /// Higher rerank_k = better recall but slower
    size_t rerank_k = 200;

    // Builder pattern
    SearchParams& set_k(size_t num) { k = num; return *this; }
    SearchParams& set_ef(size_t e) { ef = e; return *this; }
    SearchParams& set_rerank(bool r, size_t rk = 200) { rerank = r; rerank_k = rk; return *this; }
};

// ============================================================================
// Mode Configuration
// ============================================================================

/**
 * CodeMode: Selects the quantization mode.
 */
enum class CodeMode {
    /// Phase 1: Pure RaBitQ (K bits, no residual)
    /// Fastest, lowest memory, moderate recall
    RaBitQ,

    /// Phase 2: With residual quantization (K + R bits)
    /// Slower, more memory, better recall
    Residual,

    /// Phase 3: Learned rotations (future)
    /// Best recall, requires training
    Learned
};

/**
 * ModeConfig: Compile-time configuration for code mode.
 *
 * Use this with the template-based API for compile-time optimization.
 */
template <size_t K_, size_t R_ = 0, int Shift_ = 2>
struct ModeConfig {
    static constexpr size_t K = K_;
    static constexpr size_t R = R_;
    static constexpr int Shift = Shift_;
    static constexpr bool has_residual = (R > 0);

    static constexpr CodeMode mode() {
        return has_residual ? CodeMode::Residual : CodeMode::RaBitQ;
    }
};

// Common configurations
using Mode32 = ModeConfig<32, 0>;      // 32-bit Phase 1
using Mode64 = ModeConfig<64, 0>;      // 64-bit Phase 1
using Mode64_32 = ModeConfig<64, 32>;  // 64+32-bit Phase 2
using Mode32_16 = ModeConfig<32, 16>;  // 32+16-bit Phase 2

}  // namespace cphnsw
