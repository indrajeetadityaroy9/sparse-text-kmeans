#pragma once

/**
 * Common Evaluation Utilities
 *
 * Shared helper functions for CP-HNSW evaluation benchmarks.
 * Consolidates utilities previously duplicated across eval_*.cpp files.
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __unix__
#include <sys/resource.h>
#elif defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace cphnsw {
namespace eval {

// =============================================================================
// System Information
// =============================================================================

/**
 * Print system information (hardware threads, SIMD capabilities, OpenMP).
 */
inline void print_system_info() {
    std::cout << "=== System Information ===\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP: DISABLED\n";
#endif

    std::cout << "Compilation: ";
#ifdef __AVX512F__
    std::cout << "AVX-512 ";
#endif
#ifdef __AVX2__
    std::cout << "AVX2 ";
#endif
#ifdef __SSE4_2__
    std::cout << "SSE4.2 ";
#endif
    std::cout << "\n\n";
}

// =============================================================================
// Directory and Formatting Utilities
// =============================================================================

/**
 * Ensure a directory exists (creates if necessary).
 */
inline void ensure_directory(const std::string& dir) {
    std::string cmd = "mkdir -p " + dir;
    (void)system(cmd.c_str());
}

/**
 * Format a number with specified precision.
 */
inline std::string format_number(double val, int precision = 2) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << val;
    return ss.str();
}

/**
 * Format bytes as human-readable string (KB, MB, GB).
 */
inline std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    return format_number(size, 2) + " " + units[unit_idx];
}

// =============================================================================
// Memory Utilities
// =============================================================================

/**
 * Get current process RSS (Resident Set Size) in bytes.
 */
inline size_t get_rss_bytes() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss is in KB on Linux, bytes on macOS
#ifdef __APPLE__
    return static_cast<size_t>(usage.ru_maxrss);
#else
    return static_cast<size_t>(usage.ru_maxrss) * 1024;
#endif
#else
    return 0;  // Not supported
#endif
}

// =============================================================================
// Statistical Utilities
// =============================================================================

/**
 * Compute Pearson correlation coefficient between two vectors.
 *
 * @param x First vector
 * @param y Second vector (must be same size as x)
 * @return Correlation coefficient in [-1, 1], or 0 if invalid
 */
inline double compute_pearson_correlation(const std::vector<double>& x,
                                          const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;

    size_t n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double num = n * sum_xy - sum_x * sum_y;
    double den = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return (den > 1e-10) ? num / den : 0.0;
}

/**
 * Compute mean of a vector.
 */
inline double compute_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = 0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
}

/**
 * Compute standard deviation of a vector.
 */
inline double compute_stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mean = compute_mean(v);
    double sum_sq = 0;
    for (double x : v) {
        double diff = x - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<double>(v.size() - 1));
}

}  // namespace eval
}  // namespace cphnsw
