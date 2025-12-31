#pragma once

/**
 * Debug Infrastructure for CP-HNSW
 *
 * Provides compile-time controllable debug output macros.
 *
 * Debug Levels:
 *   0 - No debug output (production)
 *   1 - Basic progress reporting
 *   2 - Detailed diagnostics
 *
 * Control via:
 *   -DCPHNSW_DEBUG_LEVEL=0  (disable all debug output)
 *   -DCPHNSW_DEBUG_LEVEL=1  (basic progress)
 *   -DCPHNSW_DEBUG_LEVEL=2  (detailed)
 *
 * Default:
 *   - Release builds (NDEBUG defined): Level 0
 *   - Debug builds: Level 1
 */

#include <iostream>

// Determine default debug level based on build type
#ifndef CPHNSW_DEBUG_LEVEL
    #ifdef NDEBUG
        #define CPHNSW_DEBUG_LEVEL 0
    #else
        #define CPHNSW_DEBUG_LEVEL 1
    #endif
#endif

// ============================================================================
// Debug Output Macros
// ============================================================================

#if CPHNSW_DEBUG_LEVEL >= 1

/**
 * Basic debug message (Level 1+).
 */
#define CPHNSW_DEBUG(msg) \
    std::cerr << "[CP-HNSW] " << msg << "\n"

/**
 * Progress reporting for batch operations (Level 1+).
 * Reports every `interval` items.
 */
#define CPHNSW_DEBUG_PROGRESS(done, total, interval) \
    do { \
        if ((done) % (interval) == 0) { \
            std::cerr << "[CP-HNSW] Progress: " << (done) << "/" << (total) << "\n"; \
        } \
    } while (0)

/**
 * Phase marker for multi-phase operations (Level 1+).
 */
#define CPHNSW_DEBUG_PHASE(phase_num, description) \
    std::cerr << "[CP-HNSW] Phase " << (phase_num) << ": " << (description) << "\n"

#else

#define CPHNSW_DEBUG(msg) ((void)0)
#define CPHNSW_DEBUG_PROGRESS(done, total, interval) ((void)(done), (void)(total), (void)(interval))
#define CPHNSW_DEBUG_PHASE(phase_num, description) ((void)(phase_num))

#endif

#if CPHNSW_DEBUG_LEVEL >= 2

/**
 * Detailed debug message (Level 2 only).
 */
#define CPHNSW_DEBUG_DETAIL(msg) \
    std::cerr << "[CP-HNSW:DETAIL] " << msg << "\n"

/**
 * Timing debug for performance analysis (Level 2 only).
 */
#define CPHNSW_DEBUG_TIMING(operation, elapsed_ms) \
    std::cerr << "[CP-HNSW:TIMING] " << (operation) << ": " << (elapsed_ms) << " ms\n"

#else

#define CPHNSW_DEBUG_DETAIL(msg) ((void)0)
#define CPHNSW_DEBUG_TIMING(operation, elapsed_ms) ((void)0)

#endif

// ============================================================================
// Compile-time Feature Flags (for optional functionality)
// ============================================================================

// CPHNSW_ENABLE_CONNECTIVITY_REPAIR - Enable repair_connectivity() method
// CPHNSW_DEBUG_LEVEL - Control debug output verbosity
