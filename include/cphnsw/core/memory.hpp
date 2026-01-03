#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>
#include <type_traits>

namespace cphnsw {

// ============================================================================
// Memory Constants
// ============================================================================

#ifndef CPHNSW_CACHE_LINE_SIZE_DEFINED
#define CPHNSW_CACHE_LINE_SIZE_DEFINED
constexpr size_t CACHE_LINE_SIZE = 64;
#endif
constexpr size_t SIMD_ALIGNMENT = 64;  // AVX-512 alignment

// ============================================================================
// Aligned Allocator
// ============================================================================

/**
 * AlignedAllocator: STL-compatible allocator for SIMD-aligned memory.
 *
 * Ensures allocations are aligned to Alignment bytes (default 64 for AVX-512).
 * Works with std::vector and other STL containers.
 *
 * @tparam T Element type
 * @tparam Alignment Alignment in bytes (must be power of 2)
 */
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;

        size_t bytes = n * sizeof(T);

#if defined(_MSC_VER)
        void* ptr = _aligned_malloc(bytes, Alignment);
#else
        void* ptr = std::aligned_alloc(Alignment, bytes);
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
#if defined(_MSC_VER)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
    }

    template <typename U, size_t A>
    bool operator==(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment == A;
    }

    template <typename U, size_t A>
    bool operator!=(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment != A;
    }
};

// ============================================================================
// Aligned Vector Type Alias
// ============================================================================

/**
 * AlignedVector: std::vector with SIMD-aligned memory.
 */
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

// ============================================================================
// Memory Traits
// ============================================================================

/**
 * MemoryTraits: Compile-time information about memory requirements.
 */
template <typename T>
struct MemoryTraits {
    static constexpr size_t alignment = alignof(T);
    static constexpr size_t size = sizeof(T);
    static constexpr bool is_trivially_copyable = std::is_trivially_copyable_v<T>;
    static constexpr bool is_cache_aligned = (alignment >= CACHE_LINE_SIZE);
    static constexpr bool is_simd_aligned = (alignment >= SIMD_ALIGNMENT);
};

// ============================================================================
// Prefetch Hints
// ============================================================================

/**
 * Prefetch data into cache (L1, highest locality).
 *
 * @param addr Memory address to prefetch
 */
inline void prefetch(const void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3);  // Read, L1 locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
    (void)addr;
#endif
}

/**
 * Prefetch for write access (L1, highest locality).
 */
inline void prefetch_write(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3);  // Write, L1 locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
    (void)addr;
#endif
}

/**
 * Prefetch with template-specified locality.
 * @tparam Locality 0=non-temporal, 1=L3, 2=L2, 3=L1
 */
template <int Locality = 3>
inline void prefetch_t(const void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, Locality);
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#else
    (void)addr;
#endif
}

// ============================================================================
// Cache Line Padding
// ============================================================================

/**
 * CacheLinePad: Padding to prevent false sharing.
 *
 * Place between hot fields that are accessed by different threads.
 */
struct alignas(CACHE_LINE_SIZE) CacheLinePad {
    char padding[CACHE_LINE_SIZE];
};

/**
 * Ensure a value occupies its own cache line to prevent false sharing.
 */
template <typename T>
struct alignas(CACHE_LINE_SIZE) CacheLineIsolated {
    T value;
    char padding[CACHE_LINE_SIZE - sizeof(T) % CACHE_LINE_SIZE];

    CacheLineIsolated() = default;
    explicit CacheLineIsolated(const T& v) : value(v) {}
    explicit CacheLineIsolated(T&& v) : value(std::move(v)) {}

    operator T&() { return value; }
    operator const T&() const { return value; }
};

// ============================================================================
// Unique Pointer with Custom Deleter for Aligned Memory
// ============================================================================

/**
 * AlignedDeleter: Custom deleter for aligned memory.
 */
struct AlignedDeleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) {
#if defined(_MSC_VER)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }
};

/**
 * AlignedUniquePtr: Unique pointer for aligned allocations.
 */
template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter>;

/**
 * Allocate aligned memory and return as unique_ptr.
 */
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
AlignedUniquePtr<T> make_aligned(size_t count = 1) {
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");

    size_t bytes = count * sizeof(T);

#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(bytes, Alignment);
#else
    void* ptr = std::aligned_alloc(Alignment, bytes);
#endif

    if (!ptr) {
        throw std::bad_alloc();
    }

    return AlignedUniquePtr<T>(static_cast<T*>(ptr));
}

}  // namespace cphnsw
