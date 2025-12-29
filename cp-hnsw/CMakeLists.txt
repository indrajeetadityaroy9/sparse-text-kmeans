cmake_minimum_required(VERSION 3.16)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CPHNSW_USE_AVX512 "Enable AVX-512 support" OFF)
option(CPHNSW_BUILD_TESTS "Build unit tests" ON)
option(CPHNSW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CPHNSW_BUILD_EVAL "Build evaluation framework" ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -O3)

    # SIMD flags
    if(CPHNSW_USE_AVX512)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl)
        add_compile_definitions(CPHNSW_HAS_AVX512=1)
    else()
        # Check for AVX2 support
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2)
        endif()
    endif()

    # Enable native optimizations on non-Apple platforms
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()

elseif(MSVC)
    add_compile_options(/W4 /O2)
    if(CPHNSW_USE_AVX512)
        add_compile_options(/arch:AVX512)
    else()
        add_compile_options(/arch:AVX2)
    endif()
endif()

# Header-only library target
add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Evaluation executable
if(CPHNSW_BUILD_EVAL)
    add_executable(eval_cphnsw
        evaluation/eval_main.cpp
    )
    target_link_libraries(eval_cphnsw PRIVATE cphnsw)
    target_include_directories(eval_cphnsw PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # Comprehensive evaluation
    add_executable(eval_comprehensive
        evaluation/eval_comprehensive.cpp
    )
    target_link_libraries(eval_comprehensive PRIVATE cphnsw)
    target_include_directories(eval_comprehensive PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # SIFT-1M evaluation (requires OpenMP for ground truth computation)
    find_package(OpenMP)
    add_executable(eval_sift
        evaluation/eval_sift.cpp
    )
    target_include_directories(eval_sift PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(eval_sift PRIVATE cphnsw OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(eval_sift PRIVATE cphnsw)
        message(WARNING "OpenMP not found - SIFT ground truth computation will be slow!")
    endif()
endif()

# Unit tests
if(CPHNSW_BUILD_TESTS)
    # Try to find GTest
    find_package(GTest QUIET)

    if(GTest_FOUND)
        enable_testing()

        add_executable(cphnsw_tests
            tests/unit/test_hadamard.cpp
        )
        target_link_libraries(cphnsw_tests PRIVATE cphnsw GTest::gtest_main)

        include(GoogleTest)
        gtest_discover_tests(cphnsw_tests)
    else()
        message(STATUS "GTest not found, skipping tests")
    endif()
endif()

# Benchmarks
if(CPHNSW_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)

    if(benchmark_FOUND)
        add_executable(bench_fht benchmarks/bench_fht.cpp)
        target_link_libraries(bench_fht PRIVATE cphnsw benchmark::benchmark)

        add_executable(bench_hamming benchmarks/bench_hamming.cpp)
        target_link_libraries(bench_hamming PRIVATE cphnsw benchmark::benchmark)
    else()
        message(STATUS "Google Benchmark not found, skipping benchmarks")
    endif()
endif()

# Install
install(TARGETS cphnsw EXPORT cphnsw-targets)
install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "CP-HNSW Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX-512: ${CPHNSW_USE_AVX512}")
message(STATUS "  Build Tests: ${CPHNSW_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${CPHNSW_BUILD_BENCHMARKS}")
message(STATUS "  Build Evaluation: ${CPHNSW_BUILD_EVAL}")
message(STATUS "")
