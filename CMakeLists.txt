cmake_minimum_required(VERSION 3.18)
project(cp-hnsw VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(CPHNSW_USE_AVX512 "Enable AVX-512 support" ON)
option(CPHNSW_USE_OPENMP "Enable OpenMP parallelization" ON)
option(CPHNSW_USE_CUDA "Enable CUDA GPU acceleration" ON)
option(CPHNSW_BUILD_TESTS "Build unit tests" ON)
option(CPHNSW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CPHNSW_BUILD_EVAL "Build evaluation framework (requires porting to unified API)" OFF)

# CUDA setup (before other compiler flags)
if(CPHNSW_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # Find CUDA toolkit for cuBLAS
        find_package(CUDAToolkit REQUIRED)

        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        # Set CUDA architectures (H100 = sm_90, A100 = sm_80, V100 = sm_70)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
        endif()

        add_compile_definitions(CPHNSW_HAS_CUDA=1)
    else()
        message(WARNING "CUDA requested but not found, disabling GPU support")
        set(CPHNSW_USE_CUDA OFF)
    endif()
endif()

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

# OpenMP support
if(CPHNSW_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# Header-only library target
add_library(cphnsw INTERFACE)
target_include_directories(cphnsw INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add OpenMP to the interface library
if(CPHNSW_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(cphnsw INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(cphnsw INTERFACE CPHNSW_USE_OPENMP=1)
endif()

# CUDA library (compiled, not header-only)
if(CPHNSW_USE_CUDA)
    add_library(cphnsw_cuda STATIC
        src/cuda/gpu_encoder.cu
        src/cuda/gpu_knn_graph.cu
    )
    target_include_directories(cphnsw_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cphnsw_cuda PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_definitions(cphnsw_cuda PUBLIC CPHNSW_USE_CUDA=1)
    set_target_properties(cphnsw_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    # Add OpenMP to CUDA library for hybrid CPU+GPU
    if(CPHNSW_USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(cphnsw_cuda PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

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

    # SIFT-1M evaluation (OpenMP inherited from cphnsw library)
    add_executable(eval_sift
        evaluation/eval_sift.cpp
    )
    target_link_libraries(eval_sift PRIVATE cphnsw)
    target_include_directories(eval_sift PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # Tiered construction test
    add_executable(test_tiered
        evaluation/test_tiered.cpp
    )
    target_link_libraries(test_tiered PRIVATE cphnsw)
    target_include_directories(test_tiered PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # Correlation test
    add_executable(test_correlation
        evaluation/test_correlation.cpp
    )
    target_link_libraries(test_correlation PRIVATE cphnsw)
    target_include_directories(test_correlation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # Master evaluation protocol (all 6 experiments for PhD portfolio)
    add_executable(eval_master
        evaluation/eval_master.cpp
    )
    target_link_libraries(eval_master PRIVATE cphnsw)
    target_include_directories(eval_master PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # K-specific eval_master variants for Research Evaluation
    foreach(K_VAL 16 32 64)
        add_executable(eval_master_k${K_VAL}
            evaluation/eval_master.cpp
        )
        target_compile_definitions(eval_master_k${K_VAL} PRIVATE EVAL_K=${K_VAL})
        target_link_libraries(eval_master_k${K_VAL} PRIVATE cphnsw)
        target_include_directories(eval_master_k${K_VAL} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )
    endforeach()

    # QA Protocol test suite
    add_executable(qa_protocol
        evaluation/qa_protocol.cpp
    )
    target_link_libraries(qa_protocol PRIVATE cphnsw)
    target_include_directories(qa_protocol PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # CP Estimator Debug
    add_executable(debug_estimator
        evaluation/debug_estimator.cpp
    )
    target_link_libraries(debug_estimator PRIVATE cphnsw)
    target_include_directories(debug_estimator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
    )

    # GPU evaluation (SIFT-1M with CUDA acceleration)
    if(CPHNSW_USE_CUDA)
        add_executable(eval_sift_gpu
            evaluation/eval_sift_gpu.cpp
        )
        target_link_libraries(eval_sift_gpu PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(eval_sift_gpu PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )

        # GPU k-NN Build test
        add_executable(test_gpu_build
            evaluation/test_gpu_build.cpp
        )
        target_link_libraries(test_gpu_build PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(test_gpu_build PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )

        # GPU k-NN Debug
        add_executable(debug_gpu_knn
            evaluation/debug_gpu_knn.cpp
        )
        target_link_libraries(debug_gpu_knn PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(debug_gpu_knn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )

        # Code Storage Debug
        add_executable(debug_code_storage
            evaluation/debug_code_storage.cpp
        )
        target_link_libraries(debug_code_storage PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(debug_code_storage PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )

        # Recall Debug
        add_executable(debug_recall
            evaluation/debug_recall.cpp
        )
        target_link_libraries(debug_recall PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(debug_recall PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )

        # K Value Variants Test
        add_executable(test_k_variants
            evaluation/test_k_variants.cpp
        )
        target_link_libraries(test_k_variants PRIVATE cphnsw cphnsw_cuda)
        target_include_directories(test_k_variants PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/evaluation
        )
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

    # Standalone test executables (no GTest dependency)
    # Unified Distance Tests (Phase 1 RaBitQ + Phase 2 Residual)
    add_executable(test_unified_distance
        tests/unit/test_unified_distance.cpp
    )
    target_link_libraries(test_unified_distance PRIVATE cphnsw)
    target_include_directories(test_unified_distance PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
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
message(STATUS "  OpenMP: ${CPHNSW_USE_OPENMP}")
message(STATUS "  CUDA: ${CPHNSW_USE_CUDA}")
if(CPHNSW_USE_CUDA)
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build Tests: ${CPHNSW_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${CPHNSW_BUILD_BENCHMARKS}")
message(STATUS "  Build Evaluation: ${CPHNSW_BUILD_EVAL}")
message(STATUS "")
