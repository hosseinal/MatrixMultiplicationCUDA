cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# set host compiler
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures (override by passing -DCMAKE_CUDA_ARCHITECTURES="..." to cmake)
# Note: some older CUDA toolkits / nvcc do not recognize very recent SM targets
# (e.g. compute_89). Set a conservative default here and allow users to override it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

project(MatrixMultiplicationCUDA LANGUAGES CXX CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#add_compile_definitions(CHECK_CORRECTNESS)

# Add executable
# add_executable(matrix_multiplication main.cu
#         Matrix.cuh
#         Matrix.cu
#         CSRMatrix.cu
#         CSRMatrix.cuh
#         BCSRMatrix.cu
#         BCSRMatrix.cuh
#         miscutil.cpp
#         miscutil.h
#         HCSRMatrix.cpp
#         HCSRMatrix.h
#         src/cuda_kernels.cuh
#         src/cuda_kernels.cu
# )

# Set CUDA specific properties
# set_target_properties(matrix_multiplication PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

find_package(CUDAToolkit REQUIRED)

# target_link_libraries(matrix_multiplication CUDA::cusparse stdc++ -lcublas
#         -lcurand)

# Optional: build a small matrix generator test (CPU-only)
option(BUILD_MG_TEST "Build matrix generator test" OFF)
if(BUILD_MG_TEST)
    add_executable(matrix_generator_test matrix_generator_test.cpp matrix_generator.cpp)
    target_include_directories(matrix_generator_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(matrix_generator_test PROPERTIES CXX_STANDARD 17)
endif()

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif() 

# Use CPM to fetch NVBench's main branch (matches demo CMakeLists)
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_executable(matrix_multiplication_nvbench
    nvbench_main.cu
    cute_test/gemm_cute_tensor.cu
    cute_test/gemm_cute_simt.cu
    cute_test/gemm_cute_tensor.cuh
    cute_test/gemm_cute_simt.cuh
    Matrix.cuh
    Matrix.cu
    formats/CSRMatrix.cu
    formats/CSRMatrix.cuh
    formats/BCSRMatrix.cu
    formats/BCSRMatrix.cuh
    miscutil.cpp
    miscutil.h
    formats/HCSRMatrix.cpp
    formats/HCSRMatrix.h
    src/cuda_kernels.cuh
    src/cuda_kernels.cu
)
set_target_properties(matrix_multiplication_nvbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_multiplication_nvbench PROPERTIES CXX_STANDARD 17)
target_compile_definitions(matrix_multiplication_nvbench PRIVATE USE_NVBENCH)
# If nvbench headers are not in a standard location, set NVBENCH_INCLUDE_DIR when running cmake
if(DEFINED NVBENCH_INCLUDE_DIR)
    target_include_directories(matrix_multiplication_nvbench PRIVATE ${NVBENCH_INCLUDE_DIR})
endif()

# Expose local CUTE headers to the target so callers can `#include "cute/..."`
target_include_directories(matrix_multiplication_nvbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cute)
target_link_libraries(matrix_multiplication_nvbench PRIVATE nvbench::main CUDA::cudart CUDA::cusparse stdc++ -lcublas -lcurand)