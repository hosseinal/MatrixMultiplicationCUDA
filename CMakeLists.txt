cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# set host compiler
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 86)

project(MatrixMultiplicationCUDA LANGUAGES CXX CUDA)

# Set CUDA specific flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#add_compile_definitions(CHECK_CORRECTNESS)

# Add executable
add_executable(matrix_multiplication main.cu
        Matrix.cuh
        Matrix.cu
        CSRMatrix.cu
        CSRMatrix.cuh
        BCSRMatrix.cu
        BCSRMatrix.cuh
        miscutil.cpp
        miscutil.h
        HCSRMatrix.cpp
        HCSRMatrix.h
)

# Set CUDA specific properties
set_target_properties(matrix_multiplication PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(matrix_multiplication CUDA::cusparse stdc++ -lcublas
        -lcurand)

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

# nvbench target (integration)
include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(nvbench)

add_executable(matrix_multiplication_nvbench
    nvbench_main.cu
    main.cu
    Matrix.cuh
    Matrix.cu
    CSRMatrix.cu
    CSRMatrix.cuh
    BCSRMatrix.cu
    BCSRMatrix.cuh
    miscutil.cpp
    miscutil.h
    HCSRMatrix.cpp
    HCSRMatrix.h
)
set_target_properties(matrix_multiplication_nvbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_multiplication_nvbench PROPERTIES CXX_STANDARD 17)
target_compile_definitions(matrix_multiplication_nvbench PRIVATE USE_NVBENCH)
# If nvbench headers are not in a standard location, set NVBENCH_INCLUDE_DIR when running cmake
if(DEFINED NVBENCH_INCLUDE_DIR)
    target_include_directories(matrix_multiplication_nvbench PRIVATE ${NVBENCH_INCLUDE_DIR})
endif()
target_link_libraries(matrix_multiplication_nvbench PRIVATE nvbench_main CUDA::cusparse stdc++ -lcublas -lcurand)