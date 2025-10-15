cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

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

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif() 