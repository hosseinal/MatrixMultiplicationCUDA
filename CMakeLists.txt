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

# Optional nvbench target (minimal integration)
option(USE_NVBENCH "Build nvbench harness (requires nvbench headers installed)" OFF)
if(USE_NVBENCH)
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
    target_compile_definitions(matrix_multiplication_nvbench PRIVATE USE_NVBENCH)
    # If nvbench headers are not in a standard location, set NVBENCH_INCLUDE_DIR when running cmake
    if(DEFINED NVBENCH_INCLUDE_DIR)
        target_include_directories(matrix_multiplication_nvbench PRIVATE ${NVBENCH_INCLUDE_DIR})
    endif()
    target_link_libraries(matrix_multiplication_nvbench CUDA::cusparse stdc++ -lcublas -lcurand)
endif()