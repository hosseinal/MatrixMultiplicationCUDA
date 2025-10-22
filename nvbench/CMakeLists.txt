cmake_minimum_required(VERSION 3.18)

project(nvbench_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Allow user to specify where nvbench headers are installed
option(USE_NVBENCH "Enable building nvbench benchmark executable" ON)

if(NOT USE_NVBENCH)
	message(STATUS "nvbench disabled (USE_NVBENCH=OFF). Skipping nvbench target.")
	return()
endif()

find_package(CUDAToolkit REQUIRED)

set(NVBENCH_SRCS main.cu)

add_executable(nvbench_main ${NVBENCH_SRCS})

# Enable separable compilation for CUDA files (if needed)
set_target_properties(nvbench_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# If nvbench headers are installed in a non-standard location, allow the user to pass NVBENCH_INCLUDE_DIR
if(DEFINED NVBENCH_INCLUDE_DIR)
	target_include_directories(nvbench_main PRIVATE ${NVBENCH_INCLUDE_DIR})
else()
	# try a few common locations silently (no failure)
	find_path(NVBENCH_INCLUDE_DIR_FOUND nvbench/nvbench.cuh PATHS /usr/include /usr/local/include /opt/local/include NO_DEFAULT_PATH)
	if(NVBENCH_INCLUDE_DIR_FOUND)
		target_include_directories(nvbench_main PRIVATE ${NVBENCH_INCLUDE_DIR_FOUND})
	endif()
endif()

# Link with CUDA toolkit libraries if needed (none required for pure header-only nvbench)
target_link_libraries(nvbench_main PRIVATE CUDA::cudart)

# Add useful compile definitions and flags
target_compile_definitions(nvbench_main PRIVATE NVBENCH_HOST_COMPILER_GCC)

message(STATUS "Added nvbench target 'nvbench_main'. Configure with -DNVBENCH_INCLUDE_DIR=/path/to/nvbench if headers are in a custom location.")
