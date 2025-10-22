cmake_minimum_required(VERSION 3.30)
project(nvbench_vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)          # <-- NVBench needs CUDA C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        main            # for stability, pin a release tag if you prefer
)
FetchContent_MakeAvailable(nvbench)

add_executable(vector_add main.cu)

# Resolve NVBench target names (namespaced vs. plain)
if (TARGET nvbench::nvbench_main)
  set(NVBENCH_MAIN nvbench::nvbench_main)
  set(NVBENCH_LIB  nvbench::nvbench)
elseif (TARGET nvbench_main)
  set(NVBENCH_MAIN nvbench_main)
  set(NVBENCH_LIB  nvbench)
else()
  message(FATAL_ERROR "NVBench targets not found after FetchContent.")
endif()

target_link_libraries(vector_add PRIVATE ${NVBENCH_MAIN} ${NVBENCH_LIB})

# Safety net: add include dir explicitly (harmless if already propagated)
# The *_SOURCE_DIR variable is defined by FetchContent.
target_include_directories(vector_add PRIVATE ${nvbench_SOURCE_DIR}/include)

# (Optional) set your GPU arch; Jetson Nano=53, Xavier=72, Orin=87
# set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES "native")
# or: set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES "53")
