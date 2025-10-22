cmake_minimum_required(VERSION 3.30)
project(nvbench_vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)           # NVBench headers use C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        main            # (or pin a tag)
)

# ---- Robust: populate + add_subdirectory manually
FetchContent_GetProperties(nvbench)
if(NOT nvbench_POPULATED)
  FetchContent_Populate(nvbench)
  add_subdirectory(${nvbench_SOURCE_DIR} ${nvbench_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(vector_add main.cu)

# Resolve NVBench target names (namespaced vs plain)
if (TARGET nvbench::nvbench_main)
  set(NVBENCH_MAIN nvbench::nvbench_main)
  set(NVBENCH_LIB  nvbench::nvbench)
elseif (TARGET nvbench_main)
  set(NVBENCH_MAIN nvbench_main)
  set(NVBENCH_LIB  nvbench)
else()
  message(FATAL_ERROR "NVBench targets still not found; looked for nvbench::nvbench_main/nvbench_main.")
endif()

target_link_libraries(vector_add PRIVATE ${NVBENCH_MAIN} ${NVBENCH_LIB})

# Belt-and-suspenders include path (harmless if already propagated)
target_include_directories(vector_add PRIVATE ${nvbench_SOURCE_DIR}/include)

# (Optional) Arch — Nano=53, Xavier=72, Orin=87
# set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES "53")
