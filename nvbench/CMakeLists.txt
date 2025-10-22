cmake_minimum_required(VERSION 3.30)
project(nvbench_vector_add LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Option A: use FetchContent to pull NVBench
include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(nvbench)

add_executable(vector_add main.cu)
# NVBench provides the nvbench_main target which supplies main() & flags parsing
target_link_libraries(vector_add PRIVATE nvbench_main)
# (Optional) set your GPU architectures explicitly if needed
# set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES "53;60;70;75;80;86;89")
