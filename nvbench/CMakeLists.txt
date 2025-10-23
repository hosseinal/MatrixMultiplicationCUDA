cmake_minimum_required(VERSION 3.22)

project(NVBench CUDA)

# Use CPM to fetch NVBench's main branch.
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_executable(main_bench main.cu)
target_link_libraries(main_bench nvbench::main)

