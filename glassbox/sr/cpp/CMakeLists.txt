cmake_minimum_required(VERSION 3.14...)
project(glassbox_sr_cpp)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)

# Add our C++ module
pybind11_add_module(_core core.cpp)

# Optimize for speed
target_compile_options(_core PRIVATE -O3 -march=native)
