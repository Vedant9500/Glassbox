cmake_minimum_required(VERSION 3.14)
project(glassbox_sr_cpp)

# §3.389: pin the language standard explicitly (mirrors setup.py).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)

# Add our C++ module (same sources/flags intent as setup.py)
pybind11_add_module(_core core.cpp)

# Optimize for speed + OpenMP (align with setup.py / P6-004)
if(MSVC)
  target_compile_options(_core PRIVATE /O2 /openmp)
else()
  target_compile_options(_core PRIVATE -O3 -march=native -fopenmp)
  target_link_options(_core PRIVATE -fopenmp)
endif()
