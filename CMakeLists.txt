cmake_minimum_required(VERSION 3.24)

project(qwen36_2080ti LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(QWEN36_BUILD_TESTS "Build C++/CUDA tests" ON)

add_library(qwen36_core INTERFACE)
target_include_directories(qwen36_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_language(CUDA)

if(QWEN36_BUILD_TESTS)
    enable_testing()
endif()
