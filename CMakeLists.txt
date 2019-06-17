cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(python_from_cpp)

find_package(OpenCV REQUIRED)

add_subdirectory(pybind11)

add_executable(example_001 src/example_001.cpp)
target_link_libraries(example_001 PRIVATE pybind11::embed)
add_executable(example_002 src/example_002.cpp)
target_link_libraries(example_002 PRIVATE pybind11::embed)
add_executable(example_003 src/example_003.cpp)
target_link_libraries(example_003 PRIVATE pybind11::embed)
add_executable(example_004 src/example_004.cpp)
target_link_libraries(example_004 PRIVATE pybind11::embed ${OpenCV_LIBS})
