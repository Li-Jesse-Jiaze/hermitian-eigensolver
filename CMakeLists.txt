# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(NLA_final VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the include directory
include_directories(${ARMADILLO_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cpp
)

# Define the executable target
add_executable(main ${SOURCES})

# Link Armadillo library
target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Eigen3::Eigen)
