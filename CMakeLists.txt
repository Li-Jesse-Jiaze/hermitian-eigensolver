# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(NLA_final VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_BUILD_TYPE Release)

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add the include directory
include_directories(${ARMADILLO_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cpp
)

# Define the executable target
add_executable(main ${SOURCES})

# Link Armadillo library
target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# Find Doxygen
find_package(Doxygen REQUIRED)

# Set Doxygen input and output paths
set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include) # Directory with source code
set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs) # Documentation output directory
set(DOXYGEN_CONFIG_FILE ${CMAKE_BINARY_DIR}/Doxyfile)

# Configure Doxyfile template
set(DOXYGEN_IN ${PROJECT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${DOXYGEN_CONFIG_FILE})
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

# Add a custom target for generating documentation
add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)