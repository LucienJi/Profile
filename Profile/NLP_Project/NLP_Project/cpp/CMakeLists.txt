cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(INF442 VERSION 0.1.0)

# Include dir
include_directories(/usr/local/include)
include_directories("eigen3")
# Src
AUX_SOURCE_DIRECTORY(src SRC_FILES)

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include/mylib_bits")

# Source files
set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/svm.hpp
    ${PROJECT_INCLUDE_DIR}/kernel.hpp
    ${PROJECT_INCLUDE_DIR}/logisticregression.hpp
        ${PROJECT_INCLUDE_DIR}/ConfusionMatrix.hpp


        ${PROJECT_SOURCE_DIR}/svm.cpp
    ${PROJECT_SOURCE_DIR}/kernel.cpp
    ${PROJECT_SOURCE_DIR}/logisticregression.cpp
        ${PROJECT_SOURCE_DIR}/ConfusionMatrix.cpp

        )

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(mylib SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(mylib PRIVATE include/mylib_bits)

# Install
install(TARGETS mylib DESTINATION lib)

# Install the headers
install(FILES include/mylib DESTINATION include)

# Create base directory
install(DIRECTORY include/mylib_bits DESTINATION include)