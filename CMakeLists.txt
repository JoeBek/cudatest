cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(line_detector LANGUAGES CXX CUDA)

# Enable debugging
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Set CUDA architecture (adjust to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


# Add executable
add_executable(line_detector main.cpp line_detector.cpp cuda.cu)

# Include directories
target_include_directories(line_detector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(line_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV libraries
target_link_libraries(line_detector 
                      ${OpenCV_LIBS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_nppi_LIBRARY}  # For image processing
                      ${CUDA_nppc_LIBRARY}  # Core NPP functionality
)
