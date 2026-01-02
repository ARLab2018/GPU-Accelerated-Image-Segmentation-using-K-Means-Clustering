cmake_minimum_required(VERSION 3.18)
project(CudaKMeans LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target Ampere (86)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/vendor)

# --- Main Benchmark Executable ---
set(SOURCES_IMG
    src/main.cu
    src/cpu_kmeans.cpp
    src/opencv_kmeans.cpp
    src/gpu_naive.cu
    src/gpu_shared.cu
)
add_executable(CudaKMeans ${SOURCES_IMG})
target_link_libraries(CudaKMeans PRIVATE CUDA::cudart ${OpenCV_LIBS} stdc++fs)

# --- OpenCV GPU Executable ---
add_executable(OpenCVGPU src/main_opencv_gpu.cpp)
target_link_libraries(OpenCVGPU PRIVATE ${OpenCV_LIBS} stdc++fs)

# --- Video Naive Benchmark ---
add_executable(VideoNaive 
    src/main_video_naive.cpp
    src/gpu_naive_video.cu
)
# Fixed typo here: removed asterisks
target_link_libraries(VideoNaive PRIVATE CUDA::cudart ${OpenCV_LIBS} stdc++fs)

# --- Video Shared (Optimized) Benchmark ---
add_executable(VideoShared 
    src/main_video_shared.cpp
    src/gpu_shared_video.cu
)
# Fixed typo here: removed asterisks
target_link_libraries(VideoShared PRIVATE CUDA::cudart ${OpenCV_LIBS} stdc++fs)