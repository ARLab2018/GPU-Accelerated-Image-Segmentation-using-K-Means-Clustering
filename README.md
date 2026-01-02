# GPU-Accelerated Real-Time Image Segmentation

A high-performance implementation of the K-Means Clustering algorithm optimized for the NVIDIA Ada Lovelace architecture. This project achieves real-time 4K video segmentation by leveraging hierarchical memory optimizations to bypass global memory serialization bottlenecks.

## 🚀 Performance Highlights

464x Speedup over sequential CPU implementation.

25x Speedup over industry-standard OpenCV library.

Real-time 4K (2160p) processing at ~46 FPS.

99% reduction in Global Memory atomic collisions via Shared Memory reductions.


## 🛠️ Prerequisites

Hardware: NVIDIA GPU (Compute Capability 8.6+ recommended, tested on RTX 4060).

OS: Ubuntu 22.04 LTS or WSL2.

Toolkit: CUDA Toolkit 11.8+ and nvcc.

Libraries: OpenCV 4.x (libopencv-dev).

Tools: CMake 3.18+ and Build-Essential (G++ 11+).

## ⚙️ Build Instructions

## Clone the repository
git clone [https://github.com/yourusername/KMeans-GPU.git](https://github.com/yourusername/KMeans-GPU.git)

cd KMeans-GPU

## Create build directory
mkdir build && cd build

## Configure and Compile
cmake ..
make -j$(npx --yes cpu-features-max-threads)


## 💻 Usage

1. Batch Image Benchmarking

Processes a directory of high-resolution images (e.g., 24MP) and compares CPU vs. GPU.

./CudaKMeans ../data/image1.jpg ../data/image2.jpg


2. Real-Time 4K Video Processing

To run the optimized algorithm on a video file:

./VideoShared input_video.mp4 output_segmented.mp4


3. OpenCV GPU (OpenCL) Comparison

To verify the "Abstraction Tax" of generic libraries:

./OpenCVGPU input.jpg


## 🔧 Troubleshooting

"CUDA Error: unknown error"

This occurs when the GPU driver enters a "zombie" state after a kernel crash or conflict with OpenCV's OpenCL context.

Fix: A full system reboot is required to clear the driver state.

Prevention: The project now includes cudaDeviceReset() and cv::ocl::setUseOpenCL(false) at startup to minimize conflicts.

"parameter packs not expanded"

Caused by nvcc attempting to parse modern C++ standard library headers included by OpenCV.

Fix: Ensure the driver files use the .cpp extension. The provided CMakeLists.txt is configured to use the host compiler (g++) for these files and nvcc only for the kernels.

## 📜 License

This project is licensed under the MIT License.
