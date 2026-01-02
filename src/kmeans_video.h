#pragma once
#include <vector>
#include <string>

// K-Means Settings
#define K_CLUSTERS 16
#define APP_MAX_ITER 10  // Fewer iterations for video to hit target FPS (temporal coherence)

struct Pixel {
    float r, g, b;
};

// Function Declarations for Video Processing
// Returns execution time in milliseconds for one frame
float run_gpu_naive_frame(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids);
float run_gpu_shared_frame(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids, int blockSize);

// Helper for CUDA error checking
#include <iostream>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) {     cudaError_t err = call;     if (err != cudaSuccess) {         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;         exit(1); } }