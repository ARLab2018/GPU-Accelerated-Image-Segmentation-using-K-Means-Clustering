#pragma once
#include <vector>
#include <string>

#define K_CLUSTERS 16
#define APP_MAX_ITER 50

struct Pixel {
    float r, g, b;
};

// All functions now return elapsed time in milliseconds (float)
float run_cpu_kmeans(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids);
float run_gpu_naive(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids);
float run_gpu_shared(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids, int blockSize);
float run_opencv_kmeans(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids);

void init_opencv();
void save_image(const std::string& filename, int width, int height, const std::vector<int>& labels, const std::vector<Pixel>& centroids);