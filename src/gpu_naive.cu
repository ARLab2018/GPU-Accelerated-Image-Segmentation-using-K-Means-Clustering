#include "kmeans.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void k_means_naive(const Pixel* pixels, int N, Pixel* centroids, int* labels, Pixel* new_sums, int* new_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Pixel p = pixels[idx];
    float min_dist = FLT_MAX;
    int best_k = 0;

    for (int k = 0; k < K_CLUSTERS; k++) {
        float r = p.r - centroids[k].r;
        float g = p.g - centroids[k].g;
        float b = p.b - centroids[k].b;
        float dist = r*r + g*g + b*b;
        
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }

    labels[idx] = best_k;

    atomicAdd(&new_sums[best_k].r, p.r);
    atomicAdd(&new_sums[best_k].g, p.g);
    atomicAdd(&new_sums[best_k].b, p.b);
    atomicAdd(&new_counts[best_k], 1);
}

float run_gpu_naive(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids) {
    int N = width * height;
    size_t pixel_size = N * sizeof(Pixel);
    size_t centroid_size = K_CLUSTERS * sizeof(Pixel);

    Pixel *d_pixels, *d_centroids, *d_new_sums;
    int *d_labels, *d_new_counts;

    CHECK_CUDA(cudaMalloc(&d_pixels, pixel_size));
    CHECK_CUDA(cudaMalloc(&d_centroids, centroid_size));
    CHECK_CUDA(cudaMalloc(&d_new_sums, centroid_size));
    CHECK_CUDA(cudaMalloc(&d_labels, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_new_counts, K_CLUSTERS * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pixels, pixels.data(), pixel_size, cudaMemcpyHostToDevice));
    centroids.resize(K_CLUSTERS); 
    CHECK_CUDA(cudaMemcpy(d_centroids, pixels.data(), centroid_size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // printf("Starting GPU Naive (K=%d)...\n", K_CLUSTERS);
    cudaEventRecord(start);

    for (int iter = 0; iter < APP_MAX_ITER; iter++) {
        CHECK_CUDA(cudaMemset(d_new_sums, 0, centroid_size));
        CHECK_CUDA(cudaMemset(d_new_counts, 0, K_CLUSTERS * sizeof(int)));

        k_means_naive<<<gridSize, blockSize>>>(d_pixels, N, d_centroids, d_labels, d_new_sums, d_new_counts);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<Pixel> h_sums(K_CLUSTERS);
        std::vector<int> h_counts(K_CLUSTERS);
        CHECK_CUDA(cudaMemcpy(h_sums.data(), d_new_sums, centroid_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_new_counts, K_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<Pixel> h_centroids(K_CLUSTERS);
        for(int k=0; k<K_CLUSTERS; k++) {
            if(h_counts[k] > 0) {
                h_centroids[k].r = h_sums[k].r / h_counts[k];
                h_centroids[k].g = h_sums[k].g / h_counts[k];
                h_centroids[k].b = h_sums[k].b / h_counts[k];
            }
        }
        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), centroid_size, cudaMemcpyHostToDevice));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    CHECK_CUDA(cudaMemcpy(labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(centroids.data(), d_centroids, centroid_size, cudaMemcpyDeviceToHost));

    cudaFree(d_pixels); cudaFree(d_centroids); cudaFree(d_new_sums);
    cudaFree(d_labels); cudaFree(d_new_counts);

    return milliseconds;
}