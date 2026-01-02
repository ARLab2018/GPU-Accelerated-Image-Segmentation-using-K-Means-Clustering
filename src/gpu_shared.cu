#include "kmeans.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>

__constant__ Pixel c_centroids[K_CLUSTERS];

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void k_means_shared(const Pixel* pixels, int N, int* labels, Pixel* new_sums, int* new_counts) {
    __shared__ float s_sums_r[K_CLUSTERS];
    __shared__ float s_sums_g[K_CLUSTERS];
    __shared__ float s_sums_b[K_CLUSTERS];
    __shared__ int   s_counts[K_CLUSTERS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < K_CLUSTERS) {
        s_sums_r[tid] = 0.0f;
        s_sums_g[tid] = 0.0f;
        s_sums_b[tid] = 0.0f;
        s_counts[tid] = 0;
    }
    __syncthreads();

    if (idx < N) {
        Pixel p = pixels[idx];
        float min_dist = FLT_MAX;
        int best_k = 0;

        for (int k = 0; k < K_CLUSTERS; k++) {
            float r = p.r - c_centroids[k].r;
            float g = p.g - c_centroids[k].g;
            float b = p.b - c_centroids[k].b;
            float dist = r*r + g*g + b*b;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_k = k;
            }
        }
        labels[idx] = best_k;

        atomicAdd(&s_sums_r[best_k], p.r);
        atomicAdd(&s_sums_g[best_k], p.g);
        atomicAdd(&s_sums_b[best_k], p.b);
        atomicAdd(&s_counts[best_k], 1);
    }
    
    __syncthreads();

    if (tid < K_CLUSTERS) {
        atomicAdd(&new_sums[tid].r, s_sums_r[tid]);
        atomicAdd(&new_sums[tid].g, s_sums_g[tid]);
        atomicAdd(&new_sums[tid].b, s_sums_b[tid]);
        atomicAdd(&new_counts[tid], s_counts[tid]);
    }
}

float run_gpu_shared(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids, int blockSize) {
    int N = width * height;
    size_t pixel_size = N * sizeof(Pixel);
    size_t centroid_size = K_CLUSTERS * sizeof(Pixel);

    Pixel *d_pixels, *d_new_sums;
    int *d_labels, *d_new_counts;

    CHECK_CUDA(cudaMalloc(&d_pixels, pixel_size));
    CHECK_CUDA(cudaMalloc(&d_new_sums, centroid_size));
    CHECK_CUDA(cudaMalloc(&d_labels, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_new_counts, K_CLUSTERS * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_pixels, pixels.data(), pixel_size, cudaMemcpyHostToDevice));
    
    centroids.resize(K_CLUSTERS); 
    CHECK_CUDA(cudaMemcpyToSymbol(c_centroids, pixels.data(), centroid_size));

    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // printf("Starting GPU Optimized (Shared Mem)...\n");
    cudaEventRecord(start);

    for (int iter = 0; iter < APP_MAX_ITER; iter++) {
        CHECK_CUDA(cudaMemset(d_new_sums, 0, centroid_size));
        CHECK_CUDA(cudaMemset(d_new_counts, 0, K_CLUSTERS * sizeof(int)));

        k_means_shared<<<gridSize, blockSize>>>(d_pixels, N, d_labels, d_new_sums, d_new_counts);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<Pixel> h_sums(K_CLUSTERS);
        std::vector<int> h_counts(K_CLUSTERS);
        CHECK_CUDA(cudaMemcpy(h_sums.data(), d_new_sums, centroid_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_new_counts, K_CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));

        for(int k=0; k<K_CLUSTERS; k++) {
            if(h_counts[k] > 0) {
                centroids[k].r = h_sums[k].r / h_counts[k];
                centroids[k].g = h_sums[k].g / h_counts[k];
                centroids[k].b = h_sums[k].b / h_counts[k];
            }
        }
        CHECK_CUDA(cudaMemcpyToSymbol(c_centroids, centroids.data(), centroid_size));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CHECK_CUDA(cudaMemcpy(labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_pixels); cudaFree(d_new_sums);
    cudaFree(d_labels); cudaFree(d_new_counts);

    return milliseconds;
}