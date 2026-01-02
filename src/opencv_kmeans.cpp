#include "kmeans.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>

void init_opencv() {
    cv::ocl::setUseOpenCL(false);
}

float run_opencv_kmeans(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids) {
    auto start = std::chrono::high_resolution_clock::now();

    int N = width * height;

    cv::Mat data(N, 1, CV_32FC3, (void*)pixels.data());
    cv::Mat data_reshaped = data.reshape(1, N);

    cv::Mat bestLabels;
    cv::Mat centers;

    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, APP_MAX_ITER, 0.01);
    
    cv::kmeans(data_reshaped, K_CLUSTERS, bestLabels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

    for(int i=0; i<N; i++) {
        labels[i] = bestLabels.at<int>(i);
    }

    centroids.resize(K_CLUSTERS);
    for(int i=0; i<K_CLUSTERS; i++) {
        centroids[i].r = centers.at<float>(i, 0);
        centroids[i].g = centers.at<float>(i, 1);
        centroids[i].b = centers.at<float>(i, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count();
}