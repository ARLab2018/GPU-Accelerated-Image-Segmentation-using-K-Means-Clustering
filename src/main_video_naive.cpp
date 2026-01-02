#include "kmeans_video.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h> // Include CUDA runtime

int main(int argc, char** argv) {
    // --- CRITICAL FIX: Force CUDA Initialization FIRST ---
    // We make a benign CUDA call to establish the context before OpenCV touches anything.
    cudaFree(0); 
    // ----------------------------------------------------

    // Disable OpenCL to avoid conflicts
    cv::ocl::setUseOpenCL(false);

    if (argc < 3) {
        std::cerr << "Usage: ./VideoNaive <input_video> <output_video>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video " << input_path << std::endl;
        return 1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "Processing Video: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    std::cout << "Algorithm: NAIVE GPU" << std::endl;

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    // Initialize Centroids randomly once
    std::vector<Pixel> centroids(K_CLUSTERS);
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0, 255);
    for(int i=0; i<K_CLUSTERS; i++) {
        centroids[i] = {dist(rng), dist(rng), dist(rng)};
    }

    cv::Mat frame;
    int frame_idx = 0;
    double total_compute_time = 0.0;

    std::vector<Pixel> pixels(width * height);
    std::vector<int> labels(width * height);

    while (cap.read(frame)) {
        // Convert cv::Mat (BGR) to Pixel vector (RGB)
        for(int i=0; i<width*height; i++) {
            cv::Vec3b bgr = frame.at<cv::Vec3b>(i);
            pixels[i].r = (float)bgr[2];
            pixels[i].g = (float)bgr[1];
            pixels[i].b = (float)bgr[0];
        }

        // RUN KERNEL
        float ms = run_gpu_naive_frame(pixels, width, height, labels, centroids);
        total_compute_time += ms;

        // Reconstruct Frame
        for(int i=0; i<width*height; i++) {
            int l = labels[i];
            // Safety check
            if (l >= 0 && l < K_CLUSTERS) {
                frame.at<cv::Vec3b>(i)[0] = (unsigned char)centroids[l].b;
                frame.at<cv::Vec3b>(i)[1] = (unsigned char)centroids[l].g;
                frame.at<cv::Vec3b>(i)[2] = (unsigned char)centroids[l].r;
            }
        }

        writer.write(frame);

        if (frame_idx % 10 == 0) {
            std::cout << "Frame " << frame_idx << "/" << total_frames 
                      << " | Last Frame: " << ms << " ms (" << (1000.0/ms) << " FPS)" << std::endl;
        }
        frame_idx++;
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << "NAIVE GPU SUMMARY" << std::endl;
    std::cout << "Total Frames: " << frame_idx << std::endl;
    std::cout << "Avg Compute Time: " << (total_compute_time / frame_idx) << " ms" << std::endl;
    std::cout << "Avg Processing FPS: " << (1000.0 / (total_compute_time / frame_idx)) << std::endl;
    std::cout << "==============================================" << std::endl;

    cap.release();
    writer.release();
    return 0;
}