#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

#define K_CLUSTERS 16
#define APP_MAX_ITER 50

struct Pixel {
    float r, g, b;
};

void save_image(const std::string& folder, const std::string& tag, int width, int height, const std::vector<int>& labels, const std::vector<Pixel>& centroids) {
    std::string filename = folder + "/" + tag + ".png";
    std::vector<unsigned char> output_data(width * height * 3);
    for(int i=0; i<width*height; i++) {
        int label = labels[i];
        if(label >= 0 && label < K_CLUSTERS) {
            output_data[i*3 + 0] = (unsigned char)centroids[label].r;
            output_data[i*3 + 1] = (unsigned char)centroids[label].g;
            output_data[i*3 + 2] = (unsigned char)centroids[label].b;
        }
    }
    stbi_write_png(filename.c_str(), width, height, 3, output_data.data(), width * 3);
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: ./OpenCVGPU image1.jpg image2.jpg ..." << std::endl;
        return 1;
    }

    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::haveOpenCL()) {
        std::cerr << "Warning: OpenCL is not available. This will run on CPU." << std::endl;
    } else {
        std::cout << "OpenCL Device: " << cv::ocl::Context::getDefault().device(0).name() << std::endl;
    }

    double total_time = 0.0;
    int num_images = argc - 1;

    std::cout << "================================================================" << std::endl;
    std::cout << " OPENCV GPU (OpenCL) SUITE: " << num_images << " Images" << std::endl;
    std::cout << "================================================================" << std::endl;

    for (int i = 1; i < argc; i++) {
        std::string filepath = argv[i];
        fs::path p(filepath);
        std::string stem = p.stem().string();
        std::string out_dir = "results/" + stem;
        
        // Ensure directory exists (in case main.cu wasn't run first)
        fs::create_directory("results");
        fs::create_directory(out_dir);

        std::cout << "\nProcessing: " << stem << std::endl;

        int w, h, ch;
        unsigned char* img_data = stbi_load(filepath.c_str(), &w, &h, &ch, 3);
        if(!img_data) continue;

        std::vector<Pixel> pixels(w * h);
        for(int p=0; p<w*h; p++) {
            pixels[p].r = (float)img_data[p*3 + 0];
            pixels[p].g = (float)img_data[p*3 + 1];
            pixels[p].b = (float)img_data[p*3 + 2];
        }
        stbi_image_free(img_data);

        // Prep Data
        int N = w * h;
        cv::Mat host_data(N, 1, CV_32FC3, (void*)pixels.data());
        cv::Mat host_reshaped = host_data.reshape(1, N);
        cv::UMat gpu_data;
        host_reshaped.copyTo(gpu_data);

        cv::Mat bestLabels;
        cv::Mat centers;
        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, APP_MAX_ITER, 0.01);

        // Run
        auto start = std::chrono::high_resolution_clock::now();
        cv::kmeans(gpu_data, K_CLUSTERS, bestLabels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        float ms = diff.count() * 1000.0;
        total_time += ms;
        
        std::cout << "  -> Time: " << ms << " ms" << std::endl;

        // Save
        std::vector<int> labels(N);
        std::vector<Pixel> centroids(K_CLUSTERS);
        for(int i=0; i<N; i++) labels[i] = bestLabels.at<int>(i);
        for(int i=0; i<K_CLUSTERS; i++) {
            centroids[i].r = centers.at<float>(i, 0);
            centroids[i].g = centers.at<float>(i, 1);
            centroids[i].b = centers.at<float>(i, 2);
        }
        save_image(out_dir, "opencv_gpu", w, h, labels, centroids);
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << " FINAL AVERAGE: " << (total_time / num_images) << " ms" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}