#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "kmeans.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem> // C++17 Filesystem

namespace fs = std::filesystem;

// Helper to save images with full path construction
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
    init_opencv();

    if(argc < 2) {
        std::cerr << "Usage: ./CudaKMeans image1.jpg image2.jpg ..." << std::endl;
        return 1;
    }

    // Create Main Results Directory
    fs::create_directory("results");

    double total_time_shared = 0.0;
    double total_time_naive = 0.0;
    double total_time_opencv = 0.0;
    double total_time_cpu = 0.0;
    
    int num_images = argc - 1;

    std::cout << "================================================================" << std::endl;
    std::cout << " MAIN BENCHMARK SUITE: " << num_images << " Images" << std::endl;
    std::cout << "================================================================" << std::endl;

    for (int i = 1; i < argc; i++) {
        std::string filepath = argv[i];
        fs::path p(filepath);
        std::string stem = p.stem().string(); // "image.jpg" -> "image"
        
        // Create Sub-directory: results/image_name/
        std::string out_dir = "results/" + stem;
        fs::create_directory(out_dir);

        std::cout << "\nProcessing [" << i << "/" << num_images << "]: " << stem << std::endl;

        int w, h, ch;
        unsigned char* img_data = stbi_load(filepath.c_str(), &w, &h, &ch, 3);
        if(!img_data) {
            std::cerr << "  -> Failed to load." << std::endl;
            continue;
        }

        std::vector<Pixel> pixels(w * h);
        for(int p=0; p<w*h; p++) {
            pixels[p].r = (float)img_data[p*3 + 0];
            pixels[p].g = (float)img_data[p*3 + 1];
            pixels[p].b = (float)img_data[p*3 + 2];
        }
        stbi_image_free(img_data);

        // --- 1. GPU SHARED ---
        {
            std::vector<int> labels(w * h);
            std::vector<Pixel> centroids;
            float t = run_gpu_shared(pixels, w, h, labels, centroids, 128); 
            total_time_shared += t;
            std::cout << "  -> GPU Shared: " << t << " ms" << std::endl;
            save_image(out_dir, "gpu_shared", w, h, labels, centroids);
        }

        // --- 2. GPU NAIVE ---
        {
            std::vector<int> labels(w * h);
            std::vector<Pixel> centroids;
            float t = run_gpu_naive(pixels, w, h, labels, centroids);
            total_time_naive += t;
            std::cout << "  -> GPU Naive:  " << t << " ms" << std::endl;
            save_image(out_dir, "gpu_naive", w, h, labels, centroids);
        }

        // --- 3. OPENCV CPU ---
        {
            std::vector<int> labels(w * h);
            std::vector<Pixel> centroids;
            float t = run_opencv_kmeans(pixels, w, h, labels, centroids);
            total_time_opencv += t;
            std::cout << "  -> OpenCV CPU: " << t << " ms" << std::endl;
            save_image(out_dir, "opencv_cpu", w, h, labels, centroids);
        }
        
        // --- 4. CPU SEQUENTIAL ---
        // WARNING: Uncomment only if you have patience!
        {
            std::vector<int> labels(w * h);
            std::vector<Pixel> centroids;
            float t = run_cpu_kmeans(pixels, w, h, labels, centroids);
            total_time_cpu += t;
            std::cout << "  -> CPU Seq:    " << t << " ms" << std::endl;
            save_image(out_dir, "CPU_Seq", w, h, labels, centroids);
        }
    }

//     std::cout << "\n================================================================" << std::endl;
//     std::cout << " AVERAGE EXECUTION TIME (Per Image)" << std::endl;
//     std::cout << "================================================================" << std::endl;
//     std::cout << std::fixed << std::setprecision(2);
//     std::cout << "GPU Optimized: " << (total_time_shared / num_images) << " ms" << std::endl;
//     std::cout << "GPU Baseline:  " << (total_time_naive / num_images) << " ms" << std::endl;
//     std::cout << "OpenCV CPU:    " << (total_time_opencv / num_images) << " ms" << std::endl;
//     std::cout << "================================================================" << std::endl;

//     return 0;
// }
        

    std::cout << "\n================================================================" << std::endl;
    std::cout << " FINAL RESULTS (Average per Image)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GPU Optimized (Shared): " << (total_time_shared / num_images) << " ms" << std::endl;
    std::cout << "GPU Baseline  (Naive):  " << (total_time_naive / num_images) << " ms" << std::endl;
    std::cout << "OpenCV        (CPU):    " << (total_time_opencv / num_images) << " ms" << std::endl;
    std::cout << "CPU           (Seq):    " << (total_time_cpu / num_images) << " ms" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}