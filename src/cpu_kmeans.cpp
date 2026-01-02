#include "kmeans.h"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <chrono>

float get_dist_sq(const Pixel& p1, const Pixel& p2) {
    return (p1.r - p2.r)*(p1.r - p2.r) + 
           (p1.g - p2.g)*(p1.g - p2.g) + 
           (p1.b - p2.b)*(p1.b - p2.b);
}

float run_cpu_kmeans(const std::vector<Pixel>& pixels, int width, int height, std::vector<int>& labels, std::vector<Pixel>& centroids) {
    auto start = std::chrono::high_resolution_clock::now();

    int N = width * height;
    
    std::mt19937 rng(1337); 
    std::uniform_int_distribution<int> dist(0, N - 1);
    
    centroids.resize(K_CLUSTERS);
    for(int i=0; i<K_CLUSTERS; i++) {
        centroids[i] = pixels[dist(rng)];
    }
    labels.resize(N);

    for(int iter=0; iter<APP_MAX_ITER; iter++) {
        std::vector<Pixel> new_centroids(K_CLUSTERS, {0,0,0});
        std::vector<int> counts(K_CLUSTERS, 0);

        for(int i=0; i<N; i++) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for(int k=0; k<K_CLUSTERS; k++) {
                float d = get_dist_sq(pixels[i], centroids[k]);
                if(d < min_dist) {
                    min_dist = d;
                    best_cluster = k;
                }
            }
            labels[i] = best_cluster;

            new_centroids[best_cluster].r += pixels[i].r;
            new_centroids[best_cluster].g += pixels[i].g;
            new_centroids[best_cluster].b += pixels[i].b;
            counts[best_cluster]++;
        }

        for(int k=0; k<K_CLUSTERS; k++) {
            if(counts[k] > 0) {
                centroids[k].r = new_centroids[k].r / counts[k];
                centroids[k].g = new_centroids[k].g / counts[k];
                centroids[k].b = new_centroids[k].b / counts[k];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count();
}