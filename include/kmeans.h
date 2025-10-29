#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>

struct KMeansParams {
    int seed = 1;
    int k = 50;           // number of clusters
    int max_iters = 100;  // maximum iterations
    double tol = 1e-3;    // convergence tolerance
};

class KMeans {
public:
    KMeans(const KMeansParams& params = KMeansParams());
    
    // Fit k-means to data using Lloyd's algorithm (EM)
    std::vector<int> fit(const std::vector<std::vector<float>>& data);
    
    // Predict cluster assignments for new data
    std::vector<int> predict(const std::vector<std::vector<float>>& data) const;
    
    // Get cluster centers
    const std::vector<std::vector<float>>& get_centers() const { return centers; }
    
    // Get cluster assignments from last fit
    const std::vector<int>& get_labels() const { return labels; }
    
    // Get number of iterations performed
    int get_iterations() const { return iterations; }
    
    // Calculate silhouette score for clustering quality
    double silhouette_score(const std::vector<std::vector<float>>& data) const;
    
    //Kmeans objective is to minimize the cluster sum
    double cluster_sum(const std::vector<std::vector<float>>& data) const;

    // k-medians objective (alternative)
    double kmedians_objective(const std::vector<std::vector<float>>& data) const;

    // Distance functions
    double squared_euclidean(const std::vector<float>& a, const std::vector<float>& b) const;

private:
    KMeansParams params;
    std::vector<std::vector<float>> centers;
    std::vector<int> labels;
    int iterations;
    std::mt19937 rng;
    
    // Initialize centers using k-means++ algorithm (improved initialization)
    void kmeans_plus_plus_init(const std::vector<std::vector<float>>& data);
    
    // Lloyd's algorithm steps:
    void expectation_step(const std::vector<std::vector<float>>& data);  // Assignment
    bool maximization_step(const std::vector<std::vector<float>>& data); // Update centers
};

#endif
