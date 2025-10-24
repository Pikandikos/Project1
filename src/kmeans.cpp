#include <numeric>
#include <iostream>
#include <queue>

#include "kmeans.h"

using namespace std;

KMeans::KMeans(const KMeansParams& params) 
    : params(params), iterations(0), rng(params.seed) {
}

vector<int> KMeans::fit(const vector<vector<float>>& data) {
    if (data.empty()) return {};
    
    cout << "Running Lloyd's algorithm for k-means with k=" << params.k << endl;
    
    // Step 1: Initialize centers using k-means++
    kmeans_plus_plus_init(data);
    
    // Lloyd's algorithm iterations
    bool converged = false;
    iterations = 0;
    
    for (int iter = 0; iter < params.max_iters && !converged; ++iter) {
        iterations++;
        
        // E-step: Assign points to nearest centers (Expectation)
        expectation_step(data);
        
        // M-step: Update centers (Maximization)
        converged = maximization_step(data);
        
        if (iter % 10 == 0) {
            double current_wcss = wcss(data);
            cout << "Iteration " << iter << ", WCSS = " << current_wcss << endl;
        }
    }
    
    cout << "Lloyd's algorithm converged after " << iterations << " iterations" << endl;
    cout << "Final k-means objective (WCSS): " << wcss(data) << endl;
    cout << "Silhouette score: " << silhouette_score(data) << endl;
    
    return labels;
}

void KMeans::kmeans_plus_plus_init(const vector<vector<float>>& data) {
    int n = data.size();
    centers.resize(params.k);
    
    // Step 1: Choose first center uniformly at random
    uniform_int_distribution<int> uniform(0, n - 1);
    centers[0] = data[uniform(rng)];
    
    vector<double> min_distances(n, numeric_limits<double>::max());
    
    // Steps 2-k: Choose remaining centers with probability proportional to D(x)^2
    for (int i = 1; i < params.k; ++i) {
        // Update minimum distances to nearest center
        double total_sq_distance = 0.0;
        for (int j = 0; j < n; ++j) {
            double dist = squared_euclidean(data[j], centers[i - 1]);
            if (dist < min_distances[j]) {
                min_distances[j] = dist;
            }
            total_sq_distance += min_distances[j];
        }
        
        // Choose next center with probability proportional to squared distance
        uniform_real_distribution<double> prob_dist(0.0, total_sq_distance);
        double threshold = prob_dist(rng);
        
        double cumulative = 0.0;
        for (int j = 0; j < n; ++j) {
            cumulative += min_distances[j];
            if (cumulative >= threshold) {
                centers[i] = data[j];
                break;
            }
        }
    }
    
    cout << "k-means++ initialization completed" << endl;
}

void KMeans::expectation_step(const vector<vector<float>>& data) {
    int n = data.size();
    labels.resize(n);
    
    // Assign each point to nearest centroid (Voronoi cell assignment)
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = 0;
        
        for (int j = 0; j < params.k; ++j) {
            double dist = squared_euclidean(data[i], centers[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
}

bool KMeans::maximization_step(const vector<vector<float>>& data) {
    int dim = data[0].size();
    vector<vector<float>> new_centers(params.k, vector<float>(dim, 0.0));
    vector<int> counts(params.k, 0);
    
    // Sum points in each cluster
    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        for (int d = 0; d < dim; ++d) {
            new_centers[cluster][d] += data[i][d];
        }
        counts[cluster]++;
    }
    
    // Compute new centers as means of each cluster
    bool converged = true;
    for (int i = 0; i < params.k; ++i) {
        if (counts[i] > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centers[i][d] /= counts[i];
            }
            
            // Check convergence: if any center moved significantly
            double movement = squared_euclidean(centers[i], new_centers[i]);
            if (movement > params.tol) {
                converged = false;
            }
        } else {
            // Empty cluster - reinitialize using k-means++ strategy
            cout << "Warning: Empty cluster " << i << ", reinitializing..." << endl;
            vector<double> distances(data.size());
            double total_dist = 0.0;
            for (size_t j = 0; j < data.size(); ++j) {
                distances[j] = squared_euclidean(data[j], centers[labels[j]]);
                total_dist += distances[j];
            }
            
            uniform_real_distribution<double> dist(0.0, total_dist);
            double threshold = dist(rng);
            double cumulative = 0.0;
            for (size_t j = 0; j < data.size(); ++j) {
                cumulative += distances[j];
                if (cumulative >= threshold) {
                    new_centers[i] = data[j];
                    break;
                }
            }
            converged = false;
        }
    }
    
    centers = move(new_centers);
    return converged;
}

vector<int> KMeans::predict(const vector<vector<float>>& data) const {
    if (centers.empty()) {
        throw runtime_error("KMeans must be fitted before prediction");
    }
    
    vector<int> predictions(data.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        
        for (int j = 0; j < params.k; ++j) {
            double dist = squared_euclidean(data[i], centers[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        predictions[i] = best_cluster;
    }
    
    return predictions;
}

double KMeans::squared_euclidean(const vector<float>& a, const vector<float>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

double KMeans::euclidean(const vector<float>& a, const vector<float>& b) const {
    return sqrt(squared_euclidean(a, b));
}

double KMeans::wcss(const vector<vector<float>>& data) const {
    double total = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        total += squared_euclidean(data[i], centers[labels[i]]);
    }
    return total;
}

double KMeans::kmedians_objective(const vector<vector<float>>& data) const {
    double total = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        total += euclidean(data[i], centers[labels[i]]);
    }
    return total;
}

double KMeans::silhouette_score(const vector<vector<float>>& data) const {
    int n = data.size();
    if (n == 0) return 0.0;
    
    vector<double> silhouette_scores(n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int cluster_i = labels[i];
        
        // Calculate a(i): average distance to points in same cluster
        double a_i = 0.0;
        int count_same = 0;
        
        for (int j = 0; j < n; ++j) {
            if (i != j && labels[j] == cluster_i) {
                a_i += euclidean(data[i], data[j]);
                count_same++;
            }
        }
        a_i = (count_same > 0) ? a_i / count_same : 0.0;
        
        // Calculate b(i): minimum average distance to other clusters
        double b_i = numeric_limits<double>::max();
        
        for (int c = 0; c < params.k; ++c) {
            if (c == cluster_i) continue;
            
            double avg_dist = 0.0;
            int count_other = 0;
            
            for (int j = 0; j < n; ++j) {
                if (labels[j] == c) {
                    avg_dist += euclidean(data[i], data[j]);
                    count_other++;
                }
            }
            
            if (count_other > 0) {
                avg_dist /= count_other;
                if (avg_dist < b_i) {
                    b_i = avg_dist;
                }
            }
        }
        
        // Handle case where b_i wasn't updated (only one cluster)
        if (b_i == numeric_limits<double>::max()) {
            b_i = a_i;
        }
        
        // Calculate silhouette for this point
        double max_ab = max(a_i, b_i);
        if (max_ab == 0.0) {
            silhouette_scores[i] = 0.0;
        } else {
            silhouette_scores[i] = (b_i - a_i) / max_ab;
        }
    }
    
    // Return average silhouette score
    double total = 0.0;
    for (double score : silhouette_scores) total += score;
    return total / n;
}
