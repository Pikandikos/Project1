#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cmath>

#include "ivfpq.h"
#include "utils.h"
#include "kmeans.h"

using namespace std;

IVFPQ::IVFPQ(const IVFPQParams& params) : params(params) {
    // Initialize
    KMeansParams kmeans_params;
    kmeans_params.k = params.kclusters;
    kmeans_params.seed = params.seed;
    cluster_quantizer = KMeans(kmeans_params);
}

vector<vector<float>> IVFPQ::split_vector(const vector<float>& v) const {
    int dim = v.size();
    int sub_dim = dim / params.M;  // Dimension of each subvector
    
    vector<vector<float>> subvectors(params.M, vector<float>(sub_dim));
    
    // Split the vector into M equal parts
    for (int m = 0; m < params.M; ++m) {
        for (int d = 0; d < sub_dim; ++d) {
            subvectors[m][d] = v[m * sub_dim + d];
        }
    }
    return subvectors;
}

void IVFPQ::build(const vector<vector<float>>& data) {
    this->data = data;
    int n = data.size();
    int dim = data[0].size();
    int sub_dim = dim / params.M;  // Dimension per subspace
    int n_pq_centroids = 1 << params.nbits;  // 2^nbits centroids per subspace
    
    cout << "=== Building IVFPQ Index ===" << endl;
    cout << "Cluster quantizer: k=" << params.kclusters << " clusters" << endl;
    cout << "Product quantization: M=" << params.M << " subvectors, " 
         << n_pq_centroids << " centroids per subspace" << endl;
    cout << "Subvector dimension: " << sub_dim << endl;
    
    // 1: Cluster Quantization ===
    cout << "\n1. Building cluster quantizer..." << endl;
    vector<int> cluster_labels = cluster_quantizer.fit(data);
    
    // Build inverted lists for cluster quantization
    inverted_lists.resize(params.kclusters);
    for (int i = 0; i < n; ++i) {
        inverted_lists[cluster_labels[i]].push_back(i);
    }
    
    // === STEP 2: Compute Residuals ===
    cout << "2. Computing residuals..." << endl;
    const auto& cluster_centers = cluster_quantizer.get_centers();
    vector<vector<vector<float>>> residuals(params.kclusters);
    
    // For each point, compute residual = point - cluster_center
    for (int i = 0; i < n; ++i) {
        int cluster_id = cluster_labels[i];
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = data[i][d] - cluster_centers[cluster_id][d];
        }
        residuals[cluster_id].push_back(residual);
    }
    
    // === STEP 3: Train Product Quantizers ===
    cout << "3. Training product quantizers..." << endl;
    pq_centroids.resize(params.M);
    
    // Flatten all residuals for PQ training
    vector<vector<float>> all_residuals;
    for (const auto& cluster_residuals : residuals) {
        all_residuals.insert(all_residuals.end(), 
                           cluster_residuals.begin(), cluster_residuals.end());
    }
    
    // Train a separate k-means for each subspace
    KMeansParams pq_kmeans_params;
    pq_kmeans_params.k = n_pq_centroids;
    pq_kmeans_params.seed = params.seed;
    pq_kmeans_params.max_iters = 50;
    
    for (int m = 0; m < params.M; ++m) {
        cout << "   Training subspace " << m+1 << "/" << params.M << "..." << endl;
        
        // Extract m-th subspace from all residuals
        vector<vector<float>> subspace_data;
        for (const auto& residual : all_residuals) {
            vector<float> subvec(sub_dim);
            for (int d = 0; d < sub_dim; ++d) {
                subvec[d] = residual[m * sub_dim + d];
            }
            subspace_data.push_back(subvec);
        }
        
        // Train k-means on this subspace
        KMeans pq_kmeans(pq_kmeans_params);
        pq_kmeans.fit(subspace_data);
        pq_centroids[m] = pq_kmeans.get_centers();
    }
    
    // === STEP 4: Encode All Vectors ===
    cout << "4. Encoding vectors with product quantization..." << endl;
    pq_codes.resize(n, vector<uint8_t>(params.M));
    
    for (int i = 0; i < n; ++i) {
        int cluster_cluster = cluster_labels[i];
        
        // Compute residual for this point
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = data[i][d] - cluster_centers[cluster_cluster][d];
        }
        
        // Split residual and quantize each part
        auto subvectors = split_vector(residual);
        for (int m = 0; m < params.M; ++m) {
            double min_dist = numeric_limits<double>::max();
            uint8_t best_code = 0;
            
            // Find nearest centroid in this subspace
            for (int c = 0; c < n_pq_centroids; ++c) {
                double dist = squared_euclidean(subvectors[m], pq_centroids[m][c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = c;
                }
            }
            pq_codes[i][m] = best_code;
        }
    }
    
    cout << "=== IVFPQ Index Built Successfully ===" << endl;
    cout << "Total compressed vectors: " << n << endl;
}

void IVFPQ::build_lookup_table(const vector<float>& query_residual,
                              vector<vector<double>>& LUT) const {
    int sub_dim = data[0].size() / params.M;
    int n_pq_centroids = 1 << params.nbits;
    
    LUT.resize(params.M, vector<double>(n_pq_centroids));
    
    // Split query residual
    auto query_subvectors = split_vector(query_residual);
    
    // Precompute distances from query to all PQ centroids in each subspace
    for (int m = 0; m < params.M; ++m) {
        for (int c = 0; c < n_pq_centroids; ++c) {
            LUT[m][c] = squared_euclidean(query_subvectors[m], pq_centroids[m][c]);
        }
    }
}

double IVFPQ::asymmetric_distance(const vector<uint8_t>& code,
                                 const vector<vector<double>>& LUT) const {
    double total_dist = 0.0;
    
    // Sum distances from lookup table using the stored codes
    for (int m = 0; m < params.M; ++m) {
        total_dist += LUT[m][code[m]];
    }
    
    return total_dist;  // This is squared distance
}

vector<pair<int, double>> IVFPQ::query(const vector<float>& q, int N) const {
    const auto& cluster_centers = cluster_quantizer.get_centers();
    int dim = data[0].size();
    
    cout << "IVFPQ: Processing query with ASYMMETRIC distance computation..." << endl;
    
    // 1: Find nearest clusters
    vector<pair<double, int>> cluster_dists;
    for (int i = 0; i < params.kclusters; ++i) {
        double dist = squared_euclidean(q, cluster_centers[i]);
        cluster_dists.emplace_back(dist, i);
    }
    sort(cluster_dists.begin(), cluster_dists.end());
    
    // 2: Build lookup tables for each probed cluster
    vector<vector<vector<double>>> cluster_LUTs;
    int probes_used = min(params.nprobe, params.kclusters);
    
    for (int i = 0; i < probes_used; ++i) {
        int cluster_id = cluster_dists[i].second;
        
        // Compute residual = query - cluster_center
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = q[d] - cluster_centers[cluster_id][d];
        }
        
        // Build lookup table for this cluster
        vector<vector<double>> LUT;
        build_lookup_table(residual, LUT);
        cluster_LUTs.push_back(LUT);
    }
    
    // 3: Search in probed clusters and rank by ASYMMETRIC PQ distance
    vector<pair<double, int>> scored_candidates;  // (pq_distance, point_id)
    
    for (int i = 0; i < probes_used; ++i) {
        int cluster_id = cluster_dists[i].second;
        const auto& LUT = cluster_LUTs[i];
        
        // Score all points in this cluster using ASYMMETRIC distance
        for (int point_idx : inverted_lists[cluster_id]) {
            double pq_distance = asymmetric_distance(pq_codes[point_idx], LUT);
            scored_candidates.emplace_back(pq_distance, point_idx);
        }
    }
    
    // 4: Sort by ASYMMETRIC distance and take top N
    sort(scored_candidates.begin(), scored_candidates.end());
    
    vector<pair<int, double>> final_results;
    int result_count = min(N, static_cast<int>(scored_candidates.size()));
    
    // Return ASYMMETRIC distances
    for (int i = 0; i < result_count; ++i) {
        int point_id = scored_candidates[i].second;
        double asymmetric_dist = sqrt(scored_candidates[i].first); // Convert squared to actual distance
        
        final_results.emplace_back(point_id, asymmetric_dist);
    }
    
    cout << "IVFPQ: Found " << scored_candidates.size() << " candidates, returning top " 
         << result_count << " with ASYMMETRIC distances" << endl;
    
    return final_results;
}


vector<int> IVFPQ::range_query(const vector<float>& q, double R) const {
    // Get candidates with ACTUAL distances
    auto candidates_with_dist = query(q, data.size());
    vector<int> results;
    
    for (const auto& candidate : candidates_with_dist) {
        // candidate.second is the ACTUAL distance
        if (candidate.second <= R) {
            results.push_back(candidate.first);
        }
    }
    
    return results;
}

