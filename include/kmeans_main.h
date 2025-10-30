#ifndef KMEANS_MAIN_H
#define KMEANS_MAIN_H

#include <string>
#include <vector>

using namespace std;

// Function to find optimal k using silhouette score (as required by assignment)
int find_optimal_k(const vector<vector<float>>& data, 
                   int k_min = 20, int k_max = 100, int step = 10);

// IVFFlat main function that uses k-means clustering
bool ivfflat_main(const string& data_file, const string& query_file, 
                  const string& output_file, int kclusters, int nprobe, 
                  int seed, int N, double R, const string& type, bool do_range);

// IVFPQ main function that uses k-means clustering  
bool ivfpq_main(const string& data_file, const string& query_file,
                const string& output_file, int kclusters, int nprobe, 
                int M, int nbits, int seed, int N, double R, 
                const string& type, bool do_range);

#endif
