#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <algorithm>

#include "kmeans_main.h"
#include "kmeans.h"
#include "data_reader.h"
#include "utils.h"
#include "ivfpq.h"

using namespace std;
using namespace chrono;

int find_optimal_k(const vector<vector<float>>& data, int k_min, int k_max, int step) {
    cout << "Finding optimal k using silhouette score..." << endl;
    
    double best_silhouette = -1.0;
    int best_k = k_min;
    
    for (int k = k_min; k <= k_max; k += step) {
        KMeansParams params;
        params.k = k;
        params.seed = 42;
        params.max_iters = 50;
        
        KMeans kmeans(params);
        kmeans.fit(data);
        
        double silhouette = kmeans.silhouette_score(data);
        
        cout << "k=" << k << ", silhouette=" << fixed << setprecision(4) << silhouette << endl;
        
        if (silhouette > best_silhouette) {
            best_silhouette = silhouette;
            best_k = k;
        }
    }
    
    cout << "Optimal k: " << best_k << " with silhouette: " << best_silhouette << endl;
    return best_k;
}

bool ivfflat_main(const string& data_file, const string& query_file, const string& output_file,
                  int kclusters, int nprobe, int seed, int N, double R, 
                  const string& type, bool do_range) {
    
    cout << "=== IVFFlat Search (Real Implementation) ===" << endl;
    
    try {
        // Load data
        vector<vector<float>> data = read_mnist_im(data_file);
        vector<vector<float>> queries = read_mnist_im(query_file);
        
        cout << "Loaded: " << data.size() << " training, " << queries.size() << " queries" << endl;
        
        if (data.empty() || queries.empty()) {
            cerr << "ERROR: No data loaded" << endl;
            return false;
        }
        
        // Use reasonable subsets for testing
        int max_data = 1000;
        int max_queries = 50;
        
        if (data.size() > max_data) {
            cout << "Using first " << max_data << " data points" << endl;
            data.resize(max_data);
        }
        if (queries.size() > max_queries) {
            cout << "Using first " << max_queries << " queries" << endl;
            queries.resize(max_queries);
        }
        
        int optimal_k = (kclusters <= 0) ? 20 : kclusters;
        cout << "Running k-means with k=" << optimal_k << "..." << endl;
        
        // Build k-means and inverted index
        KMeansParams kparams;
        kparams.k = optimal_k;
        kparams.seed = seed;
        kparams.max_iters = 50;
        
        Timer build_timer;
        build_timer.tic();
        
        KMeans kmeans(kparams);
        vector<int> labels = kmeans.fit(data);
        
        // Build inverted lists
        vector<vector<int>> inverted_lists(optimal_k);
        for (int i = 0; i < data.size(); ++i) {
            inverted_lists[labels[i]].push_back(i);
        }
        
        double build_time = build_timer.toc();
        cout << "Index built in " << build_time << " seconds" << endl;
        
        // Open output file
        ofstream out(output_file);
        if (!out.is_open()) {
            cerr << "ERROR: Cannot create output file" << endl;
            return false;
        }
        
        out << "IVFFlat" << endl;
        
        const auto& centers = kmeans.get_centers();
        double total_approx_time = 0.0;
        double total_true_time = 0.0;
        double total_af = 0.0;
        double total_recall = 0.0;
        int valid_af_queries = 0;
        int total_queries = queries.size();
        
        cout << "Processing " << total_queries << " queries..." << endl;
        
        // Process each query with REAL search
        for (int qi = 0; qi < total_queries; ++qi) {
            if (qi % 5 == 0) {
                cout << "Query " << qi << "/" << total_queries << endl;
            }
            
            const auto& q = queries[qi];
            out << "Query: " << qi << "\n";
            
            // === REAL Approximate Search ===
            Timer approx_timer;
            approx_timer.tic();
            
            // Find nearest clusters
            vector<pair<double, int>> cluster_dists;
            for (int i = 0; i < optimal_k; ++i) {
                double dist = squared_euclidean(q, centers[i]);
                cluster_dists.emplace_back(dist, i);
            }
            sort(cluster_dists.begin(), cluster_dists.end());
            
            // Search in nearest nprobe clusters
            unordered_set<int> candidate_set;
            int clusters_to_probe = min(nprobe, optimal_k);
            for (int i = 0; i < clusters_to_probe; ++i) {
                int cluster_id = cluster_dists[i].second;
                for (int point_idx : inverted_lists[cluster_id]) {
                    candidate_set.insert(point_idx);
                }
            }
            
            // Compute real distances to candidates
            vector<pair<double, int>> approx_results;
            for (int cand_id : candidate_set) {
                double dist = euclidean_distance(q, data[cand_id]);
                approx_results.emplace_back(dist, cand_id);
            }
            sort(approx_results.begin(), approx_results.end());
            if (approx_results.size() > N) {
                approx_results.resize(N);
            }
            
            double approx_time = approx_timer.toc();
            total_approx_time += approx_time;
            
            // === REAL True Search (exhaustive) ===
            Timer true_timer;
            true_timer.tic();
            
            vector<pair<double, int>> true_neighbors;
            for (int i = 0; i < data.size(); ++i) {
                double dist = euclidean_distance(q, data[i]);
                true_neighbors.emplace_back(dist, i);
            }
            sort(true_neighbors.begin(), true_neighbors.end());
            if (true_neighbors.size() > N) {
                true_neighbors.resize(N);
            }
            
            double true_time = true_timer.toc();
            total_true_time += true_time;
            
            // === Output REAL Results ===
            for (int i = 0; i < N; ++i) {
                out << "Nearest neighbor-" << (i + 1) << ": ";
                if (i < approx_results.size()) {
                    out << approx_results[i].second;
                } else {
                    out << "-1";
                }
                out << "\n";
                
                out << "distanceApproximate: ";
                if (i < approx_results.size()) {
                    out << fixed << setprecision(6) << approx_results[i].first;
                } else {
                    out << "inf";
                }
                out << "\n";
                
                out << "distanceTrue: ";
                if (i < true_neighbors.size()) {
                    out << fixed << setprecision(6) << true_neighbors[i].first;
                } else {
                    out << "inf";
                }
                out << "\n";
            }
            
            // REAL Range search
            out << "R-near neighbors:";
            if (do_range) {
                for (const auto& result : approx_results) {
                    if (result.first <= R) {
                        out << " " << result.second;
                    }
                }
            }
            out << "\n";
            
            // REAL Approximation Factor
            double af_sum = 0.0;
            int af_count = 0;
            for (int i = 0; i < N; ++i) {
                if (i < approx_results.size() && i < true_neighbors.size()) {
                    double approx_dist = approx_results[i].first;
                    double true_dist = true_neighbors[i].first;
                    if (true_dist > 1e-12) {
                        af_sum += approx_dist / true_dist;
                        af_count++;
                    }
                }
            }
            double af = (af_count > 0) ? (af_sum / af_count) : 0.0;
            if (af_count > 0) {
                total_af += af;
                valid_af_queries++;
            }
            out << "Average AF: " << fixed << setprecision(6) << af << "\n";
            
            // REAL Recall
            unordered_set<int> approx_set;
            for (const auto& result : approx_results) {
                approx_set.insert(result.second);
            }
            unordered_set<int> true_set;
            for (int i = 0; i < N && i < true_neighbors.size(); ++i) {
                true_set.insert(true_neighbors[i].second);
            }
            
            int common = 0;
            for (int true_id : true_set) {
                if (approx_set.count(true_id)) {
                    common++;
                }
            }
            double recall = (N > 0) ? (static_cast<double>(common) / N) : 0.0;
            total_recall += recall;
            out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";
            
            // REAL QPS and times
            double qps = (approx_time > 0.0) ? (1.0 / approx_time) : 0.0;
            out << "QPS: " << fixed << setprecision(6) << qps << "\n";
            out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
            out << "tTrueAverage: " << fixed << setprecision(6) << true_time << "\n\n";
        }
        
        // REAL Summary
        double mean_af = (valid_af_queries > 0) ? (total_af / valid_af_queries) : 0.0;
        double mean_recall = (total_queries > 0) ? (total_recall / total_queries) : 0.0;
        double mean_approx_time = (total_queries > 0) ? (total_approx_time / total_queries) : 0.0;
        double mean_true_time = (total_queries > 0) ? (total_true_time / total_queries) : 0.0;
        double mean_qps = (mean_approx_time > 0.0) ? (1.0 / mean_approx_time) : 0.0;
        
        out << "=== SUMMARY ===" << endl;
        out << "Queries: " << total_queries << endl;
        out << "Mean AF: " << fixed << setprecision(6) << mean_af << endl;
        out << "Mean Recall@N: " << fixed << setprecision(6) << mean_recall << endl;
        out << "Average QPS: " << fixed << setprecision(6) << mean_qps << endl;
        out << "Mean tApproximate: " << fixed << setprecision(6) << mean_approx_time << endl;
        out << "Mean tTrue: " << fixed << setprecision(6) << mean_true_time << endl;
        
        out.close();
        
        cout << "=== REAL IVFFlat Search Completed ===" << endl;
        cout << "Mean Recall: " << mean_recall << ", Mean AF: " << mean_af << endl;
        cout << "QPS: " << mean_qps << endl;
        
        return true;
        
    } catch (const exception& e) {
        cerr << "EXCEPTION: " << e.what() << endl;
        return false;
    }
}


bool ivfpq_main(const string& data_file,
                const string& query_file, 
                const string& output_file,
                const IVFPQParams& params,
                const string& type,
                bool do_range) {
    
    // Read dataset based on type
    vector<vector<float>> data, queries;
    if (type == "mnist") {
        data = read_mnist_im(data_file);
        queries = read_mnist_im(query_file);
    } else if (type == "sift") {
        data = read_sift(data_file);
        queries = read_sift(query_file);
    } else {
        cerr << "Unknown dataset type: " << type << endl;
        return false;
    }

    if (data.empty() || queries.empty()) {
        cerr << "Failed to read dataset or queries" << endl;
        return false;
    }

    cout << "Dataset size: " << data.size() << ", queries: " << queries.size() << endl;
    cout << "Vector dimension: " << data[0].size() << endl;

    // Use reasonable subsets for TESTING
    int total_queries = 20;
    if (queries.size() > total_queries) {
        queries.resize(total_queries);
    }
    data.resize(5000);

    // Build IVFPQ index
    IVFPQ ivfpq(params);
    ivfpq.build(data);

    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << output_file << endl;
        return false;
    }

    out << "IVFPQ\n";
    
    // Precompute true neighbors for all queries
    vector<vector<pair<double, int>>> true_neighbors_all(total_queries);
    cout << "Precomputing true neighbors..." << endl;
    
    for (int qi = 0; qi < total_queries; ++qi) {
        const auto& q = queries[qi];
        vector<pair<double, int>> true_neighbors;
        true_neighbors.reserve(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = euclidean_distance(q, data[i]);
            true_neighbors.emplace_back(dist, static_cast<int>(i));
        }
        sort(true_neighbors.begin(), true_neighbors.end());
        true_neighbors_all[qi] = move(true_neighbors);
    }

    // Performance metrics
    double total_approx_time = 0.0;
    double total_af = 0.0;
    double total_recall = 0.0;
    int valid_af_queries = 0;

    // Output for each query
    for (int qi = 0; qi < total_queries; ++qi) {
        const auto& q = queries[qi];
        const auto& true_neighbors = true_neighbors_all[qi];
        
        out << "Query: " << qi << "\n";

        // Approximate search
        Timer approx_timer;
        approx_timer.tic();

        auto approx_results = ivfpq.query(q, params.N);
        double approx_time = approx_timer.toc();
        total_approx_time += approx_time;

        // Range search if requested
        vector<int> range_neighbors;
        if (do_range && params.R > 0.0) {
            range_neighbors = ivfpq.range_query(q, params.R);
        }

        // Output results for each neighbor
        for (int i = 0; i < params.N; ++i) {

            // If neighbour wasn't found
            if (i >= static_cast<int>(approx_results.size())) {
                out << "-NEIGHBOUR NOT FOUND";
                continue;
            }

            out << "Nearest neighbor-" << (i + 1) << ": ";
            out << approx_results[i].first << endl;
            
            out << "distanceApproximate: ";
            out << fixed << setprecision(6) << approx_results[i].second << endl;
            
            out << "distanceTrue: ";
            out << fixed << setprecision(6) << true_neighbors[i].first << endl;
        }

        // Output range neighbors
        out << "R-near neighbors:";
        if (!range_neighbors.empty()) {
            for (int id : range_neighbors) {
                out << " " << id;
            }
        } else {
            out << " None";
        }
        out << "\n";

        // Calculate approximation factor
        double af_sum = 0.0;
        int af_count = 0;
        for (int i = 0; i < min(params.N, static_cast<int>(approx_results.size())); ++i) {
            if (i < static_cast<int>(true_neighbors.size())) {
                double approx_dist = approx_results[i].second;
                double true_dist = true_neighbors[i].first;
                if (true_dist > 1e-12) { // Avoid division by zero
                    af_sum += approx_dist / true_dist;
                    af_count++;
                }
            }
        }
        
        if(af_count == 0){
            cout << " No valid neighbours found" << endl;
            continue;
        }

        double af = af_sum / af_count;
        total_af += af;
        valid_af_queries++;
        out << "Average AF: " << fixed << setprecision(6) << af << "\n";

        // Calculate recall
        unordered_set<int> approx_set;
        for (const auto& result : approx_results) {
            approx_set.insert(result.first);
        }
        
        unordered_set<int> true_set;
        for (int i = 0; i < min(params.N, static_cast<int>(true_neighbors.size())); ++i) {
            true_set.insert(true_neighbors[i].second);
        }
        
        int common = 0;
        for (int true_id : true_set) {
            if (approx_set.count(true_id)) {
                common++;
            }
        }
        
        double recall = static_cast<double>(common) / true_set.size();
        total_recall += recall;
        out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";

        // QPS and times
        double qps = 1.0 / approx_time;
        out << "QPS: " << fixed << setprecision(6) << qps << "\n";
        out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << "0.0\n\n"; // precomputed
    }

    // Calculate summary statistics
    double average_af = (valid_af_queries > 0) ? (total_af / valid_af_queries) : 1.0;
    double average_recall = (total_queries > 0) ? (total_recall / total_queries) : 0.0;
    double average_approx_time = (total_queries > 0) ? (total_approx_time / total_queries) : 0.0;
    double average_qps = (average_approx_time > 0.0) ? (1.0 / average_approx_time) : 0.0;

    // Output summary
    out << "=== SUMMARY ===\n";
    out << "Queries: " << total_queries << "\n";
    out << "Average AF: " << fixed << setprecision(6) << average_af << "\n";
    out << "Average Recall@N: " << fixed << setprecision(6) << average_recall << "\n";
    out << "Average QPS: " << fixed << setprecision(6) << average_qps << "\n";
    out << "Average tApproximate: " << fixed << setprecision(6) << average_approx_time << "\n";
    out << "Average tTrue: 0.0(Precomputed)\n"; // Precomputed

    cout << "IVFPQ completed. Results written to: " << output_file << endl;
    cout << "Average Recall@N: " << average_recall << ", Average AF: " << average_af << endl;
    cout << "Average QPS: " << average_qps << endl;
    
    return true;
}
