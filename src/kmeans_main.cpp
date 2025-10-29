#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <algorithm>

#include "kmeans_main.h"
#include "kmeans.h"
#include "mnist_reader.h"
#include "sift_reader.h"
#include "utils.h"
#include "ivfpq.h"

using namespace std;
using namespace std::chrono;

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
                double dist = kmeans.squared_euclidean(q, centers[i]);
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
                double dist = sqrt(kmeans.squared_euclidean(q, data[cand_id]));
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
                double dist = sqrt(kmeans.squared_euclidean(q, data[i]));
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


bool ivfpq_main(const string& data_file, const string& query_file, const string& output_file,
                int kclusters, int nprobe, int M, int nbits, int seed, int N, double R, 
                const string& type, bool do_range) {
    
    cout << "=== IVFPQ Search ===" << endl;
    cout << "Parameters: kclusters=" << kclusters << ", nprobe=" << nprobe 
         << ", M=" << M << ", nbits=" << nbits << endl;
    
    try {
        // Load data
        vector<vector<float>> data;
        vector<vector<float>> queries;
        if( type == "mnist"){
            data = read_mnist_im(data_file);
            queries = read_mnist_im(query_file);
        }else if ( type == "sift")
        {
            data = read_sift(data_file);
            queries = read_sift(query_file);
        }else {
            cerr << "Unknown dataset type: " << type << endl;
            return false;
        }

        cout << "Loaded: " << data.size() << " training, " << queries.size() << " queries" << endl;
        
        if (data.empty() || queries.empty()) {
            cerr << "ERROR: No data loaded" << endl;
            return false;
        }
        
        // Use reasonable subsets for testing
        int max_queries = 20;

        if (queries.size() > max_queries) {
            cout << "Using first " << max_queries << " queries" << endl;
            queries.resize(max_queries);
        }
        
        // Build IVFPQ index
        IVFPQParams pq_params;
        pq_params.kclusters = (kclusters <= 0) ? 20 : kclusters;
        pq_params.nprobe = (nprobe <= 0) ? 3 : nprobe;
        pq_params.M = (M <= 0) ? 8 : M;
        pq_params.nbits = (nbits <= 0) ? 6 : nbits;
        pq_params.seed = seed;
        pq_params.N = N;
        pq_params.R = R;
        
        cout << "Building IVFPQ index..." << endl;
        Timer build_timer;
        build_timer.tic();
        
        IVFPQ ivfpq(pq_params);
        ivfpq.build(data);
        
        double build_time = build_timer.toc();
        cout << "IVFPQ index built in " << build_time << " seconds" << endl;
        
        // Open output file
        ofstream out(output_file);
        if (!out.is_open()) {
            cerr << "ERROR: Cannot create output file" << endl;
            return false;
        }
        
        out << "IVFPQ" << endl;
        
        double total_approx_time = 0.0;
        double total_true_time = 0.0;
        double total_af = 0.0;
        double total_recall = 0.0;
        int valid_af_queries = 0;
        int total_queries = queries.size();
        
        cout << "Processing " << total_queries << " queries..." << endl;
        
        // Process each query
        for (int qi = 0; qi < total_queries; ++qi) {
            if (qi % 2 == 0) {
                cout << "Query " << qi << "/" << total_queries << endl;
            }
            
            const auto& q = queries[qi];
            out << "Query: " << qi << "\n";
            
            // === IVFPQ Approximate Search ===
            Timer approx_timer;
            approx_timer.tic();
            
            auto approx_results = ivfpq.query(q, N);
            double approx_time = approx_timer.toc();
            total_approx_time += approx_time;
            
            // === True Exhaustive Search ===
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
            
            // Output results (same format as IVFFlat)
            for (int i = 0; i < N; ++i) {
                out << "Nearest neighbor-" << (i + 1) << ": ";
                if (i < approx_results.size()) {
                    out << approx_results[i].first;
                } else {
                    out << "-1";
                }
                out << "\n";
                
                out << "distanceApproximate: ";
                if (i < approx_results.size()) {
                    out << fixed << setprecision(6) << approx_results[i].second;
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
            
            // Range search
            out << "R-near neighbors:";
            if (do_range) {
                auto range_results = ivfpq.range_query(q, R);
                for (int id : range_results) {
                    out << " " << id;
                }
            }
            out << "\n";
            
            // Calculate metrics (same as IVFFlat)
            double af_sum = 0.0;
            int af_count = 0;
            for (int i = 0; i < N; ++i) {
                if (i < approx_results.size() && i < true_neighbors.size()) {
                    double approx_dist = approx_results[i].second;
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
            
            // Recall calculation
            unordered_set<int> approx_set;
            for (const auto& result : approx_results) {
                approx_set.insert(result.first);
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
            
            // QPS and times
            double qps = (approx_time > 0.0) ? (1.0 / approx_time) : 0.0;
            out << "QPS: " << fixed << setprecision(6) << qps << "\n";
            out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
            out << "tTrueAverage: " << fixed << setprecision(6) << true_time << "\n\n";
        }
        
        // Summary
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
        
        cout << "=== IVFPQ Search Completed ===" << endl;
        cout << "Mean Recall: " << mean_recall << ", Mean AF: " << mean_af << endl;
        cout << "QPS: " << mean_qps << endl;
        
        return true;
        
    } catch (const exception& e) {
        cerr << "EXCEPTION in ivfpq_main: " << e.what() << endl;
        return false;
    }
}
