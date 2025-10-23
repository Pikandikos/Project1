#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

#include "mnist_reader.h"
#include "lsh.h"
#include "utils.h"

using namespace std;

bool lsh_main(const string& data_file,
              const string& query_file,
              const string& output_file,
              const LSHParams& params,
              const string& type,
              bool do_range) {
    
    // Read dataset based on type
    vector<vector<float>> data, queries;
    if (type == "mnist") {
        data = read_mnist_im(data_file);
        queries = read_mnist_im(query_file);
    } else if (type == "sift") {
        //data = read_sift(data_file);
        //queries = read_sift(query_file);
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

    // Build LSH index
    LSH lsh(static_cast<int>(data[0].size()), params);
    lsh.build(data);
    cout << "LSH construction time: " << lsh.get_construction_time() << " seconds" << endl;

    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << output_file << endl;
        return false;
    }

    out << "LSH\n";
    
    // Performance metrics
    double total_approx_time = 0.0;
    double total_true_time = 0.0;
    double total_af = 0.0;
    double total_recall = 0.0;
    int valid_af_queries = 0;
    //int total_queries = static_cast<int>(queries.size());
    int total_queries = 20;

    for (int qi = 0; qi < total_queries; ++qi) {
        const auto& q = queries[qi];
        out << "Query: " << qi << "\n";

        // Approximate search
        Timer approx_timer;
        approx_timer.tic();
        auto approx_results = lsh.query(q, params.N);
        double approx_time = approx_timer.toc();
        total_approx_time += approx_time;

        // True nearest neighbors (exhaustive search)
        Timer true_timer;
        true_timer.tic();
        vector<pair<double, int>> true_neighbors;
        true_neighbors.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = euclidean_distance(q, data[i]);
            true_neighbors.emplace_back(dist, static_cast<int>(i));
        }
        sort(true_neighbors.begin(), true_neighbors.end());
        double true_time = true_timer.toc();
        total_true_time += true_time;

        // Range search if requested
        vector<int> range_neighbors;
        if (do_range && params.R > 0.0) {
            range_neighbors = lsh.range_query(q, params.R);
        }

        // Output results for each neighbor
        for (int i = 0; i < params.N; ++i) {
            out << "Nearest neighbor-" << (i + 1) << ": ";
            if (i < static_cast<int>(approx_results.size())) {
                out << approx_results[i].first;
            } else {
                out << "-1";
            }
            out << "\n";
            
            out << "distanceApproximate: ";
            if (i < static_cast<int>(approx_results.size())) {
                out << fixed << setprecision(6) << approx_results[i].second;
            } else {
                out << "inf";
            }
            out << "\n";
            
            out << "distanceTrue: ";
            if (i < static_cast<int>(true_neighbors.size())) {
                out << fixed << setprecision(6) << true_neighbors[i].first;
            } else {
                out << "inf";
            }
            out << "\n";
        }

        // Output range neighbors
        out << "R-near neighbors:";
        if (!range_neighbors.empty()) {
            for (int id : range_neighbors) {
                out << " " << id;
            }
        }
        out << "\n";

        // Calculate approximation factor
        double af_sum = 0.0;
        int af_count = 0;
        for (int i = 0; i < params.N; ++i) {
            if (i < static_cast<int>(approx_results.size()) && 
                i < static_cast<int>(true_neighbors.size())) {
                double approx_dist = approx_results[i].second;
                double true_dist = true_neighbors[i].first;
                if (true_dist > 1e-12) { // Avoid division by zero
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

        // Calculate recall
        unordered_set<int> approx_set;
        for (const auto& result : approx_results) {
            approx_set.insert(result.first);
        }
        
        unordered_set<int> true_set;
        for (int i = 0; i < params.N && i < static_cast<int>(true_neighbors.size()); ++i) {
            true_set.insert(true_neighbors[i].second);
        }
        
        int common = 0;
        for (int true_id : true_set) {
            if (approx_set.count(true_id)) {
                common++;
            }
        }
        
        double recall = (params.N > 0) ? (static_cast<double>(common) / params.N) : 0.0;
        total_recall += recall;
        out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";

        // QPS and times
        double qps = (approx_time > 0.0) ? (1.0 / approx_time) : 0.0;
        out << "QPS: " << fixed << setprecision(6) << qps << "\n";
        out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << fixed << setprecision(6) << true_time << "\n\n";
    }

    // Calculate summary statistics
    double mean_af = (valid_af_queries > 0) ? (total_af / valid_af_queries) : 0.0;
    double mean_recall = (total_queries > 0) ? (total_recall / total_queries) : 0.0;
    double mean_approx_time = (total_queries > 0) ? (total_approx_time / total_queries) : 0.0;
    double mean_true_time = (total_queries > 0) ? (total_true_time / total_queries) : 0.0;
    double mean_qps = (mean_approx_time > 0.0) ? (1.0 / mean_approx_time) : 0.0;

    // Output summary
    out << "=== SUMMARY ===\n";
    out << "Queries: " << total_queries << "\n";
    out << "Mean AF: " << fixed << setprecision(6) << mean_af << "\n";
    out << "Mean Recall@N: " << fixed << setprecision(6) << mean_recall << "\n";
    out << "Average QPS: " << fixed << setprecision(6) << mean_qps << "\n";
    out << "Mean tApproximate: " << fixed << setprecision(6) << mean_approx_time << "\n";
    out << "Mean tTrue: " << fixed << setprecision(6) << mean_true_time << "\n";

    cout << "LSH completed. Results written to: " << output_file << endl;
    cout << "Average Recall@N: " << mean_recall << ", Average AF: " << mean_af << endl;
    
    return true;
}
