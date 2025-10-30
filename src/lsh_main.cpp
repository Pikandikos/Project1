#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <iomanip>

#include "data_reader.h"
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


    //int total_queries = static_cast<int>(queries.size());
    int total_queries = 20; // just for testing

    // Precompute true neighbors for all queries (more efficient)
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
        // get neighbours from brute force
        const auto& true_neighbors = true_neighbors_all[qi];
        
        out << "Query: " << qi << "\n";

        // Approximate search
        Timer approx_timer;
        approx_timer.tic();

        auto approx_results = lsh.query(q, params.N);
        double approx_time = approx_timer.toc();
        total_approx_time += approx_time;

        // Range search if requested
        vector<int> range_neighbors;
        if (do_range && params.R > 0.0) {
            range_neighbors = lsh.range_query(q, params.R);
        }

        // Output results for each neighbor
        for (int i = 0; i < params.N; ++i) {

            // If neighbour wasnt found
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

        double af = af_sum / af_count; // 1.0 if no valid comparisons 
        total_af += af;
        valid_af_queries++;
        out << "Average AF: " << fixed << setprecision(6) << af << "\n";

        // Calculate recall
        // get approximate neighbours
        unordered_set<int> approx_set;
        for (const auto& result : approx_results) {
            approx_set.insert(result.first);
        }
        
        // get true neighbours
        unordered_set<int> true_set;
        for (int i = 0; i < min(params.N, static_cast<int>(true_neighbors.size())); ++i) {
            true_set.insert(true_neighbors[i].second);
        }
        
        // find common results
        int common = 0;
        for (int true_id : true_set) {
            if (approx_set.count(true_id)) {
                common++;
            }
        }
        
        double recall = static_cast<double>(common) / true_set.size();
        total_recall += recall;
        out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";

        // QPS and times (use precomputed true time or estimate)
        double qps = 1.0 / approx_time;
        out << "QPS: " << fixed << setprecision(6) << qps << "\n";
        out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << "0.0\n\n"; // Since we precomputed
    }

    // Calculate summary statistics
    // Check to not devide by 0
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

    cout << "LSH completed. Results written to: " << output_file << endl;
    cout << "Average Recall@N: " << average_recall << ", Average AF: " << average_af << endl;
    cout << "Average QPS: " << average_qps << endl;
    
    return true;
}
