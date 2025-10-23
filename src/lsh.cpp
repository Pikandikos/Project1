#include <cmath>
#include <functional>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "lsh.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;

LSH::LSH(int dim_, const LSHParams& p, size_t table_size_)
    : dim(dim_), params(p), table_size(table_size_), rng(p.seed), construction_time(0.0) {
    
    vs.resize(params.L);
    ts.resize(params.L);
    rints.resize(params.L);
    tables.resize(params.L);
    buckets_idx.resize(params.L);

    normal_distribution<float> nd(0.0f, 1.0f);
    uniform_real_distribution<double> ud(0.0, params.w);
    uniform_int_distribution<int64_t> rid(1, (1LL << 30) - 1);

    for (int i = 0; i < params.L; ++i) {
        vs[i].resize(params.k, vector<float>(dim));
        ts[i].resize(params.k);
        rints[i].resize(params.k);
        for (int j = 0; j < params.k; ++j) {
            for (int d = 0; d < dim; ++d) {
                vs[i][j][d] = nd(rng);
            }
            ts[i][j] = ud(rng);
            rints[i][j] = rid(rng);
        }
    }
}

int64_t LSH::compute_id(int table_id, const vector<float>& v) {
    int64_t sum = 0;
    for (int j = 0; j < params.k; ++j) {
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(v[d]) * static_cast<double>(vs[table_id][j][d]);
        }
        double val = (dot + ts[table_id][j]) / params.w;
        int64_t h = static_cast<int64_t>(floor(val + 1e-12));
        sum += rints[table_id][j] * h;
    }

    const uint64_t M = ((uint64_t)1 << 32) - 5;
    return sum % M;
}

int LSH::compute_bucket(int64_t fullID) {
    return static_cast<int>((static_cast<uint64_t>(fullID)) % table_size);
}

void LSH::build(const vector<vector<float>>& data) {
    auto start = high_resolution_clock::now();
    
    data_ptr = &data;
    size_t n = data.size();
    table_size = max<size_t>(31, n / 8);

    for (int i = 0; i < params.L; ++i) {
        tables[i].clear();
        buckets_idx[i].clear();
    }

    for (size_t id = 0; id < n; ++id) {
        const auto& vec = data[id];
        for (int i = 0; i < params.L; ++i) {
            int64_t fullID = compute_id(i, vec);
            int bucket = compute_bucket(fullID);
            tables[i][bucket].emplace_back(fullID, static_cast<int>(id));
            buckets_idx[i].push_back(bucket);
        }
    }
    
    auto end = high_resolution_clock::now();
    construction_time = duration_cast<duration<double>>(end - start).count();
}

vector<pair<int, double>> LSH::query(const vector<float>& q, int N) {
    unordered_set<int> candidates;
    
    // Collect candidates from all tables
    for (int i = 0; i < params.L; ++i) {
        int64_t queryFullID = compute_id(i, q);
        int bucket = compute_bucket(queryFullID);
        auto it = tables[i].find(bucket);
        if (it != tables[i].end()) {
            for (const auto& entry : it->second) {
                candidates.insert(entry.second);
            }
        }

        // Strategy 2: Also check neighboring buckets for high recall
        // Fix: cast table_size to int64_t for the calculation
        vector<int> neighbor_buckets;
        int64_t table_size_int = static_cast<int64_t>(table_size);
        
        // Calculate neighboring buckets with proper unsigned handling
        int prev_bucket = static_cast<int>((bucket - 1 + table_size_int) % table_size_int);
        int next_bucket = static_cast<int>((bucket + 1) % table_size_int);
        
        neighbor_buckets.push_back(prev_bucket);
        neighbor_buckets.push_back(next_bucket);
        
        for (int neighbor_bucket : neighbor_buckets) {
            auto neighbor_it = tables[i].find(neighbor_bucket);
            if (neighbor_it != tables[i].end()) {
                for (const auto& entry : neighbor_it->second) {
                    candidates.insert(entry.second);
                }
            }
        }

    }

    // Calculate distances and sort
    vector<pair<double, int>> cand;
    cand.reserve(candidates.size());
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        cand.emplace_back(dist, id);
    }
    
    sort(cand.begin(), cand.end());
    
    // Return top N results with distances
    vector<pair<int, double>> result;
    size_t result_count = min<size_t>(N, cand.size());
    result.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        result.emplace_back(cand[i].second, cand[i].first);
    }
    
    return result;
}

vector<int> LSH::range_query(const vector<float>& q, double R) {
    unordered_set<int> candidates;
    
    for (int i = 0; i < params.L; ++i) {
        int64_t queryFullID = compute_id(i, q);
        int bucket = compute_bucket(queryFullID);
        auto it = tables[i].find(bucket);
        if (it != tables[i].end()) {
            for (const auto& entry : it->second) {
                candidates.insert(entry.second);
            }
        }

        // Strategy 2: Also check neighboring buckets for high recall
        // Fix: cast table_size to int64_t for the calculation
        vector<int> neighbor_buckets;
        int64_t table_size_int = static_cast<int64_t>(table_size);
        
        // Calculate neighboring buckets with proper unsigned handling
        int prev_bucket = static_cast<int>((bucket - 1 + table_size_int) % table_size_int);
        int next_bucket = static_cast<int>((bucket + 1) % table_size_int);
        
        neighbor_buckets.push_back(prev_bucket);
        neighbor_buckets.push_back(next_bucket);
        
        for (int neighbor_bucket : neighbor_buckets) {
            auto neighbor_it = tables[i].find(neighbor_bucket);
            if (neighbor_it != tables[i].end()) {
                for (const auto& entry : neighbor_it->second) {
                    candidates.insert(entry.second);
                }
            }
        }

    }

    
    
    vector<int> result;
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        if (dist <= R) {
            result.push_back(id);
        }
    }
    
    return result;
}
