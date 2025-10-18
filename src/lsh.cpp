#include "lsh.h"
#include "utils.h"

#include <cmath>
#include <functional>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <iostream>

using namespace std;

LSH::LSH(int dim_, const LSHParams& p, size_t table_size_)
: dim(dim_), params(p), table_size(table_size_), rng(p.seed) {
    // allocate containers
    vs.resize(params.L);
    ts.resize(params.L);
    rints.resize(params.L);
    tables.resize(params.L);
    buckets_idx.resize(params.L);

    // RNG distributions
    std::normal_distribution<float> nd(0.0f, 1.0f);         // v ~ N(0,1)
    std::uniform_real_distribution<double> ud(0.0, params.w); // t ~ U[0,w)
    std::uniform_int_distribution<int64_t> rid(1, (1LL<<30)-1); // random ints for r_i

    // generate random projections, offsets and rints
    for (int i = 0; i < params.L; ++i) {
        vs[i].resize(params.k, vector<float>(dim));
        ts[i].resize(params.k);
        rints[i].resize(params.k);
        for (int j = 0; j < params.k; ++j) {
            for (int d = 0; d < dim; ++d) vs[i][j][d] = nd(rng);
            ts[i][j] = ud(rng);
            rints[i][j] = rid(rng);
        }
    }
}

// compute full ID = sum_j r_{i,j} * h_j(p)
int64_t LSH::compute_id(int table_id, const vector<float>& v) {
    int64_t sum = 0;
    for (int j = 0; j < params.k; ++j) {
        double dot = 0.0;
        // dot product v_{i,j} . v
        const vector<float>& proj = vs[table_id][j];
        for (int d = 0; d < dim; ++d) dot += double(proj[d]) * double(v[d]);

        double val = (dot + ts[table_id][j]) / params.w;
        int64_t h = (int64_t) std::floor(val);
        sum += rints[table_id][j] * h;
    }

    // Μ = 2^32 - 5
    const uint64_t M = ((uint64_t)1 << 32) - 5;
    sum = sum % M;

    return sum;
}

int LSH::compute_bucket(int64_t fullID) {
    int b = (int)( (uint64_t)(fullID) % table_size );
    return b;
}

void LSH::build(const vector<vector<float>>& data) {
    data_ptr = &data;
    size_t n = data.size();
    // 
    table_size = max<size_t>(31, n / 8);

    // clear tables (in case build called twice)
    for (int i = 0; i < params.L; ++i) {
        tables[i].clear();
        buckets_idx[i].clear();
    }

    for (size_t id = 0; id < n; ++id) {
        const auto &vec = data[id];
        for (int i = 0; i < params.L; ++i) {

            int64_t fullID = compute_id(i, vec);

            int bucket = compute_bucket(fullID);

            // store pair(fullID, pointIndex)
            tables[i][bucket].emplace_back(fullID, (int)id);
            // keep simple index list for debugging/inspection if needed
            buckets_idx[i].push_back(bucket);
        }
    }

}

// Query: φέρνει τους candidates με βασικό φίλτρο fullID == queryFullID
std::vector<int> LSH::query(const std::vector<float>& q, int N) {
    std::unordered_set<int> candidates;
    for (int i = 0; i < params.L; ++i) {
        // υπολογίζουμε fullID του query για αυτό το table
        int64_t queryFullID = compute_id(i, q);
        //cout << " Table: " << i << " ,ID : " << queryFullID << endl;
        int bucket = compute_bucket(queryFullID);

        auto it = tables[i].find(bucket);


        if (it != tables[i].end()) {
            //cout << " FOUND BUCKET: " << bucket << endl;
            // κάθε entry είναι pair(fullID, pointIndex)
            for (const auto &entry : it->second) {
                if (entry.first == queryFullID) {
                    cout << " FOUND same ID " << endl;
                    candidates.insert(entry.second);
                }else{
                    candidates.insert(entry.second);
                    //cout << " DIDNT FIND ID MATCHING: " << queryFullID << endl;
                    //cout << " entry ID is: " << entry.first << endl;
                    //int dataid = entry.second;

                }
            }
        }else{
            cout << " DIDNT FIND BUCKET: " << bucket<< endl;
        }
    }

    // Αν δεν βρέθηκαν candidates με ακριβές match του fullID, μπορούμε να κάνουμε fallback
    // (π.χ. να πάρουμε όλα τα στοιχεία των buckets). Εδώ επιλέγουμε as-is: δεν κάνουμε fallback.


    // Υπολογίζουμε ακριβείς αποστάσεις για τους candidates
    std::vector<std::pair<double,int>> cand;
    cand.reserve(candidates.size());

    
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        cand.emplace_back(dist, id);
    }
    std::sort(cand.begin(), cand.end());
    std::vector<int> result;
    for (int i = 0; i < (int)std::min<size_t>(N, cand.size()); ++i) result.push_back(cand[i].second);
    return result;
}

// Range query: παρόμοια λογική, επιστρέφουμε όλα με dist <= R
std::vector<int> LSH::range_query(const std::vector<float>& q, double R) {
    std::unordered_set<int> candidates;
    for (int i = 0; i < params.L; ++i) {
        int64_t queryFullID = compute_id(i, q);
        int bucket = (int)((uint64_t)queryFullID % table_size);
        auto it = tables[i].find(bucket);
        if (it != tables[i].end()) {
            for (const auto &entry : it->second) {
                if (entry.first == queryFullID) cout << "\n FOUND SAME ID" << endl;
                candidates.insert(entry.second);
            }
        }
    }
    std::vector<int> result;
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        if (dist <= R) result.push_back(id);
    }
    return result;
}