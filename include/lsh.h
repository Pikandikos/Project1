#ifndef LSH_H
#define LSH_H

#include <vector>
#include <unordered_map>
#include <random>
#include <cstdint>
#include <utility>   // για std::pair

using namespace std;

// Δομή για όλες τις παραμέτρους του LSH
struct LSHParams {
    int seed = 1;     // τυχαίος seed για αναπαραγωγιμότητα
    int k = 4;        // αριθμός συναρτήσεων h ανά g
    int L = 5;        // αριθμός hash tables
    double w = 4.0;   // πλάτος κελιού (cell width)
    int N = 1;        // πόσους γείτονες να επιστρέφει
    double R = 2000.0;// radius για range query
};

// Κύρια κλάση LSH
class LSH {
public:
    // Constructor
    LSH(int dim, const LSHParams& params, size_t table_size = 10007);

    // Κατασκευή hash tables με βάση τα data vectors
    void build(const vector<vector<float>>& data);

    // Επιστρέφει τους Ν κοντινότερους (approximate)
    vector<int> query(const vector<float>& q, int N);

    // Επιστρέφει όλα τα σημεία εντός απόστασης R
    vector<int> range_query(const vector<float>& q, double R);

    // Debugging / πληροφορίες
    const vector<vector<int>>& buckets_for_L() const { return buckets_idx; }

private:
    int dim;                // διάσταση των διανυσμάτων
    LSHParams params;       // παράμετροι LSH
    size_t table_size;      // μέγεθος κάθε hash πίνακα
    mt19937 rng;            // random generator

    // Τυχαίες προβολές: L πίνακες, καθένας με k vectors των dim
    vector<vector<vector<float>>> vs;   // v_i,j
    vector<vector<double>> ts;          // t_i,j (offsets)
    vector<vector<int64_t>> rints;      // r_i,j (τυχαίοι ακέραιοι)

    // L hash tables: bucket -> λίστα από (fullID, pointID)
    vector<unordered_map<int, vector<pair<int64_t,int>>>> tables;

    // Helper functions
    int compute_bucket(int64_t fullID);
    int64_t compute_id(int table_id, const vector<float>& v);

    // Για debug
    vector<vector<int>> buckets_idx;

    // pointer στο dataset (ώστε να μη γίνεται αντιγραφή)
    const vector<vector<float>>* data_ptr = nullptr;
};

#endif
