#include "../include/hypercube.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

// Global variables (local to this translation unit)
static unordered_map<string, vector<int>> hypercube; // vertex key → point indices
static vector<vector<double>> randomProjections;     // projection vectors
static vector<vector<double>> dataset_global;        // store dataset globally for search

// Utility: dot product
static double dot(const vector<double> &a, const vector<double> &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// Utility: Euclidean distance
static double euclidean(const vector<double> &a, const vector<double> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void buildHypercube(const vector<vector<double>> &dataset, int kproj)
{
    dataset_global = dataset;    // save for later query distance computations
    int dim = dataset[0].size(); // dimesnions tha points have

    // Create random projection vectors
    mt19937 gen(42);
    normal_distribution<> dist(0.0, 1.0);

    randomProjections.assign(kproj, vector<double>(dim));
    for (int i = 0; i < kproj; ++i)
        for (int j = 0; j < dim; ++j)
            randomProjections[i][j] = dist(gen);

    // Map each data point to a binary key based on projections
    for (int idx = 0; idx < (int)dataset.size(); ++idx)
    {
        string key;
        key.reserve(kproj);
        for (int i = 0; i < kproj; ++i)
            key.push_back(dot(dataset[idx], randomProjections[i]) >= 0 ? '1' : '0');

        hypercube[key].push_back(idx);
    }

    cout << "Hypercube built with " << hypercube.size() << " vertices.\n";
}

// Query: Approximate Nearest Neighbor using hypercube
vector<int> queryHypercube(const vector<vector<double>> &dataset, const vector<double> &query, int probes, int num_neighbors, int M)
{
    string key;
    int k = randomProjections.size();
    key.reserve(k);

    // Compute query's binary hash key
    for (int i = 0; i < k; ++i)
        key.push_back(dot(query, randomProjections[i]) >= 0 ? '1' : '0');

    vector<int> candidates;

    // Generate nearby binary keys (Hamming neighbors - each bit change represents a different vertex of the Hypercube)
    auto generateNeighbors = [&](const string &base_key, int max_dist)
    {
        vector<string> result;
        result.push_back(base_key);
        for (int d = 1; d <= max_dist; ++d)
        {
            for (int i = 0; i < (int)base_key.size(); ++i)
            {
                string flipped = base_key; // Bit flip
                flipped[i] = (flipped[i] == '1') ? '0' : '1';
                result.push_back(flipped);
                if ((int)result.size() >= probes)
                    return result;
            }
        }
        return result;
    };

    auto neighbor_keys = generateNeighbors(key, probes); // Generate possible neighboring buckets

    // Collect up to M candidate points
    for (const auto &nk : neighbor_keys)
    {
        auto it = hypercube.find(nk);
        if (it != hypercube.end())
        {
            const auto &pts = it->second;
            for (int idx : pts) // Add possible neighbors from that bucket in candidates
            {
                candidates.push_back(idx);
                if ((int)candidates.size() >= M)
                    break;
            }
            if ((int)candidates.size() >= M)
                break;
        }
    }

    // Compute distances and get top-N closest
    priority_queue<pair<double, int>> pq;
    for (int idx : candidates)
    {
        double d = euclidean(dataset[idx], query);
        pq.push({-d, idx}); // Adding -d so that bigger distances are concived as smaller and do not get top spots
        if ((int)pq.size() > num_neighbors)
            pq.pop();
    }

    vector<int> neighbors_out;
    while (!pq.empty())
    {
        neighbors_out.push_back(pq.top().second);
        pq.pop();
    }
    reverse(neighbors_out.begin(), neighbors_out.end());
    return neighbors_out;
}

// ----------------- True (exhaustive) Nearest Neighbors -----------------
static vector<int> bruteForceNN(const vector<vector<double>> &dataset, const vector<double> &query, int N)
{
    vector<pair<double, int>> dists;
    for (int i = 0; i < (int)dataset.size(); ++i)
        dists.push_back({euclidean(dataset[i], query), i});

    sort(dists.begin(), dists.end());
    vector<int> result;
    for (int i = 0; i < N && i < (int)dists.size(); ++i)
        result.push_back(dists[i].second);
    return result;
}

// Search: Iterate over queries and output results
void searchHypercube(const vector<vector<double>> &queries, int N, double R, bool rangeSearch, int M, int probes, const string &outputFile)
{
    ofstream out(outputFile);
    if (!out.is_open())
    {
        cerr << "Error: Could not open output file: " << outputFile << endl;
        return;
    }

    cout << "Starting Hypercube Search..." << endl;

    double totalApproxTime = 0.0, totalTrueTime = 0.0, totalRangeTime = 0.0;
    double totalAF = 0.0, totalRecall = 0.0;
    int totalQueries = (int)queries.size();

    cout << "   Mode: " << (rangeSearch ? "Range Search" : "Nearest Neighbor") << endl;
    cout << "   Output: " << outputFile << endl;

    auto startTotal = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < queries.size(); ++i)
    {
        const auto &q = queries[i];

        // Nearest neighbor search
        // --- Approximate Search ---
        auto startApprox = chrono::high_resolution_clock::now();
        vector<int> neighbors = queryHypercube(dataset_global, q, probes, N, M);
        auto endApprox = chrono::high_resolution_clock::now();
        double tApprox = chrono::duration<double>(endApprox - startApprox).count();
        totalApproxTime += tApprox;

        // --- True Search (ground truth) ---
        auto startTrue = chrono::high_resolution_clock::now();
        vector<int> trueNeighbors = bruteForceNN(dataset_global, q, N);
        auto endTrue = chrono::high_resolution_clock::now();
        double tTrue = chrono::duration<double>(endTrue - startTrue).count();
        totalTrueTime += tTrue;

        vector<int> range_neighbors;
        double rangeTime = 0.0;
        if (rangeSearch)
        {
            auto startRange = chrono::high_resolution_clock::now();
            for (int idx : neighbors)
            {
                double dist = euclidean(dataset_global[idx], q);
                if (dist <= R)
                    range_neighbors.push_back(idx);
            }
            auto endRange = chrono::high_resolution_clock::now();
            rangeTime = chrono::duration<double>(endRange - startRange).count();
            totalRangeTime += rangeTime;
        }

        // --- Compute Evaluation Metrics ---
        // AF = (distance to approximate NN) / (distance to true NN)
        // distances & AF: only if we have at least one approximate and one true neighbor
        double AF = -1.0;
        if (!neighbors.empty() && !trueNeighbors.empty())
        {
            double distApprox = euclidean(dataset_global[neighbors[0]], q);
            double distTrue = euclidean(dataset_global[trueNeighbors[0]], q);
            if (distTrue > 0.0)
                AF = distApprox / distTrue;
            else
                AF = (distApprox == 0.0) ? 1.0 : std::numeric_limits<double>::infinity();
            totalAF += AF;
        }
        else
        {
            // no valid AF for this query — don't add to totalAF (so we'll average properly below)
            AF = -1.0;
        }

        // Recall@N = overlap(true, approx) / N  (if N > 0)
        double recall = 0.0;
        if (N > 0 && !trueNeighbors.empty() && !neighbors.empty())
        {
            unordered_set<int> trueSet(trueNeighbors.begin(), trueNeighbors.end());
            int overlap = 0;
            for (int idx : neighbors)
                if (trueSet.count(idx))
                    overlap++;
            double recall = (double)overlap / N;
            totalRecall += recall;
        }
        else
        {
            recall = 0.0;
            // If you prefer not to count this query toward average recall,
            // you could track a separate counter. This example includes it as 0.
        }

        // --- Output ---
        out << "Query: " << i + 1 << "\n";
        cout << "\nQuery " << i + 1 << ":\n";

        if (!rangeSearch)
        {
            // Print up to N entries, but guard indices and distances
            for (int j = 0; j < N; ++j)
            {
                out << "Nearest neighbor-" << j + 1 << ": ";
                cout << "Nearest neighbor-" << j + 1 << ": ";

                if (j < (int)neighbors.size())
                {
                    int idxApprox = neighbors[j];
                    double distApprox = euclidean(dataset_global[idxApprox], q);
                    out << idxApprox << "\n";
                    out << "distanceApproximate: " << distApprox << "\n";
                    cout << idxApprox << " "; // keep console similar
                }
                else
                {
                    out << "-1\n";                      // placeholder for missing neighbor
                    out << "distanceApproximate: -1\n"; // placeholder distance
                }

                if (j < (int)trueNeighbors.size())
                {
                    int idxTrue = trueNeighbors[j];
                    double distTrue = euclidean(dataset_global[idxTrue], q);
                    out << "distanceTrue: " << distTrue << "\n";
                }
                else
                {
                    out << "distanceTrue: -1\n";
                }
            }

            // print AF and recall (AF might be -1 if not computable)
            if (AF >= 0.0)
                out << "ApproximationFactor(AF): " << AF << "\n";
            else
                out << "ApproximationFactor(AF): -1\n";
            out << "Recall@N: " << recall << "\n";
            out << "tApproximate: " << tApprox << "\n";
            out << "tTrue: " << tTrue << "\n";
        }
        else
        {
            out << "R-near neighbors:\n";
            for (int idx : range_neighbors)
                out << idx << "\n";
            out << "tRange: " << rangeTime << "\n"; // individual range time
        }

        out << "\n";
    }

    auto endTotal = chrono::high_resolution_clock::now(); // ⏱️ end total timer
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    // --- Summary Metrics ---
    double avgApproxTime = totalApproxTime / totalQueries;
    double avgTrueTime = totalTrueTime / totalQueries;
    double avgRangeTime = rangeSearch ? totalRangeTime / totalQueries : 0.0;
    double avgAF = totalAF / totalQueries;
    double avgRecall = totalRecall / totalQueries;
    double QPS = totalQueries / totalTime;

    // --- Summary in output file ---
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << QPS << "\n";
    out << "tApproximateAverage: " << avgApproxTime << "\n";
    out << "tTrueAverage: " << avgTrueTime << "\n";
    if (rangeSearch)
        out << "tRangeAverage: " << avgRangeTime << "\n";

    cout << "\nHypercube search completed.\nResults saved to: " << outputFile << endl;
    out.close();
}

bool hypercube_main(vector<vector<double>> dataset, vector<vector<double>> queries, string outputFile, int kproj, double w, int M, int probes,
                    int N, double R, string type, bool rangeSearch)
{
    cout << "\n=== Running Hypercube Algorithm ===\n";
    cout << "Parameters: kproj=" << kproj << ", w=" << w
         << ", M=" << M << ", probes=" << probes
         << ", N=" << N << ", R=" << R
         << ", rangeSearch=" << (rangeSearch ? "true" : "false") << "\n";

    buildHypercube(dataset, kproj);
    searchHypercube(queries, N, R, rangeSearch, M, probes, outputFile);

    cout << "\nHypercube algorithm finished successfully.\n";
    return true;
}