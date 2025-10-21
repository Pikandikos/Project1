#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <vector>
#include <string>
using namespace std;

bool hypercube_main(vector<vector<double>> dataset, vector<vector<double>> query, string outputFile,
                    int kproj, double w, int M, int probes, int N, double R, string type, bool rangeSearch);

// =====================================================================================
// Function: buildHypercube
// Description:
//   Builds the hypercube structure for fast nearest-neighbor search in high-dimensional
//   spaces using the dataset provided.
//
// Parameters:
//   data - The dataset points (each point is a vector of doubles).
//   k    - Number of hash projections (dimensions) for the hypercube.
// =====================================================================================
void buildHypercube(const vector<vector<double>> &data, int k);

// =====================================================================================
// Function: queryHypercube
// Description:
//   Searches the hypercube for the approximate nearest neighbors of a single query
//   point. It retrieves candidates from the same and nearby hypercube vertices,
//   calculates distances, and returns the closest N points.
//
// Parameters:
//   data          - The dataset points used to build the hypercube.
//   query         - The query vector to find neighbors for.
//   probes        - Maximum number of neighboring vertices to check.
//   num_neighbors - Number of nearest neighbors to return.
//
// Returns:
//   A vector of indices corresponding to the nearest dataset points.
// =====================================================================================
vector<int> queryHypercube(const vector<vector<double>> &data, const vector<double> &query,
                           int probes, int num_neighbors);

// =====================================================================================
// Function: searchHypercube
// Description:
//   Performs nearest-neighbor or range search using the hypercube.
//
// Parameters:
//   queries      - Set of query points.
//   N            - Number of nearest neighbors to find for each query.
//   R            - Radius for range search.
//   rangeSearch  - If true, perform range search; else, standard nearest neighbor.
//   M            - Maximum number of candidate points to check.
//   probes       - Maximum number of vertices to probe in the hypercube.
//   outputFile   - Path to the output file where results are stored.
// =====================================================================================
void searchHypercube(const vector<vector<double>> &queries, int N, double R, bool rangeSearch,
                     int M, int probes, const std::string &outputFile);

static vector<int> bruteForceNN(const vector<vector<double>> &dataset, const vector<double> &query, int N);

#endif // HYPERCUBE_H
