#include "utils.h"

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;


double euclidean_distance(const vector<float>& v1, const vector<float>& v2) {
    if (v1.size() != v2.size()) {
        throw invalid_argument("Vectors must have the same dimension for distance calculation");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = static_cast<double>(v1[i]) - static_cast<double>(v2[i]);
        sum += diff * diff;
    }
    return sqrt(sum);
}
