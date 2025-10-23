#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <chrono>
#include <string>

using namespace std;

// Euclidean distance between two vectors
double euclidean_distance(const std::vector<float>& v1, const std::vector<float>& v2);

using Clock = std::chrono::high_resolution_clock;
struct Timer {
    Clock::time_point start;
    void tic() { start = Clock::now(); }
    double toc() { auto d = Clock::now() - start; return std::chrono::duration<double>(d).count(); }
};

#endif
