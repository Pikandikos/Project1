#include "utils.h"

#include <iostream>
#include <cmath>


double euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    size_t d = a.size();
    for (size_t i = 0; i < d; ++i) {

        //std::cout << double(a[i]) << " - " << double(b[i]) << std::endl;
        double diff = double(a[i]) - double(b[i]);
        s += diff * diff;
    }
    return std::sqrt(s);
}
