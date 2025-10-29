#ifndef SIFT_READER_H
#define SIFT_READER_H

#include <string>
#include <vector>

using namespace std;

// Read SIFT vectors from .fvecs file
vector<vector<float>> read_sift(const string &full_path);

#endif
