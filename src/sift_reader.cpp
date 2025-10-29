#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>

#include "sift_reader.h"

using namespace std;

vector<vector<float>> read_sift(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }

    vector<vector<float>> vectors;
    
    while (file) {
        // Read dimension (32-bit integer)
        int32_t dimension;
        file.read((char*)&dimension, sizeof(dimension));
        
        // Check if we reached end of file
        if (!file || file.eof()) {
            break;
        }
        
        // Read the 128 float coordinates
        vector<float> vector_data(dimension);
        file.read((char*)vector_data.data(), dimension * sizeof(float));
        
        // Check if we read all the data
        if (!file) {
            break;
        }
        
        vectors.push_back(vector_data);
    }
    
    file.close();
    return vectors;
}
