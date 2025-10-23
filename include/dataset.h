#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

using namespace std;

//---- Function Declarations ----

unordered_map<string, string> parseArgs(int argc, char *argv[]);

//Converts dataset (e.g., vector<Point>) into vector<vector<double>>
vector<vector<double>> convertToDouble(const vector<vector<unsigned char>> &data);
vector<vector<double>> convertToDouble(const vector<vector<float>> &data);

//Reads a 32-bit Big Endian integer from a file
uint32_t readBigEndianInt(ifstream &file);

//Reads MNIST image files (*.idx3-ubyte)
vector<vector<unsigned char>> read_mnist_images(const string &full_path);

//Reads MNIST label files (*.idx1-ubyte)
vector<unsigned char> read_mnist_labels(const string &full_path);

//Reads IFT dataset (little-endian binary file)
vector<vector<float>> readIFTData(const string &filename);

#endif