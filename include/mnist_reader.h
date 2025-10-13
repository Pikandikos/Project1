#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <string>
#include <vector>

using namespace std;

vector<vector<float>> read_mnist_images(const std::string &full_path);
vector<unsigned char> read_mnist_labels(const std::string &full_path);

#endif