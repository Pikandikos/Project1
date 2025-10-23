#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <string>
#include <vector>

using namespace std;

//Read mnist images
vector<vector<float>> read_mnist_im(const std::string &full_path);
//Read mnist labels
vector<unsigned char> read_mnist_l(const std::string &full_path);

#endif