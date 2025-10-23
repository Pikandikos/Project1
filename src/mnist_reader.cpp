#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "mnist_reader.h"

using namespace std;

static int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<vector<float>> read_mnist_im(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }
    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    vector<vector<float>> images(number_of_images, vector<float>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; ++i) {
        vector<unsigned char> buffer(n_rows * n_cols);
        file.read((char*)buffer.data(), n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; ++j) images[i][j] = static_cast<float>(buffer[j]);
    }
    file.close();
    return images;
}

vector<unsigned char> read_mnist_l(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }
    int magic_number = 0, number_of_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = reverseInt(number_of_items);

    vector<unsigned char> labels(number_of_items);
    file.read((char*)labels.data(), number_of_items);
    file.close();
    return labels;
}
