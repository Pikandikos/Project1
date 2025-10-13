#include <iostream>
#include <string>
#include <fstream>


#include "mnist_reader.h"

using namespace std;

int main() {
    cout << "Hello project!" << endl;
    string images_path = "dataset/t10k-images.idx3-ubyte"; // path to images

    auto images = read_mnist_images(images_path);
    int n_rows = 28;
    int n_cols = 28; 
    
    for (int r = 0; r < n_rows; ++r) { 
        for (int c = 0; c < n_cols; ++c) {
            unsigned char pixel = images[0][r * n_cols + c]; 
            cout << (pixel > 128 ? "#" : "."); // visualize 
            } cout << "\n"; 
        }
    

    return 0;
}
