#include <stdexcept>

#include "dataset.h"

using namespace std;

// Parses command line arguments into a map for easy access
unordered_map<string, string> parseArgs(int argc, char *argv[])
{
    unordered_map<string, string> args;
    for (int i = 1; i < argc; ++i)
    {
        string key = argv[i];
        if (key[0] == '-')
        {
            // If next argument exists and is not another key, assign it as value
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                args[key] = argv[i + 1];
                ++i;
            }
            else
            {
                args[key] = ""; // flag with no value
            }
        }
    }

    for (auto &p : args)
    {
        cout << "Key: " << p.first << " | Value: " << p.second << endl;
    }

    return args;
}

uint32_t readBigEndianInt(ifstream &file)
{
    unsigned char bytes[4];
    file.read(reinterpret_cast<char *>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

// Function to reverse bytes (convert from big-endian to little-endian)
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to read MNIST images from file
vector<vector<unsigned char>> read_mnist_images(const string &full_path)
{
    ifstream file(full_path, ios::binary);
    if (!file.is_open())
    {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // Read metadata (header)
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    cout << "Magic number: " << magic_number << endl;
    cout << "Number of images: " << number_of_images << endl;
    cout << "Rows: " << n_rows << ", Cols: " << n_cols << endl;

    // Prepare storage
    vector<vector<unsigned char>> images(number_of_images, vector<unsigned char>(n_rows * n_cols));

    // Read all images pixel by pixel
    for (int i = 0; i < number_of_images; ++i)
    {
        file.read((char *)images[i].data(), n_rows * n_cols);
    }

    file.close();
    return images;
}

// Function to read MNIST labels
vector<unsigned char> read_mnist_labels(const string &full_path)
{
    ifstream file(full_path, ios::binary);
    if (!file.is_open())
    {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }

    int magic_number = 0;
    int number_of_items = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_items, sizeof(number_of_items));
    number_of_items = reverseInt(number_of_items);

    cout << "Magic number (labels): " << magic_number << endl;
    cout << "Number of labels: " << number_of_items << endl;

    vector<unsigned char> labels(number_of_items);
    file.read((char *)labels.data(), number_of_items);

    file.close();
    return labels;
}

std::vector<std::vector<double>> convertToDouble(const std::vector<std::vector<unsigned char>> &data)
{
    std::vector<std::vector<double>> result;
    result.reserve(data.size());
    for (const auto &row : data)
    {
        std::vector<double> converted(row.begin(), row.end());
        result.push_back(std::move(converted));
    }
    return result;
}

std::vector<std::vector<double>> convertToDouble(const std::vector<std::vector<float>> &data)
{
    std::vector<std::vector<double>> result;
    result.reserve(data.size());
    for (const auto &row : data)
    {
        std::vector<double> converted(row.begin(), row.end());
        result.push_back(std::move(converted));
    }
    return result;
}

vector<vector<float>> readIFTData(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("Cannot open IFT file!");

    vector<vector<float>> vectors;

    while (true)
    {
        uint32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
        if (file.eof())
            break;

        if (dim != 128)
            throw runtime_error("Invalid dimension (expected 128)!");

        vector<float> coords(dim);
        file.read(reinterpret_cast<char *>(coords.data()), dim * sizeof(float));

        if (!file)
            break;
        vectors.push_back(coords);
    }

    return vectors;
}