#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>
#include <fstream>
#include <algorithm>

#include "mnist_reader.h"
#include "lsh.h"
#include "utils.h"

// Simple cmd parsing; defaults match assignment
int main(int argc, char** argv) {

    /*
    cout << "Hello project!" << endl;
    string images_path = "dataset/train-images.idx3-ubyte"; // path to images
    string labels_path = "dataset/train-labels.idx1-ubyte"; // path to images

    auto images = read_mnist_images(images_path);
    auto labels = read_mnist_labels(labels_path);

    int n_rows = 28;
    int n_cols = 28; 

    //for(int i = 0; i < images[7].size(); i++){
      //  cout << (int)images[7][i] << " ";
    //}
    
    
    for(int i = 0; i < 5; i++){
        cout << "\n image label: " << (int)labels[i] << "\n\n";
        for (int r = 0; r < n_rows; ++r) { 
            for (int c = 0; c < n_cols; ++c) {
                unsigned char pixel = images[i][r * n_cols + c]; 
                cout << (pixel > 128 ? "#" : "."); // visualize 
                } cout << "\n"; 
            }
    }

    string test1 = "dataset/t10k-images.idx3-ubyte"; // path to images
    string test2 = "dataset/t10k-labels.idx1-ubyte"; // path to images

    auto test = read_mnist_images(test1);
    auto labeltest = read_mnist_labels(test2);

    cout << "\n image label: " << (int)labels[19619] << "\n\n";
    cout << "\n test label: " << (int)labeltest[0] << "\n\n";
    cout << "\n DISTANCE IS : " << euclidean_distance(test[2],images[53196]) << endl;

    return 0;*/
    

    std::string data_file="", query_file="", type="mnist", output_file="output.txt";
    LSHParams params;
    params.seed = 1; params.k = 4; params.L = 5; params.w = 4.0; params.N = 1; params.R = 2000.0;

    bool use_lsh = false;
    bool do_range = false;

    // parse minimal args (expandable)
    int opt;
    while ((opt = getopt(argc, argv, "d:q:o:k:L:w:N:R:seed:ttype:lsh:range:")) != -1) {
        // fallback: we'll parse manually simpler below
        break;
    }
    // Simple manual parse for common flags (robust parsing would be longer)
    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        if (a=="-d" && i+1<argc) data_file = argv[++i];
        else if (a=="-q" && i+1<argc) query_file = argv[++i];
        else if (a=="-o" && i+1<argc) output_file = argv[++i];
        else if (a=="-k" && i+1<argc) params.k = std::stoi(argv[++i]);
        else if (a=="-L" && i+1<argc) params.L = std::stoi(argv[++i]);
        else if (a=="-w" && i+1<argc) params.w = std::stod(argv[++i]);
        else if (a=="-N" && i+1<argc) params.N = std::stoi(argv[++i]);
        else if (a=="-R" && i+1<argc) params.R = std::stod(argv[++i]);
        else if (a=="--seed" && i+1<argc) params.seed = std::stoi(argv[++i]);
        else if (a=="-type" && i+1<argc) type = argv[++i];
        else if (a=="-lsh") use_lsh = true;
        else if (a=="-range" && i+1<argc) do_range = (std::string(argv[++i])=="true");
    }

    if (data_file.empty() || query_file.empty() || !use_lsh) {
        std::cout << "Usage example:\n./search -d train-images.idx3-ubyte -q t10k-images.idx3-ubyte -type mnist -lsh -k 4 -L 5 -w 4.0 -N 1 -R 2000 -o output.txt\n";
        return 1;
    }

    // Read dataset (only mnist implemented here)
    std::vector<std::vector<float>> data = read_mnist_images(data_file);
    std::vector<std::vector<float>> queries = read_mnist_images(query_file);

    std::cout << "Dataset size: " << data.size() << ", queries: " << queries.size() << std::endl;


    LSH lsh((int)data[0].size(), params, std::max<size_t>(31, data.size()/4));
    lsh.build(data);



    std::ofstream out(output_file);
    if (!out.is_open()) { std::cerr << "Cannot open output file\n"; return 1; }

    out << "LSH\n";
    Timer timer_total; timer_total.tic();
    double sumApproxTime=0, sumTrueTime=0;
    int totalQueries = (int)queries.size();
    int successesRecall = 0;
    for (int qi = 0; qi < 20 ; ++qi) {
        const auto &q = queries[qi];
        out << "Query: " << qi << "\n";
        // approximate
        Timer t1; t1.tic();
        auto approx_ids = lsh.query(q, params.N);
        double tApprox = t1.toc();
        sumApproxTime += tApprox;

        // true
        Timer t2; t2.tic();
        // compute full distances
        std::vector<std::pair<double,int>> all;
        all.reserve(data.size());
        for (size_t i=0;i<data.size();++i) all.emplace_back(euclidean_distance(q, data[i]), (int)i);
        std::sort(all.begin(), all.end());
        double tTrue = t2.toc();
        sumTrueTime += tTrue;
        
        //cout << " first: " << all[0].first << " middle: " << all[34].first << " last:  " << all[data.size() - 1].first << endl;
        
        // write approx neighbors and distances
        for (int i=0;i<params.N;i++) {
            if (i < (int)approx_ids.size()) {
                int id = approx_ids[i];
                double distApprox = euclidean_distance(q, data[id]);
                out << "Nearest neighbor-" << (i+1) << ": " << id << "\n";
                out << "distanceApproximate: " << distApprox << "\n";
                out << "distanceTrue: " << all[i].first << "\n";
            } else {
                out << "Nearest neighbor-" << (i+1) << ": -1\n";
                out << "distanceApproximate: inf\n";
                out << "distanceTrue: " << all[i].first << "\n";
            }
        }

        if (do_range) {
            auto rn = lsh.range_query(q, params.R);
            out << "R-near neighbors:\n";
            for (int id: rn) out << id << "\n";
        }

        // AF and recall
        double af = 0.0;
        if (!approx_ids.empty()) {
            double approxDist = euclidean_distance(q, data[approx_ids[0]]);
            double trueDist = all[0].first;
            af = approxDist / trueDist;
        }
        out << "Average AF: " << af << "\n";
        // recall@N: check whether true NN exists in approx_ids
        bool found=false;
        for (int id: approx_ids) if (id == all[0].second) found=true;
        if (found) successesRecall++;
        double recallAtN = double(successesRecall)/(qi+1);
        out << "Recall@N: " << recallAtN << "\n";
        out << "QPS: " << (1.0 / (sumApproxTime/(qi+1))) << "\n";
        out << "tApproximateAverage: " << (sumApproxTime/(qi+1)) << "\n";
        out << "tTrueAverage: " << (sumTrueTime/(qi+1)) << "\n\n";
    }
    double totalTime = timer_total.toc();
    std::cout << "Done. Output in " << output_file << "\nTotal time: " << totalTime << "s\n";
    return 0;
}
