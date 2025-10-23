//src/driver_main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>


#include "dataset.h"
#include "lsh.h"        //για LSHParams και αν έχεις δηλώσεις wrapper
//include headers για hypercube/ivf αν υπάρχουν (όπου δηλώνεται hypercube_main_files κλπ)
#include "hypercube.h"
//#include "ivfflat.h"
//#include "ivfpq.h"

using namespace std;


int main(int argc, char** argv) {
    string data_file="", query_file="", type="mnist", output_file="output.txt";
    LSHParams params; //uses your LSHParams struct from lsh.h
    params.seed = 1; params.k = 4; params.L = 5; params.w = 4.0; params.N = 1; params.R = 2000.0;

    //hypercube / ivf specific
    int kproj = 0, M = 0, probes = 0, kclusters = 0, nprobe = 0, seed = 0, nbits = 0;

    bool use_lsh = false, use_hypercube = false, use_ivfflat = false, use_ivfpq = false;
    bool rangeSearch = false;

    //Minimal manual parse (robust enough)
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-d" && i+1 < argc) data_file = argv[++i];
        else if (a == "-q" && i+1 < argc) query_file = argv[++i];
        else if (a == "-o" && i+1 < argc) output_file = argv[++i];
        else if (a == "-k" && i+1 < argc) params.k = stoi(argv[++i]);
        else if (a == "-L" && i+1 < argc) params.L = stoi(argv[++i]);
        else if (a == "-w" && i+1 < argc) params.w = stod(argv[++i]);
        else if (a == "-N" && i+1 < argc) params.N = stoi(argv[++i]);
        else if (a == "-R" && i+1 < argc) params.R = stod(argv[++i]);
        else if (a == "--seed" && i+1 < argc) params.seed = stoi(argv[++i]);
        else if (a == "-type" && i+1 < argc) type = argv[++i];
        else if (a == "-lsh") use_lsh = true;
        else if (a == "-hypercube") use_hypercube = true;
        else if (a == "-ivfflat") use_ivfflat = true;
        else if (a == "-ivfpq") use_ivfpq = true;
        else if (a == "-kproj" && i+1 < argc) kproj = stoi(argv[++i]);
        else if (a == "-M" && i+1 < argc) M = stoi(argv[++i]);
        else if (a == "-probes" && i+1 < argc) probes = stoi(argv[++i]);
        else if (a == "-kclusters" && i+1 < argc) kclusters = stoi(argv[++i]);
        else if (a == "-nprobe" && i+1 < argc) nprobe = stoi(argv[++i]);
        else if (a == "-nbits" && i+1 < argc) nbits = stoi(argv[++i]);
        else if (a == "-range" && i+1 < argc) rangeSearch = (string(argv[++i]) == "true");
        //add any other flags needed
    }

    //Basic validation
    if (data_file.empty() || query_file.empty()) {
        cerr << "Usage example:\n"
             << "./search -d train-images.idx3-ubyte -q t10k-images.idx3-ubyte -type mnist -lsh -k 4 -L 5 -w 4.0 -N 1 -R 2000 -o output.txt\n";
        return 1;
    }


    bool success = false;
    if (use_lsh) {
        cout << "Launching LSH..." << endl;
        success = lsh_main(data_file, query_file, output_file, params, type, rangeSearch);
        cout << (success ? "LSH exited successfully\n" : "LSH exited abruptly\n");
    } else if (use_hypercube) {
         // -------------- Load dataset --------------
        vector<vector<double>> dataset;
        vector<vector<double>> queries;

        cout << "Launching Hypercube..." << endl;
        auto rawImages = read_mnist_images(data_file);
        auto rawQueries = read_mnist_images(query_file);
        dataset = convertToDouble(rawImages);
        queries = convertToDouble(rawQueries);
        success = hypercube_main(dataset, queries, output_file,kproj, params.w, M, probes, params.N, params.R, type, rangeSearch);
        cout << (success ? "Hypercube exited successfully\n" : "Hypercube exited abruptly\n");
    } else if (use_ivfflat) {
        cout << "Launching IVFFLAT..." << endl;
        //success = ivfflat_main(data_file, query_file, output_file,kclusters, nprobe, seed, params.N, params.R, type, rangeSearch);
        cout << (success ? "IVFFLAT exited successfully\n" : "IVFFLAT exited abruptly\n");
    } else if (use_ivfpq) {
        cout << "Launching IVFPQ..." << endl;
        //success = ivfpq_main(data_file, query_file, output_file,kclusters, nprobe, M, nbits, seed, params.N, params.R, type, rangeSearch);
        cout << (success ? "IVFPQ exited successfully\n" : "IVFPQ exited abruptly\n");
    } else {
        cout << "No algorithm selected. Use -lsh, -hypercube, -ivfflat or -ivfpq\n";
    }

    return success ? 0 : 1;
}
