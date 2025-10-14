#include <iostream>
#include <vector>
#include <string>
#include "../include/dataset.h"
#include "../include/hypercube.h"

using namespace std;

int main(int argc, char *argv[])
{
    // Command line arguments
    if (argc < 3)
    {
        cerr << "Usage: Command Line Error\n";
        return 1;
    }

    unordered_map<string, string> args = parseArgs(argc, argv);

    // -------------- Read command line args --------------
    bool useLSH = false, useHypercube = false, useIVFFLAT = false, useIVFPQ = false;
    int kproj = 0, M = 0, probes = 0, k = 0, L = 0;
    double w = 0.0;
    int kclusters = 0, nprobe = 0, seed = 0, nbits = 0;

    string inputFile = args.at("-d");
    string queryFile = args.at("-q");
    string outputFile = args.at("-o");
    int N = stoi(args.at("-N"));
    double R = stod(args.at("-R"));
    string type = args.at("-type");
    bool rangeSearch = (args.at("-range") == "true");

    if (args.find("-lsh") != args.end())
    {
        k = stoi(args.at("-k"));
        L = stoi(args.at("-L"));
        w = stod(args.at("-w"));
        useLSH = args.find("-lsh") != args.end();
    }
    else if (args.find("-hypercube") != args.end())
    {
        kproj = stoi(args.at("-kproj"));
        w = stod(args.at("-w"));
        M = stoi(args.at("-M"));
        probes = stoi(args.at("-probes"));
        useHypercube = args.find("-hypercube") != args.end();
    }
    else if (args.find("-ivfflat") != args.end())
    {
        kclusters = stoi(args.at("-kclusters"));
        nprobe = stoi(args.at("-nprobe"));
        seed = stoi(args.at("-seed"));
        useIVFFLAT = args.find("-ivfflat") != args.end();
    }
    else if (args.find("-ivfpq") != args.end())
    {
        kclusters = stoi(args.at("-kclusters"));
        nprobe = stoi(args.at("-nprobe"));
        M = stoi(args.at("-M"));
        nbits = stoi(args.at("-nbits"));
        seed = stoi(args.at("-seed"));
        useIVFPQ = args.find("-ivfpq") != args.end();
    }

    // -------------- Load dataset --------------
    vector<vector<double>> dataset;
    vector<vector<double>> queries;

    if (type == "MNIST")
    {
        auto rawImages = read_mnist_images(inputFile);
        auto rawQueries = read_mnist_images(queryFile);
        dataset = convertToDouble(rawImages);
        queries = convertToDouble(rawQueries);
    }
    else if (type == "SIFT")
    {
        auto rawVectors = readIFTData(inputFile);
        auto rawQueries = readIFTData(queryFile);
        dataset = convertToDouble(rawVectors);
        queries = convertToDouble(rawQueries);
    }
    else
    {
        cerr << "Unknown dataset type!\n";
        return 1;
    }

    cout << "Dataset and queries loaded successfully.\n";

    // -------------- Run Selected Algorithm --------------
    bool success = false;
    if (useLSH)
    {
        // success = LSH_main();
        if (success == 1)
            cout << "LSH exited successfully" << endl;
        else
            cout << "LSH exited abruptly" << endl;
    }
    else if (useHypercube)
    {
        success = hypercube_main(dataset, queries, outputFile, kproj, w, M, probes, N, R, type, rangeSearch);
        if (success == 1)
            cout << "Hypercube exited successfully" << endl;
        else
            cout << "Hypercube exited abruptly" << endl;
    }
    else if (useIVFFLAT)
    {
        // success = IVFFLAT_main(dataset, queries, outputFile, kproj, w, M, probes, N, R, type, rangeSearch);
        if (success == 1)
            cout << "IVFFLAT exited successfully" << endl;
        else
            cout << "IVFFLAT exited abruptly" << endl;
    }
    else if (useIVFPQ)
    {
        // success = IVFPQ_main(dataset, queries, outputFile, kproj, w, M, probes, N, R, type, rangeSearch);
        if (success == 1)
            cout << "IVFPQ exited successfully" << endl;
        else
            cout << "IVFPQ exited abruptly" << endl;
    }
    else
    {
        cout << "No algorithm selected.\n";
    }

    cout << "Program completed.\n";

    return 0;
}