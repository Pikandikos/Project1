#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <algorithm>

#include "kmeans.h"
#include "data_reader.h"
#include "utils.h"
#include "ivfpq.h"

using namespace std;
using namespace chrono;


bool ivfflat_main(const string& data_file, const string& query_file, const string& output_file,
                  int kclusters, int nprobe, int seed, int N, double R, 
                  const string& type, bool do_range) {
    
    //blabla
}
