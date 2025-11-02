# Approximate Nearest Neighbor Search Algorithms
Ομάδα 29
1115202100135 Παπαγεωργίου Πέτρος
1115201800224 Νικανδρού Νικόλας

## Τίτλος
**Κ23γ: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα - Χειμερινό εξάμηνο 2025-26**  
**1η Εργασία: Αναζήτηση διανυσμάτων στη C++**

Στην εργασία αναπτύξαμε τους παρακάτω αλγόριθμους:
- **LSH** (Locality Sensitive Hashing)
- **Hypercube** (Randomized Projections) 
- **IVFFlat** (Inverted File with Flat Quantization)
- **IVFPQ** (Inverted File with Product Quantization)

Στην εργασία χρησιμοποιήσαμε ευκλείδια απόσταση και στην ivfpq Asymmetric Distance,όπως στις διαφάνειες.

## Δομή

### Header Files (.h)
- **lsh.h** - LSH algorithm implementation and parameters
- **hypercube.h** - Hypercube algorithm implementation  
- **kmeans.h** - K-means clustering for IVF algorithms
- **ivfflat.h** - IVFFlat algorithm implementation
- **ivfpq.h** - IVFPQ algorithm implementation
- **utils.h** - Utility functions (distance calculations, timers, etc.)
- **data_reader.h** - MNIST and SIFT dataset reading functions

### Source Files (.cpp)
- **main.cpp** - Command-line interface and algorithm dispatcher
- **lsh.cpp** - LSH algorithm implementation
- **hypercube.cpp** - Hypercube algorithm implementation
- **kmeans.cpp** - K-means clustering implementation
- **ivfflat.cpp** - IVFFlat algorithm implementation  
- **ivfpq.cpp** - IVFPQ algorithm implementation
- **utils.cpp** - Utility functions implementation
- **data_reader.cpp** - Dataset reading implementation

### Dataset Files
- **MNIST**: `train-images.idx3-ubyte`, `t10k-images.idx3-ubyte`
- **SIFT**: `sift_learn.fvecs`, `sift_query.fvecs`

## Compilation
make

## Run examples with default values
### LSH
./bin/search -d dataset/train-images.idx3-ubyte -q dataset/t10k-images.idx3-ubyte -type mnist -lsh -k 4 -L 5 -w 4.0 -N 1 -seed 1 -R 2000 -range true -o output.txt
### HYPERCUBE
./bin/search -d ./dataset/train-images.idx3-ubyte -q ./dataset/t10k-images.idx3-ubyte -kproj 10 -w 40.0 -M 8000 -probes 510 -o ./output_hypercube.txt -N 2 -R 2000 -type MNIST -range true -seed 1 -hypercube
### IVFFLAT
./bin/search -d ./dataset/train-images.idx3-ubyte -q ./dataset/t10k-images.idx3-ubyte -kclusters 50 -nprobe 5 -o ./ouput_ivfflat.txt -N 1 -R 2000 -type MNIST -range true -seed 1 -ivfflat
### IVFPQ
./bin/search -d dataset/train-images.idx3-ubyte -q dataset/t10k-images.idx3-ubyte -type mnist -ivfpq -kclusters 30 -nprobe 5 -M 16 -nbits 8 -N 1 -o ivfpq_test1.txt
