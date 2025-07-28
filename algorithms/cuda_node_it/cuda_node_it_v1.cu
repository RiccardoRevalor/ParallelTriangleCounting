//NODE ITERATIR CUDA VERSION, WITH FOR LOOPS

#include <cuda_runtime.h>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <iostream>
#include "../../utils/utils.h"
#include "../../utils/matrixMath.h"

#define DEBUG 0
using namespace std;

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

__global__ void ForwardAlgorithmKernel(
    int numNodes,
    const int* d_adjacencyList_rowPtr,
    const int* d_adjacencyList_colIdx,
    const int* d_ranks,
    int* d_countTriangles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    int s = tid;
    int rank_s = d_ranks[s];

    int s_start = d_adjacencyList_rowPtr[s];
    int s_end = d_adjacencyList_rowPtr[s + 1];

    // Iterate over neighbors of s
    for (int i = s_start; i < s_end; ++i) {
        int t = d_adjacencyList_colIdx[i];
        int rank_t_ = d_ranks[t];

        if (rank_s >= rank_t_) continue;

        int t_start = d_adjacencyList_rowPtr[t];
        int t_end = d_adjacencyList_rowPtr[t + 1];

        //for each neighbor t, iterate over its neighbors
        for (int j = t_start; j < t_end; ++j) {
            int v = d_adjacencyList_colIdx[j];

            if (rank_t_ >= d_ranks[v]) continue;

            for (int k = s_start; k < s_end; ++k) {
                if (d_adjacencyList_colIdx[k] == v) {
                    atomicAdd(d_countTriangles, 1);
                    break;
                }
            }
        }
    }
}

// Funzione per stampare la matrice di adiacenza
void printMatrix(const std::vector<std::vector<int>>& matrix) {
    int numVertices = matrix.size();
    
    // Stampa l'intestazione delle colonne
    std::cout << "   ";
    for (int i = 0; i < numVertices; ++i) {
        std::cout << std::setw(3) << i;
    }
    std::cout << "\n";
    std::cout << "---";
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "---";
    }
    std::cout << "\n";

    // Stampa le righe della matrice
    for (int i = 0; i < numVertices; ++i) {
        std::cout << std::setw(2) << i << "|";
        for (int j = 0; j < numVertices; ++j) {
            std::cout << std::setw(3) << matrix[i][j];
        }
        std::cout << "\n";
    }
}


//Graphviz DOT format for printing the graph
void printDot(const std::vector<std::vector<int>>& matrix) {
    cout << "graph G {\n";

    int numVertices = matrix.size();
    for (int i = 0; i < numVertices; ++i) {
        for (int j = i + 1; j < numVertices; ++j) {
            if (matrix[i][j] == 1) {
                cout << "  " << i << " -- " << j << ";\n";
            }
        }
    }

    cout << "}\n";
}

void createOrderedList(const map<int, vector<int>> &adjacencyVectors, vector<int> &orderedList){
    //create a map to store the degree of each node, then sort it
    map<int, int> nodeDegree;
    for (const auto &keyvaluepair: adjacencyVectors) {
        int node = keyvaluepair.first;
        int degree = keyvaluepair.second.size();
        nodeDegree[node] = degree;
    }
    //sort map based on degree
    vector<pair<int, int>> nodeDegreeSorted(nodeDegree.begin(), nodeDegree.end());
    sort(nodeDegreeSorted.begin(), nodeDegreeSorted.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
        return a.second > b.second;
    });

    //just return the keys in the sorted order
    for (const auto &keyvaluepair : nodeDegreeSorted) {
        orderedList.emplace_back(keyvaluepair.first);
    }   
}



int main() {

    std::string input;
    while(true) {
        cout << "insert file name: ";
        std::getline(std::cin, input);
        input = "../../graph_file/" + input;
        
        // check whether file can be opened
        std::ifstream file(input);
        
        if (file.is_open())
            break;
        cout << input << " doesn't exist!" << endl; 
    }


    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    map<int, vector<int>> adjacencyVectors = populateAdjacencyVectors(input);
    vector<int> h_adjacencyList_rowPtr, h_adjacencyList_colIdx;
    int numNodes;

    convertToCRS(adjacencyVectors, h_adjacencyList_rowPtr, h_adjacencyList_colIdx, numNodes);


    vector<int> h_orderedList;
    createOrderedList(adjacencyVectors, h_orderedList);
    if (DEBUG) {

        std::cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : h_orderedList) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }

    //init ranks
    vector<int> h_ranks(numNodes + 1, 0); 
    for (int i = 0; i < h_orderedList.size(); ++i) {
        int nodeId = h_orderedList[i]; //node id
        int rank = i;                 //rank, based on the degree of the node, (i.e. the position in the ordered list, i.e. the number of neighbors it has)
        h_ranks[nodeId] = rank;
    }


    cout << "-----------------------------------------------------------------" << endl;
    //ALLOC ON DEVICE
    int *d_adjacencyList_rowPtr, *d_adjacencyList_colIdx;
    int *d_ranks; 
    int* d_countTriangles;
    //mallocs
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_rowPtr, (numNodes +1) * sizeof(int) ));
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_colIdx, h_adjacencyList_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_countTriangles, sizeof(int)));
    //copies
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_rowPtr, h_adjacencyList_rowPtr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_colIdx, h_adjacencyList_colIdx.data(), h_adjacencyList_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice));
    int h_countTriangles = 0;
    CUDA_CHECK(cudaMemcpy(d_countTriangles, &h_countTriangles, sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 8; //threads per block
    int gridSize = (numNodes + blockSize - 1) / blockSize; //blocks in grid

    auto startTime = chrono::high_resolution_clock::now();

    ForwardAlgorithmKernel<<<gridSize, blockSize>>>(numNodes, d_adjacencyList_rowPtr, d_adjacencyList_colIdx, d_ranks, d_countTriangles);

    CUDA_CHECK(cudaDeviceSynchronize());

    //copy back result
    CUDA_CHECK(cudaMemcpy(&h_countTriangles, d_countTriangles, sizeof(int), cudaMemcpyDeviceToHost));

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Time taken for forward algorithm: " << duration.count() << " microseconds" << endl;
    cout << "Triangles found by forward algorithm: " << h_countTriangles << endl;
    std::cout << "Total number of nodes: " << numNodes << std::endl;

    //FREE DEVICE
    CUDA_CHECK(cudaFree(d_adjacencyList_rowPtr));
    CUDA_CHECK(cudaFree(d_adjacencyList_colIdx));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_countTriangles));

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}