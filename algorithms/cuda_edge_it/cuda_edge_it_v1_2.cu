#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
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

using namespace std;

struct Edge{
    int v0;
    int v1;
};

// 1 THREAD MANAGES JUST 1 EDGE
// v1_2 kernel: Like v1_1 but with shared memory optimization
__global__ void EdgeIteratorAlgorithmKernel(
    int numEdges,
    const int* d_adjacencyList_rowPtr,
    const int* d_adjacencyList_colIdx,
    const Edge *d_edgeVector,
    const int* d_ranks,
    int* d_countTriangles,
    int MAX_SHARED_LIST_PER_EDGE_COMBINED = 32
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    Edge currentEdge = d_edgeVector[idx];
    int v0 = currentEdge.v0;
    int v1 = currentEdge.v1;

    int rank_v0 = d_ranks[v0];
    int rank_v1 = d_ranks[v1];

    if (rank_v0 > rank_v1) {
        int tmp = v0; v0 = v1; v1 = tmp;
        int tmpRank = rank_v0; rank_v0 = rank_v1; rank_v1 = tmpRank;
    }

    int v0_start = d_adjacencyList_rowPtr[v0];
    int v0_end = d_adjacencyList_rowPtr[v0 + 1];
    int v1_start = d_adjacencyList_rowPtr[v1];
    int v1_end = d_adjacencyList_rowPtr[v1 + 1];

    int len0 = v0_end - v0_start;
    int len1 = v1_end - v1_start;

    // Use shared memory when combined adjacency list lengths are small enough
    if (len0 > 0 && len1 > 0 && (len0 + len1) <= MAX_SHARED_LIST_PER_EDGE_COMBINED) {
        extern __shared__ int shared_memory[];

        int thread_shmem_offset = threadIdx.x * MAX_SHARED_LIST_PER_EDGE_COMBINED;

        int *shared_list0 = shared_memory + thread_shmem_offset;
        int *shared_list1 = shared_memory + thread_shmem_offset + len0;

        // Load adjacency lists into shared memory
        for (int i = 0; i < len0; ++i) {
            shared_list0[i] = d_adjacencyList_colIdx[v0_start + i];
        }
        for (int i = 0; i < len1; ++i) {
            shared_list1[i] = d_adjacencyList_colIdx[v1_start + i];
        }
        __syncthreads();

        // Merge-like intersection using shared memory
        int p0 = 0, p1 = 0;
        while (p0 < len0 && p1 < len1) {
            int neigh0 = shared_list0[p0];
            int neigh1 = shared_list1[p1];

            if (neigh0 == neigh1) {
                if (d_ranks[neigh0] > rank_v1) {
                    atomicAdd(d_countTriangles, 1);
                }
                p0++; p1++;
            } else if (neigh0 < neigh1) {
                p0++;
            } else {
                p1++;
            }
        }
    } else {
        // Fallback: intersection directly in global memory
        int p0 = v0_start, p1 = v1_start;
        while (p0 < v0_end && p1 < v1_end) {
            int neigh0 = d_adjacencyList_colIdx[p0];
            int neigh1 = d_adjacencyList_colIdx[p1];
            if (neigh0 == neigh1) {
                if (d_ranks[neigh0] > rank_v1) {
                    atomicAdd(d_countTriangles, 1);
                }
                p0++; p1++;
            } else if (neigh0 < neigh1) {
                p0++;
            } else {
                p1++;
            }
        }
    }
}

bool operator==(const Edge &e1, const Edge &e2) {
    return (e1.v0 == e2.v0 && e1.v1 == e2.v1) || (e1.v0 == e2.v1 && e1.v1 == e2.v0);
}

namespace std {
    template<>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            int first = min(e.v0, e.v1);
            int second = max(e.v0, e.v1);
            size_t h1 = hash<int>{}(first);
            size_t h2 = hash<int>{}(second);
            return h1 ^ (h2 << 1);
        }
    };
}

void createOrderedList(const map<int, vector<int>> &adjacencyVectors, vector<int> &orderedList){
    map<int, int> nodeDegree;
    for (const auto &kv : adjacencyVectors) {
        nodeDegree[kv.first] = kv.second.size();
    }
    vector<pair<int,int>> nodeDegreeSorted(nodeDegree.begin(), nodeDegree.end());
    sort(nodeDegreeSorted.begin(), nodeDegreeSorted.end(), [](auto &a, auto &b){
        return a.second > b.second;
    });
    for (auto &kv : nodeDegreeSorted) {
        orderedList.push_back(kv.first);
    }
}

unordered_set<Edge> createEdgeSet(map<int, vector<int>> &adjacencyVectors) {
    unordered_set<Edge> edgeSet;
    for (const auto &kv : adjacencyVectors) {
        int u = kv.first;
        for (int v : kv.second) {
            edgeSet.insert({u, v});
        }
    }
    return edgeSet;
}

int main(int argc, char *argv[]) {

    if (argc != 5){
        cerr << "Usage: " << argv[0] << " <input_file> <BLOCK_SIZE> <MAX_SHARED_LIST_PER_EDGE_COMBINED> <GPU_MODEL>" << endl;
        return 1;
    }

    std::string input;
    if (argv[1] == std::string("i")) {
        while (true) {
            std::cout << "insert file name: ";
            std::getline(std::cin, input);
            input = "../../graph_file/" + input;
            std::ifstream file(input);
            if (file.is_open()) break;
            std::cout << input << " doesn't exist!" << std::endl;
        }
    } else {
        input = "../../graph_file/" + std::string(argv[1]);
    }

    std::string gpuModel = argv[4];
    int blockSize = std::stoi(argv[2]);
    int MAX_SHARED_LIST_PER_EDGE_COMBINED = std::stoi(argv[3]);

    map<int, vector<int>> adjacencyVectors = populateAdjacencyVectors(input);
    vector<int> h_adjacencyList_rowPtr, h_adjacencyList_colIdx;
    int numNodes;

    convertToCRS(adjacencyVectors, h_adjacencyList_rowPtr, h_adjacencyList_colIdx, numNodes, true);

    vector<int> h_orderedList;
    createOrderedList(adjacencyVectors, h_orderedList);

    if (DEBUG) {
        std::cout << "Ordered list of nodes based on degree:\n";
        for (auto node : h_orderedList) std::cout << node << " ";
        std::cout << std::endl;
    }

    unordered_set<Edge> edgeSet = createEdgeSet(adjacencyVectors);
    vector<Edge> h_edgeVector(edgeSet.begin(), edgeSet.end());
    int numEdges = edgeSet.size();

    vector<int> h_ranks(numNodes + 1, 0);
    for (int i = 0; i < (int)h_orderedList.size(); i++) {
        h_ranks[h_orderedList[i]] = i;
    }

    // Device allocation
    int *d_adjacencyList_rowPtr, *d_adjacencyList_colIdx;
    Edge *d_edgeVector;
    int *d_ranks;
    int *d_countTriangles;

    CUDA_CHECK(cudaMalloc(&d_adjacencyList_rowPtr, (numNodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_colIdx, h_adjacencyList_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeVector, numEdges * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_ranks, (numNodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_countTriangles, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_adjacencyList_rowPtr, h_adjacencyList_rowPtr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_colIdx, h_adjacencyList_colIdx.data(), h_adjacencyList_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edgeVector, h_edgeVector.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));

    int h_countTriangles = 0;
    CUDA_CHECK(cudaMemcpy(d_countTriangles, &h_countTriangles, sizeof(int), cudaMemcpyHostToDevice));

    int gridSize = (numEdges + blockSize - 1) / blockSize;

    size_t shmemBytes = blockSize * MAX_SHARED_LIST_PER_EDGE_COMBINED * sizeof(int);

    auto startTime = chrono::high_resolution_clock::now();

    EdgeIteratorAlgorithmKernel<<<gridSize, blockSize, shmemBytes>>>(
        numEdges,
        d_adjacencyList_rowPtr,
        d_adjacencyList_colIdx,
        d_edgeVector,
        d_ranks,
        d_countTriangles,
        MAX_SHARED_LIST_PER_EDGE_COMBINED
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_countTriangles, d_countTriangles, sizeof(int), cudaMemcpyDeviceToHost));

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

    cout << "-----------------------------------------------------------------" << endl;
    cout << "Time taken for edge iterator algorithm v1_2: " << duration.count() << " microseconds" << endl;
    cout << "Triangles found by edge iterator algorithm v1_2: " << h_countTriangles << endl;
    cout << "Total number of edges: " << numEdges << endl
         << "Total number of nodes: " << adjacencyVectors.size() << endl;

    CUDA_CHECK(cudaFree(d_adjacencyList_rowPtr));
    CUDA_CHECK(cudaFree(d_adjacencyList_colIdx));
    CUDA_CHECK(cudaFree(d_edgeVector));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_countTriangles));

    CUDA_CHECK(cudaDeviceReset());

    // Cross-validation output file
    std::ofstream crossValidationFile;

    size_t pos = input.find_last_of(".");
    if (pos != std::string::npos) input = input.substr(0, pos);
    pos = input.find_last_of("/");
    if (pos != std::string::npos) input = input.substr(pos + 1);

    string outputFileName = "../../cross_validation_output/cuda_edge_it_v1_2/" + input + "_" + gpuModel + ".csv";
    cout << "Output file name: " << outputFileName << endl;

    crossValidationFile.open(outputFileName, std::ios::app);
    if (!crossValidationFile.is_open()) {
        cerr << "Error opening cross validation output file!" << endl;
        return -1;
    }

    crossValidationFile.seekp(0, ios::end);
    if (crossValidationFile.tellp() == 0) {
        crossValidationFile << "BLOCK_SIZE,MAX_SHARED_LIST_PER_EDGE_COMBINED,GPU_MODEL,TOTAL_DURATION_US,TRIANGLES\n";
    }
    crossValidationFile << blockSize << ","
                        << MAX_SHARED_LIST_PER_EDGE_COMBINED << ","
                        << gpuModel << ","
                        << duration.count() << ","
                        << h_countTriangles << "\n";

    crossValidationFile.close();

    return 0;
}
