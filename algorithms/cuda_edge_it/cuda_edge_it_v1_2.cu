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
inline void cuda_check(cudaError_t error_code, const char *file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

using namespace std;

struct Edge {
    int v0;
    int v1;
};

// Shared memory kernel version with naive nested loop
__global__ void EdgeIteratorAlgorithmKernelShared(
    int numEdges,
    const int* d_adjacencyList_rowPtr,
    const int* d_adjacencyList_colIdx,
    const Edge* d_edgeVector,
    const int* d_ranks,
    int* d_countTriangles,
    int startIdx,
    int endIdx
) {
    extern __shared__ int shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (endIdx - startIdx)) return;

    Edge edge = d_edgeVector[startIdx + idx];
    int v0 = edge.v0;
    int v1 = edge.v1;

    int rank_v0 = d_ranks[v0];
    int rank_v1 = d_ranks[v1];

    if (rank_v0 > rank_v1) {
        int tmp = v0; v0 = v1; v1 = tmp;
        int tmp_r = rank_v0; rank_v0 = rank_v1; rank_v1 = tmp_r;
    }

    int v0_start = d_adjacencyList_rowPtr[v0];
    int v0_end   = d_adjacencyList_rowPtr[v0 + 1];
    int v1_start = d_adjacencyList_rowPtr[v1];
    int v1_end   = d_adjacencyList_rowPtr[v1 + 1];

    int len0 = v0_end - v0_start;
    int len1 = v1_end - v1_start;

    int* neighbors_v0 = shared + threadIdx.x * (blockDim.x * 2);
    int* neighbors_v1 = neighbors_v0 + len0;

    for (int i = 0; i < len0; ++i) {
        neighbors_v0[i] = d_adjacencyList_colIdx[v0_start + i];
    }

    for (int i = 0; i < len1; ++i) {
        neighbors_v1[i] = d_adjacencyList_colIdx[v1_start + i];
    }

    __syncthreads(); // Ensure shared memory is fully loaded

    for (int i = 0; i < len0; ++i) {
        for (int j = 0; j < len1; ++j) {
            if (neighbors_v0[i] == neighbors_v1[j]) {
                int common = neighbors_v0[i];
                if (d_ranks[common] > rank_v1) {
                    atomicAdd(d_countTriangles, 1);
                }
            }
        }
    }
}

// Edge equality and hashing
bool operator==(const Edge& e1, const Edge& e2) {
    return (e1.v0 == e2.v0 && e1.v1 == e2.v1) || (e1.v0 == e2.v1 && e1.v1 == e2.v0);
}

namespace std {
    template <>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            int first = min(e.v0, e.v1);
            int second = max(e.v0, e.v1);
            return hash<int>()(first) ^ (hash<int>()(second) << 1);
        }
    };
}

// Sort nodes by degree descending
void createOrderedList(const map<int, vector<int>>& adjacencyVectors, vector<int>& orderedList) {
    vector<pair<int, int>> nodeDegreeSorted;
    for (const auto& [node, neighbors] : adjacencyVectors) {
        nodeDegreeSorted.emplace_back(node, neighbors.size());
    }

    sort(nodeDegreeSorted.begin(), nodeDegreeSorted.end(), [](auto& a, auto& b) {
        return a.second > b.second;
    });

    for (auto& [node, _] : nodeDegreeSorted) {
        orderedList.push_back(node);
    }
}

unordered_set<Edge> createEdgeSet(map<int, vector<int>>& adjacencyVectors) {
    unordered_set<Edge> edgeSet;
    for (auto& [u, neighbors] : adjacencyVectors) {
        for (int v : neighbors) {
            edgeSet.insert({u, v});
        }
    }
    return edgeSet;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_file> <BLOCK_SIZE> <DESIRED_LAUNCHES> <GPU_MODEL>" << endl;
        return 1;
    }

    string input;
    if (argv[1] == string("i")) {
        while (true) {
            cout << "insert file name: ";
            getline(cin, input);
            input = "../../graph_file/" + input;
            if (ifstream(input).is_open()) break;
            cout << input << " doesn't exist!" << endl;
        }
    } else {
        input = "../../graph_file/" + string(argv[1]);
    }

    int blockSize = stoi(argv[2]);
    int desiredLaunches = stoi(argv[3]);
    string gpuModel = argv[4];

    map<int, vector<int>> adjacencyVectors = populateAdjacencyVectors(input);
    vector<int> h_rowPtr, h_colIdx;
    int numNodes;
    convertToCRS(adjacencyVectors, h_rowPtr, h_colIdx, numNodes);

    vector<int> h_orderedList;
    createOrderedList(adjacencyVectors, h_orderedList);

    unordered_set<Edge> edgeSet = createEdgeSet(adjacencyVectors);
    vector<Edge> h_edgeVector(edgeSet.begin(), edgeSet.end());
    int numEdges = edgeSet.size();

    vector<int> h_ranks(numNodes + 1, 0);
    for (int i = 0; i < h_orderedList.size(); ++i) {
        h_ranks[h_orderedList[i]] = i;
    }

    int* d_rowPtr; int* d_colIdx; Edge* d_edgeVector;
    int* d_ranks; int* d_countTriangles;

    CUDA_CHECK(cudaMalloc(&d_rowPtr, (numNodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, h_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeVector, numEdges * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_countTriangles, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edgeVector, h_edgeVector.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice));

    int h_countTriangles = 0;
    CUDA_CHECK(cudaMemcpy(d_countTriangles, &h_countTriangles, sizeof(int), cudaMemcpyHostToDevice));

    int chunkSize = (numEdges + desiredLaunches - 1) / desiredLaunches;

    auto start = chrono::high_resolution_clock::now();
    for (int current_start = 0; current_start < numEdges; current_start += chunkSize) {
        int current_end = min(current_start + chunkSize, numEdges);
        int edges_in_chunk = current_end - current_start;

        if (edges_in_chunk == 0) continue;

        int gridSize = (edges_in_chunk + blockSize - 1) / blockSize;
        int sharedMemSize = 2 * blockSize * sizeof(int) * 32; // Each thread up to 32 neighbors for v0 and v1

        EdgeIteratorAlgorithmKernelShared<<<gridSize, blockSize, sharedMemSize>>>(
            edges_in_chunk,
            d_rowPtr,
            d_colIdx,
            d_edgeVector,
            d_ranks,
            d_countTriangles,
            current_start,
            current_end
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_countTriangles, d_countTriangles, sizeof(int), cudaMemcpyDeviceToHost));

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "-----------------------------------------------------------------" << endl;
    cout << "Time taken for shared memory edge iterator algorithm: " << duration.count() << " microseconds" << endl;
    cout << "Triangles found: " << h_countTriangles << endl;
    cout << "Edges: " << numEdges << " | Nodes: " << adjacencyVectors.size() << endl;

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_edgeVector));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_countTriangles));
    CUDA_CHECK(cudaDeviceReset());

    ofstream outFile;
    size_t pos = input.find_last_of(".");
    if (pos != string::npos) input = input.substr(0, pos);
    pos = input.find_last_of("/");
    if (pos != string::npos) input = input.substr(pos + 1);

    string outputFile = "../../cross_validation_output/cuda_edge_it_v1_2/" + input + "_" + gpuModel + ".csv";
    outFile.open(outputFile, ios::app);
    if (!outFile.is_open()) {
        cerr << "Error opening output file!" << endl;
        return -1;
    }

    outFile.seekp(0, ios::end);
    if (outFile.tellp() == 0) {
        outFile << "BLOCK_SIZE,DESIRED_LAUNCHES,GPU_MODEL,TOTAL_DURATION_US,TRIANGLES\n";
    }

    outFile << blockSize << "," << desiredLaunches << "," << gpuModel << "," << duration.count() << "," << h_countTriangles << "\n";
    outFile.close();

    return 0;
}
