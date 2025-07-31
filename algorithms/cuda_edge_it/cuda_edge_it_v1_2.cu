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
// Assuming these utility headers exist in your project
#include "../../utils/utils.h" 
#include "../../utils/matrixMath.h"

// Macro for CUDA error checking
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

// Define DEBUG macro (0 for off, 1 for on)
#define DEBUG 0

using namespace std;

// Structure to represent an edge
struct Edge{
    int v0;
    int v1;
};

// Overload equality operator for Edge struct to handle undirected edges
bool operator==(const Edge &e1, const Edge &e2) {
    return (e1.v0 == e2.v0 && e1.v1 == e2.v1) || (e1.v0 == e2.v1 && e1.v1 == e2.v0);
}

// Custom hash function for Edge struct to be used with unordered_set
namespace std {
    template<>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            // Order the nodes to ensure (u,v) and (v,u) have the same hash.
            int first = min(e.v0, e.v1);
            int second = max(e.v0, e.v1);
            
            // Simple hash combining two integers
            size_t h1 = hash<int>{}(first);
            size_t h2 = hash<int>{}(second);
            
            return h1 ^ (h2 << 1); // XOR combination
        }
    };
}

/**
 * @brief CUDA Kernel for triangle counting using an edge iterator approach with shared memory.
 *
 * This kernel assigns one thread block to process a single edge.
 * Threads within the block cooperatively load the neighbor lists of the edge's endpoints
 * into shared memory. After synchronization, a single thread performs the intersection
 * using nested loops on the shared memory data. If the neighbor lists are too large
 * for shared memory, it falls back to direct global memory access.
 *
 * @param numEdges Total number of edges in the graph.
 * @param d_adjacencyList_rowPtr Pointer to device array storing row pointers for CSR format.
 * @param d_adjacencyList_colIdx Pointer to device array storing column indices for CSR format.
 * @param d_edgeVector Pointer to device array of Edge structs.
 * @param d_ranks Pointer to device array storing ranks of nodes (based on degree).
 * @param d_countTriangles Pointer to device atomic integer for global triangle count.
 * @param MAX_SHARED_LIST_PER_EDGE_COMBINED Maximum combined size (in number of integers)
 * of neighbor lists that can be loaded into shared memory for one edge.
 */
__global__ void EdgeIteratorAlgorithmKernel(
    int numEdges,
    const int* d_adjacencyList_rowPtr,
    const int* d_adjacencyList_colIdx,
    const Edge *d_edgeVector,
    const int* d_ranks,
    int* d_countTriangles,
    int MAX_SHARED_LIST_PER_EDGE_COMBINED // Passed from host
) {
    // Each block processes one edge. The edge index is the block index.
    int edge_idx = blockIdx.x;

    // Declare dynamic shared memory for the two adjacency lists.
    // This memory is shared by all threads within this block.
    extern __shared__ int shared_lists[];

    if (edge_idx < numEdges) {
        Edge currentEdge = d_edgeVector[edge_idx];
        int v0 = currentEdge.v0;
        int v1 = currentEdge.v1;

        int rank_v0 = d_ranks[v0];
        int rank_v1 = d_ranks[v1];

        // Orient the edge from the lower-ranked node to the higher-ranked node
        // This ensures each triangle is counted exactly once.
        if (rank_v0 > rank_v1) {
            int temp_v = v0; v0 = v1; v1 = temp_v;
            // No need to swap ranks, as d_ranks[common_neighbor] > rank_v1 uses the original rank
            // for v1, which is now correctly assigned to the higher-ranked node.
        }

        // Get start and end indices for neighbor lists from CSR format
        int v0_start = d_adjacencyList_rowPtr[v0];
        int v0_end = d_adjacencyList_rowPtr[v0 + 1];
        int v1_start = d_adjacencyList_rowPtr[v1];
        int v1_end = d_adjacencyList_rowPtr[v1 + 1];

        int len0 = v0_end - v0_start; // Length of v0's neighbor list
        int len1 = v1_end - v1_start; // Length of v1's neighbor list

        // Check if the combined lists are small enough to fit in shared memory
        // and if both lists actually have elements (to avoid unnecessary shared memory operations)
        if (len0 > 0 && len1 > 0 && (len0 + len1) * sizeof(int) <= MAX_SHARED_LIST_PER_EDGE_COMBINED) {
            
            // Cooperative loading of the first list into shared memory
            // Each thread loads elements in a strided fashion
            for (int i = threadIdx.x; i < len0; i += blockDim.x) {
                shared_lists[i] = d_adjacencyList_colIdx[v0_start + i];
            }
            // Cooperative loading of the second list into shared memory, offset after the first list
            for (int i = threadIdx.x; i < len1; i += blockDim.x) {
                shared_lists[len0 + i] = d_adjacencyList_colIdx[v1_start + i];
            }
            
            // Synchronize all threads in the block to ensure data is fully loaded
            __syncthreads();

            // Only one thread (e.g., thread 0) performs the nested loop intersection
            // using the data now in shared memory. This avoids redundant computation
            // and benefits from faster shared memory access.
            if (threadIdx.x == 0) {
                for (int i_sh = 0; i_sh < len0; ++i_sh) {
                    int neighbor_v0 = shared_lists[i_sh]; // Read from shared memory
                    for (int j_sh = 0; j_sh < len1; ++j_sh) {
                        // The second list starts after len0 elements in shared_lists
                        int neighbor_v1 = shared_lists[len0 + j_sh]; // Read from shared memory
                        
                        // If a common neighbor is found
                        if (neighbor_v0 == neighbor_v1) {
                            int common_neighbor = neighbor_v0;
                            // Apply the rank condition to count each triangle exactly once
                            if (d_ranks[common_neighbor] > rank_v1) {
                                atomicAdd(d_countTriangles, 1); // Thread-safe increment
                            }
                        }
                    }
                }
            }
        } else { // Fallback to global memory access for lists that are too large or empty
            // If shared memory is not used for this edge, only thread 0 of the block
            // performs the calculation directly from global memory.
            if (threadIdx.x == 0) {
                for (int i = v0_start; i < v0_end; ++i) {
                    int neighbor_v0 = d_adjacencyList_colIdx[i]; // Read from global memory
                    for (int j = v1_start; j < v1_end; ++j) {
                        int neighbor_v1 = d_adjacencyList_colIdx[j]; // Read from global memory
                        
                        if (neighbor_v0 == neighbor_v1) {
                            int common_neighbor = neighbor_v0;
                            if (d_ranks[common_neighbor] > rank_v1) {
                                atomicAdd(d_countTriangles, 1);
                            }
                        }
                    }
                }
            }
        }
    }
}


// Function to create an ordered list of nodes based on their degree (highest to lowest)
void createOrderedList(const map<int, vector<int>> &adjacencyVectors, vector<int> &orderedList){
    map<int, int> nodeDegree;
    for (const auto &keyvaluepair: adjacencyVectors) {
        int node = keyvaluepair.first;
        int degree = keyvaluepair.second.size();
        nodeDegree[node] = degree;
    }
    // Sort map based on degree in descending order
    vector<pair<int, int>> nodeDegreeSorted(nodeDegree.begin(), nodeDegree.end());
    sort(nodeDegreeSorted.begin(), nodeDegreeSorted.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
        return a.second > b.second;
    });

    // Populate the orderedList with node IDs in sorted degree order
    for (const auto &keyvaluepair : nodeDegreeSorted) {
        orderedList.emplace_back(keyvaluepair.first);
    }   
}

// Function to create an unordered set of unique edges from adjacency vectors
unordered_set<Edge> createEdgeSet(map<int, vector<int>> &adjacencyVectors) {
    unordered_set<Edge> edgeSet;
    for (const auto &keyvaluepair : adjacencyVectors) {
        int u = keyvaluepair.first;
        for (int v : keyvaluepair.second) {
            // Insert both (u,v) and (v,u) if graph is undirected, but hash ensures uniqueness
            edgeSet.insert({u, v}); 
        }
    }
    return edgeSet;
}


int main(int argc, char *argv[]) {

    if (argc != 5){
        cerr << "Usage: " << argv[0] << " <input_file> <BLOCK_SIZE> <MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES> <GPU_MODEL>" << endl;
        return 1;
    }

    std::string input;
    if (std::string(argv[1]) == "i") {
        while (true) {
            std::cout << "insert file name: ";
            std::getline(std::cin, input);
            input = "../../graph_file/" + input;

            std::ifstream file(input);
            if (file.is_open())
                break;
            std::cout << input << " doesn't exist!" << std::endl;
        }
    } else {
        input = "../../graph_file/" + std::string(argv[1]);
    }

    std::string gpuModel = argv[4];
    int blockSize = std::stoi(argv[2]);
    // MAX_SHARED_LIST_PER_EDGE_COMBINED is now in bytes
    int MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES = std::stoi(argv[3]); 


    // Populate adjacency vectors from file and convert to CSR format
    map<int, vector<int>> adjacencyVectors = populateAdjacencyVectors(input);
    vector<int> h_adjacencyList_rowPtr, h_adjacencyList_colIdx;
    int numNodes;
    // Note: No sorting here, as v1.2 uses nested loops. Sorting is for v2.x merge.
    convertToCRS(adjacencyVectors, h_adjacencyList_rowPtr, h_adjacencyList_colIdx, numNodes, false); 


    // Create ordered list of nodes based on degree
    vector<int> h_orderedList;
    createOrderedList(adjacencyVectors, h_orderedList);
    if (DEBUG) {
        std::cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : h_orderedList) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }

    // Create edge set and convert to vector for easier iteration on device
    unordered_set<Edge> edgeSet = createEdgeSet(adjacencyVectors);
    vector<Edge> h_edgeVector(edgeSet.begin(), edgeSet.end());
    int numEdges = edgeSet.size();

    // Initialize ranks array (mapping node ID to its rank in the ordered list)
    vector<int> h_ranks(numNodes + 1, 0); // numNodes + 1 to handle 0-indexed nodes up to maxNodeId
    for (int i = 0; i < h_orderedList.size(); ++i) {
        int nodeId = h_orderedList[i]; 
        int rank = i;                 
        h_ranks[nodeId] = rank;
    }

    // --- CUDA Device Memory Allocation and Data Transfer ---
    int *d_adjacencyList_rowPtr, *d_adjacencyList_colIdx;
    Edge *d_edgeVector;
    int *d_ranks; 
    int* d_countTriangles;

    // Allocate memory on the device
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_rowPtr, (numNodes + 1) * sizeof(int) ));
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_colIdx, h_adjacencyList_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeVector, numEdges * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numNodes * sizeof(int))); // Max node ID is numNodes-1, so numNodes size is sufficient if 0-indexed
    CUDA_CHECK(cudaMalloc(&d_countTriangles, sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_edgeVector, h_edgeVector.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_rowPtr, h_adjacencyList_rowPtr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_colIdx, h_adjacencyList_colIdx.data(), h_adjacencyList_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize triangle count on device to 0
    int h_countTriangles = 0;
    CUDA_CHECK(cudaMemcpy(d_countTriangles, &h_countTriangles, sizeof(int), cudaMemcpyHostToDevice));

    // Calculate grid size (one block per edge)
    int gridSize = numEdges;

    // Determine dynamic shared memory size for the kernel launch
    // This is the total bytes of shared memory required per block.
    // The kernel expects MAX_SHARED_LIST_PER_EDGE_COMBINED to be in bytes.
    size_t shmemBytes = MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES; 
    
    // Optional: Check if requested shared memory exceeds device limits
    int maxSharedMemoryPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (shmemBytes > maxSharedMemoryPerBlock) {
        // std::cerr << "Warning: Requested shared memory (" << shmemBytes << " bytes) exceeds device max (" << maxSharedMemoryPerBlock << " bytes)." << std::endl;
        // std::cerr << "Using device max shared memory. This might lead to unexpected behavior if MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES is too large." << std::endl;
        shmemBytes = maxSharedMemoryPerBlock; // Cap at max, but warn user
    }


    // --- Kernel Launch ---
    auto startTime = chrono::high_resolution_clock::now();
    EdgeIteratorAlgorithmKernel<<<gridSize, blockSize, shmemBytes>>>(
        numEdges,
        d_adjacencyList_rowPtr,
        d_adjacencyList_colIdx,
        d_edgeVector,
        d_ranks,
        d_countTriangles,
        MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES // Pass bytes to kernel
    );

    // Synchronize device to ensure all kernel computations are complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the final triangle count back from device to host
    CUDA_CHECK(cudaMemcpy(&h_countTriangles, d_countTriangles, sizeof(int), cudaMemcpyDeviceToHost));

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Time taken for CUDA edge iterator algorithm (v1.2): " << duration.count() << " microseconds" << std::endl;
    std::cout << "Triangles found by algorithm: " << h_countTriangles << std::endl;
    std::cout << "Total number of edges: " << numEdges << std::endl
              << "Total number of nodes: " << adjacencyVectors.size() << std::endl;


    // --- Free Device Memory ---
    CUDA_CHECK(cudaFree(d_adjacencyList_rowPtr));
    CUDA_CHECK(cudaFree(d_adjacencyList_colIdx));
    CUDA_CHECK(cudaFree(d_edgeVector));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_countTriangles));
    
    // Reset CUDA device (optional, good for cleanup in single-run apps)
    CUDA_CHECK(cudaDeviceReset());


    // --- Create Cross Validation Output File ---
    std::ofstream crossValidationFile;

    // Remove .g extension and path from input file name for output filename
    size_t pos = input.find_last_of(".");
    if (pos != std::string::npos) {
        input = input.substr(0, pos);
    }
    pos = input.find_last_of("/");
    if (pos != std::string::npos) {
        input = input.substr(pos + 1);
    }
    string outputFileName("../../cross_validation_output/cuda_edge_it_v1_2_corrected/" + input + "_" + gpuModel + ".csv");
    cout << "Output file name: " << outputFileName << endl;

    crossValidationFile.open(outputFileName, std::ios::app);
    if (!crossValidationFile.is_open()) { 
        std::cerr << "Error opening cross validation output file!" << std::endl;
        return -1;
    }

    // Write header if file is empty
    crossValidationFile.seekp(0, std::ios::end); 
    if (crossValidationFile.tellp() == 0) { 
        crossValidationFile << "BLOCK_SIZE,MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES,GPU_MODEL,TOTAL_DURATION_US,TRIANGLES\n";
    }
    
    // Write parameters and final time to the file
    crossValidationFile << blockSize << ","
                      << MAX_SHARED_LIST_PER_EDGE_COMBINED_BYTES << ","
                      << gpuModel << ","
                      << duration.count() << ","
                      << h_countTriangles << "\n";

    crossValidationFile.close();

    return 0;
}