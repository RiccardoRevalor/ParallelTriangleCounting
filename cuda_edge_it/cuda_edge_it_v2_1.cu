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
#include "../utils/utils.h"
#include "../utils/matrixMath.h"

#define DEBUG 0

#define MAX_SHARED_LIST_PER_EDGE_COMBINED 16

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

#define DEBUG 0

using namespace std;

struct Edge{
    int v0;
    int v1;
};


//1 BLOCK FOR EACH EDGE
__global__ void EdgeIteratorAlgorithmKernel(
    int numEdges,
    const int* d_adjacencyList_rowPtr,
    const int* d_adjacencyList_colIdx,
    const Edge *d_edgeVector,
    const int* d_ranks,
    int* d_countTriangles
) {
    // Un blocco intero processa un solo arco. L'indice dell'arco è l'indice del blocco.
    int edge_idx = blockIdx.x;

    // Dichiarazione della memoria condivisa per le due liste di adiacenza.
    extern __shared__ int shared_lists[];

    if (edge_idx < numEdges) {
        Edge currentEdge = d_edgeVector[edge_idx];
        int v0 = currentEdge.v0;
        int v1 = currentEdge.v1;

        int rank_v0 = d_ranks[v0];
        int rank_v1 = d_ranks[v1];

        // Orienta l'arco dal nodo con rank più basso a quello con rank più alto
        if (rank_v0 > rank_v1) {
            int temp_v = v0; v0 = v1; v1 = temp_v;
            int temp_r = rank_v0; rank_v0 = rank_v1; rank_v1 = temp_r;
        }

        int v0_start = d_adjacencyList_rowPtr[v0];
        int v0_end = d_adjacencyList_rowPtr[v0 + 1];
        int v1_start = d_adjacencyList_rowPtr[v1];
        int v1_end = d_adjacencyList_rowPtr[v1 + 1];

        int len0 = v0_end - v0_start;
        int len1 = v1_end - v1_start;

        // Se le liste sono abbastanza piccole da entrare nella shared memory
        if (len0 > 0 && len1 > 0 && (len0 + len1) <= MAX_SHARED_LIST_PER_EDGE_COMBINED) {
            
            // **Caricamento Cooperativo**
            // I thread del blocco caricano la prima lista
            for (int i = threadIdx.x; i < len0; i += blockDim.x) {
                shared_lists[i] = d_adjacencyList_colIdx[v0_start + i];
            }
            // I thread del blocco caricano la seconda lista, posizionandola dopo la prima
            for (int i = threadIdx.x; i < len1; i += blockDim.x) {
                shared_lists[len0 + i] = d_adjacencyList_colIdx[v1_start + i];
            }
            
            // Sincronizza tutti i thread del blocco per assicurarsi che il caricamento sia completo
            __syncthreads();

            // **Intersezione in Shared Memory**
            // Un solo thread (il primo) esegue l'algoritmo di merge
            if (threadIdx.x == 0) {
                int p0 = 0;
                int p1 = 0;
                while (p0 < len0 && p1 < len1) {
                    int neighbor_v0 = shared_lists[p0];
                    // La seconda lista inizia dopo len0 elementi
                    int neighbor_v1 = shared_lists[len0 + p1]; 

                    if (neighbor_v0 == neighbor_v1) {
                        if (d_ranks[neighbor_v0] > rank_v1) {
                            atomicAdd(d_countTriangles, 1);
                        }
                        p0++;
                        p1++;
                    } else if (neighbor_v0 < neighbor_v1) {
                        p0++;
                    } else {
                        p1++;
                    }
                }
            }
        } else { // Fallback su memoria globale per liste troppo grandi
            // Un solo thread esegue il calcolo per evitare accessi multipli
            if (threadIdx.x == 0) {
                int p0 = v0_start;
                int p1 = v1_start;
                while (p0 < v0_end && p1 < v1_end) {
                    int neighbor_v0 = d_adjacencyList_colIdx[p0];
                    int neighbor_v1 = d_adjacencyList_colIdx[p1];

                    if (neighbor_v0 == neighbor_v1) {
                        if (d_ranks[neighbor_v0] > rank_v1) {
                            atomicAdd(d_countTriangles, 1);
                        }
                        p0++;
                        p1++;
                    } else if (neighbor_v0 < neighbor_v1) {
                        p0++;
                    } else {
                        p1++;
                    }
                }
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
            // Ordina i nodi per garantire che (u,v) e (v,u) abbiano lo stesso hash.
            int first = min(e.v0, e.v1);
            int second = max(e.v0, e.v1);

            size_t h1 = hash<int>{}(first);
            size_t h2 = hash<int>{}(second);
            
            return h1 ^ (h2 << 1); 
        }
    };
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


unordered_set<Edge> createEdgeSet(map<int, vector<int>> &adjacencyVectors) {
    unordered_set<Edge> edgeSet;

    for (const auto &keyvaluepair : adjacencyVectors) {
        int u = keyvaluepair.first;
        for (int v : keyvaluepair.second) {
            edgeSet.insert({u, v});
        }
    }

    return edgeSet;
}



int main(void){

    std::string input;
    while(true) {
        cout << "insert file name: ";
        std::getline(std::cin, input);
        input = "../graph_file/" + input;
        
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

    convertToCRS(adjacencyVectors, h_adjacencyList_rowPtr, h_adjacencyList_colIdx, numNodes, true); //sort neighbors to speed up the algorithm and use merge-like approach in the kernel function


    vector<int> h_orderedList;
    createOrderedList(adjacencyVectors, h_orderedList);
    if (DEBUG) {

        std::cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : h_orderedList) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }


    //create edge set
    unordered_set<Edge> edgeSet = createEdgeSet(adjacencyVectors);

    //create vector of edges, since openmp works best with random access and vectors
    vector<Edge> h_edgeVector(edgeSet.begin(), edgeSet.end());
    int numEdges = edgeSet.size();


    //init ranks
    vector<int> h_ranks(numNodes + 1, 0); 
    for (int i = 0; i < h_orderedList.size(); ++i) {
        int nodeId = h_orderedList[i]; //node id
        int rank = i;                 //rank, based on the degree of the node, (i.e. the position in the ordered list, i.e. the number of neighbors it has)
        h_ranks[nodeId] = rank;
    }

    //ALLOC ON DEVICE
    int *d_adjacencyList_rowPtr, *d_adjacencyList_colIdx;
    Edge *d_edgeVector;
    int *d_ranks; 
    int* d_countTriangles;
    //mallocs
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_rowPtr, (numNodes +1) * sizeof(int) ));
    CUDA_CHECK(cudaMalloc(&d_adjacencyList_colIdx, h_adjacencyList_colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edgeVector, numEdges * sizeof(Edge)));
    CUDA_CHECK(cudaMalloc(&d_ranks, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_countTriangles, sizeof(int)));
    //copies
    CUDA_CHECK(cudaMemcpy(d_edgeVector, h_edgeVector.data(), numEdges * sizeof(Edge), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_rowPtr, h_adjacencyList_rowPtr.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adjacencyList_colIdx, h_adjacencyList_colIdx.data(), h_adjacencyList_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), numNodes * sizeof(int), cudaMemcpyHostToDevice));
    int h_countTriangles = 0;
    CUDA_CHECK(cudaMemcpy(d_countTriangles, &h_countTriangles, sizeof(int), cudaMemcpyHostToDevice));


    int blockSize = 24; //threads per block1
    int gridSize = numEdges;


    //start kernel function
    //number of bytes for shared memory
    size_t shmemBytes = blockSize * MAX_SHARED_LIST_PER_EDGE_COMBINED * sizeof(int);
    int maxSharedMemoryPerBlock;
    cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (shmemBytes > maxSharedMemoryPerBlock) {
        std::cout << "Errore: la shared memory richiesta (" << shmemBytes << " bytes) supera il massimo del dispositivo (" << maxSharedMemoryPerBlock << " bytes)." << std::endl;
        shmemBytes = maxSharedMemoryPerBlock; // O gestisci l'errore
    }
    auto startTime = chrono::high_resolution_clock::now();
    EdgeIteratorAlgorithmKernel<<<gridSize, blockSize, shmemBytes>>>(
        numEdges,
        d_adjacencyList_rowPtr,
        d_adjacencyList_colIdx,
        d_edgeVector,
        d_ranks,
        d_countTriangles
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    //copy back result
    CUDA_CHECK(cudaMemcpy(&h_countTriangles, d_countTriangles, sizeof(int), cudaMemcpyDeviceToHost));

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Time taken for edge iterator algorithm: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Triangles found by edge iterator algorithm: " << h_countTriangles << std::endl;
    std::cout << "Total number of edges: " << numEdges << std::endl
              << "Total number of nodes: " << adjacencyVectors.size() << std::endl;


    //FREE DEVICE
    CUDA_CHECK(cudaFree(d_adjacencyList_rowPtr));
    CUDA_CHECK(cudaFree(d_adjacencyList_colIdx));
    CUDA_CHECK(cudaFree(d_edgeVector));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_countTriangles));

    CUDA_CHECK(cudaDeviceReset());


    return 0;
}