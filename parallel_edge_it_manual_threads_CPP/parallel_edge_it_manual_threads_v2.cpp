#include <iostream>
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
#include "../utils/utils.h"
#include "../utils/matrixMath.h"

#define NUM_THREADS 16
#define DEBUG 0

using namespace std;

struct Edge{
    int v0;
    int v1;
};

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
    std::cout << "graph G {\n";

    int numVertices = matrix.size();
    for (int i = 0; i < numVertices; ++i) {
        for (int j = i + 1; j < numVertices; ++j) {
            if (matrix[i][j] == 1) {
                std::cout << "  " << i << " -- " << j << ";\n";
            }
        }
    }

    std::cout << "}\n";
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

vector<int> getNeighbors(const map<int, vector<int>> &adjacencyVectors, int node) {
    return adjacencyVectors.at(node);
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



void EdgeIteratorAlgorithm(const vector<int> &orderedList, const map<int, vector<int>> &adjacencyVectors, const vector<Edge> &edgeVector, atomic<int> &countTriangles, int start, int end) {
    //maps of ranks of vertices based on their degree on the graph, so their position in the ordered list
    map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }

    //Iterate through edge Set
    for (int i = start; i < end; ++i) {
        const auto &edge = edgeVector[i];
        //find row in adjacency matrix for v0 and v1
        int v0 = edge.v0;
        int v1 = edge.v1;
        //if (ranks.at(v0) >= ranks.at(v1)) continue;
        if (ranks.at(v0) > ranks.at(v1)) {
            swap(v0, v1); //make sure v0 is always the one with lower rank
        }
        vector<int> neighborsV0 = getNeighbors(adjacencyVectors, v0);
        vector<int> neighborsV1 = getNeighbors(adjacencyVectors, v1);
        //find intersection of neighborsV0 and neighborsV1
        set<int> intersection;
        set_intersection(
            neighborsV0.begin(), neighborsV0.end(),
            neighborsV1.begin(), neighborsV1.end(),
            inserter(intersection, intersection.begin())
        );
        //count triangles, i.e. the size of the intersection
        if (intersection.empty()){
            //cout << "It's not possibile to form a triangle with vertexes: " << v0 << " and " << v1 << endl;
        } else {
            for (const auto &v : intersection) {
                if (ranks.at(v) > ranks.at(v1)) {
                    // Ora ogni triangolo viene contato esattamente una volta
                    ++countTriangles;
                    //cout << "Triangle found: (" << v0 << ", " << v1 << ", " << v << ")" << endl;
                }

            }
        }
        
    }
}

float getTotTriangles(const vector<vector<int>> adjacencyMatrix) {
    vector<vector<int>> A3 = cubeAdjacencyMatrix(adjacencyMatrix);

    const float factor = (float)1/ (float)6;

    // compute tractiant
    int traciant = 0;
    for (int i = 0; i < A3.size(); i++) {
        for (int j = 0; j < A3[i].size(); j++) {
            if (i == j) {
                traciant += A3[i][j];
            }
        }
    }

    return factor*traciant;
}

int main() {

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

    // Stampa la matrice risultante
    if (DEBUG) {
        std::cout << "Matrice di Adiacenza per il grafo:\n\n";
        // printMatrix(adjacencyVectors);
    
        // print with Graphviz DOT format
        // printDot(adjacencyVectors);
    }

    //print ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyVectors, orderedList);
    if (DEBUG) {

        std::cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : orderedList) {
            std::cout << node << " ";
        }
        std::cout << "\n";
    }

    //create edge set
    unordered_set<Edge> edgeSet = createEdgeSet(adjacencyVectors);

    //create vector of edges, since openmp works best with random access and vectors
    vector<Edge> edgeVector(edgeSet.begin(), edgeSet.end());

    //threads creation
    int totEdges = edgeVector.size();
    int chunkSize = (totEdges + NUM_THREADS - 1) / NUM_THREADS; // Round up division
    vector<thread> threads;
    atomic<int> countTriangles = 0;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    auto startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * chunkSize;
        int end = min(start + chunkSize, totEdges);
        threads.emplace_back(EdgeIteratorAlgorithm, ref(orderedList), ref(adjacencyVectors), ref(edgeVector), ref(countTriangles), start, end);
    }

    for (auto &t : threads) {
        t.join();
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    std::cout << "Time taken for edge iterator algorithm: " << duration.count() << " microseconds" << std::endl;

    //std::cout << "Tot Max Theoretical Triangles: " << getTotTriangles(adjacencyMatrix) << std::endl;
    std::cout << "Triangles found by edge iterator algorithm: " << countTriangles << std::endl;
    std::cout << "Total number of edges: " << edgeSet.size() << std::endl
              << "Total number of nodes: " << adjacencyVectors.size() << std::endl;

    return 0;
}