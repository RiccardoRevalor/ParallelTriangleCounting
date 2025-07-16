#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>
#include <chrono>
#include <future>
#include <mutex>
#include "../utils/utils.h"
#include "../utils/matrixMath.h"

#define DEBUG 0

using namespace std;
mutex mtx0, mtx1;


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

void createOrderedList(const vector<vector<int>> &adjacencyMatrix, vector<int> &orderedList){
    //create a map to store the degree of each node, then sort it
    map<int, int> nodeDegree;
    for (int i = 0; i < adjacencyMatrix.size(); ++i) {
        for (auto element : adjacencyMatrix[i]) {
            if (element == 1) {
                nodeDegree[i]++;
            }
        }
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

vector<int> getNeighbors(const vector<vector<int>> &adjacencyMatrix, int node) {
    vector<int> neighbors;
    for (int i = 0; i < adjacencyMatrix[node].size(); ++i) {
        if (adjacencyMatrix[node][i] == 1) {
            neighbors.emplace_back(i);
        }
    }
    return neighbors;
}


void worker(const int s, const int tStart, const int tEnd, const vector<int> &neighbors, vector<set<int>> &A, const map<int, int> &ranks, int &countTriangles) {
    for (int t = tStart; t < tEnd; ++t) {
        if (ranks.at(s) < ranks.at(neighbors[t])) {
            //intersection of the two sets (A[s] and A[t])
            set<int> intersection;
            {
                lock_guard<mutex> lock(mtx1);
                set_intersection(
                A[s].begin(), A[s].end(),
                A[neighbors[t]].begin(), A[neighbors[t]].end(),
                inserter(intersection, intersection.begin())
            );
            }
            //print triangles vertexes
            {
                lock_guard<mutex> lock(mtx0);
                if (intersection.empty()) {
                    if (DEBUG) cout << "It's not possibile to form a triangle with vertexes: " << s << " and " << neighbors[t] << endl;
                } else {
                    if (DEBUG) cout << "Triangle formed by vertexes: " << s << ", " << neighbors[t] << " and ";
                    for (const auto &v : intersection) {
                        if (DEBUG) cout << v << " ";
                        ++countTriangles;

                    }
                    if (DEBUG) cout << endl;
                }
            }

            //last step: update the set A[t]
            {
                lock_guard<mutex> lock(mtx1);
                A[neighbors[t]].insert(s);
            }
        }
    }
}

void forwardAlgorithm(const vector<int> &orderedList, const vector<vector<int>> &adjacencyMatrix, int &countTriangles, int numThreads = 1) {
    //A =  vector of sets, for each node we have a set
    vector<set<int>> A(adjacencyMatrix.size());

    //maps of ranks of vertices based on their degree on the graph, so their position in the ordered list
    map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }


    for (const auto &s: orderedList){
        //get adjacency list of the current node
        vector<int> neighbors = getNeighbors(adjacencyMatrix, s);

        const int chunkSize = (neighbors.size() + numThreads - 1) / numThreads; 
        vector<future<void>> futures;

        for (int i = 0; i < numThreads; ++i) {
            int tStart = i * chunkSize;
            int tEnd = min(tStart + chunkSize, static_cast<int>(neighbors.size()));
            if (tStart < tEnd) {
                futures.emplace_back(async(launch::async, worker, s, tStart, tEnd, ref(neighbors), ref(A), ref(ranks), ref(countTriangles)));
            }
        }

        for (auto &fut : futures) {
            fut.get(); // Wait for all threads to finish before proceeding to next node
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
    // Il grafo ha 12 nodi, numerati da 0 a 11
    const int NUM_VERTICES = 100; //12;

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0

    vector<vector<int>> adjacencyMatrix = populateAdjacencyMatrix("../graph_file/graph_100.g");

    // Stampa la matrice risultante
    if (DEBUG) {
        std::cout << "Matrice di Adiacenza per il grafo:\n\n";
        printMatrix(adjacencyMatrix);
        
        //print with Graphviz DOT format
        printDot(adjacencyMatrix);
    }


    //print ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyMatrix, orderedList);
    if (DEBUG) {
        cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : orderedList) {
            cout << node << " ";
        }
        cout << "\n";
    }


    cout << "-----------------------------------------------------------------" << endl;
    int countTriangles = 0;
    auto startTime = chrono::high_resolution_clock::now();
    // Run the forward algorithm
    forwardAlgorithm(orderedList, adjacencyMatrix, countTriangles);
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Time taken for forward algorithm: " << duration.count() << " microseconds" << endl;

    cout << "Tot Max Theoretical Triangles: " << getTotTriangles(adjacencyMatrix) << endl;
    cout << "Triangles found by forward algorithm: " << countTriangles << endl;

    return 0;
}