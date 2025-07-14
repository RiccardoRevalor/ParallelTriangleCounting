#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>
#include <chrono>
#include "matrixMath.h"
#include "../utils/utils.h"

#define DEBUG 0

using namespace std;

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

vector<int> getNeighbors(const map<int, vector<int>> &adjacencyVectors, int node) {
    return adjacencyVectors.at(node);
}


void forwardAlgorithm(const vector<int> &orderedList, const map<int, vector<int>> &adjacencyVectors, int &countTriangles) {
    //A =  vector of sets, for each node we have a set
    vector<set<int>> A(adjacencyVectors.size());

    //maps of ranks of vertices based on their degree on the graph, so their position in the ordered list
    map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }


    for (const auto &s: orderedList){
        //get adjacency list of the current node
        vector<int> neighbors = getNeighbors(adjacencyVectors, s);
        for (int t : neighbors) {
            if (ranks.at(s) < ranks.at(t)) {
                //intersection of the two sets (A[s] and A[t])
                set<int> intersection;
                set_intersection(
                    A[s].begin(), A[s].end(),   
                    A[t].begin(), A[t].end(),   
                    inserter(intersection, intersection.begin())
                );
                //print triangles vertexes
                if (intersection.empty()){
                    if (DEBUG) cout << "It's not possibile to form a triangle with vertexes: " << s << " and " << t << endl;
                } else {

                    if (DEBUG) {
                        cout << "Triangle formed by vertexes: " << s << ", " << t << " and ";
                        for (const auto &v : intersection) {
                            cout << v << " ";
                            

                        }
                        cout << endl;
                    }


                    countTriangles += intersection.size();
                }

                //last step: update the set A[t]
                A[t].insert(s);

            }
        }
    }
}


int main() {

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    map<int, vector<int>> adjacencyVectors = populateAdjacencyVectors("../graph_file/graph1.g");

    /* ESEMPIO QUER
    // Aggiungi gli archi basandoti sull'immagine del grafo a destra
    addEdge(adjacencyMatrix, 0, 1);
    addEdge(adjacencyMatrix, 0, 2);
    addEdge(adjacencyMatrix, 0, 3);
    addEdge(adjacencyMatrix, 0, 6);
    addEdge(adjacencyMatrix, 1, 2);
    addEdge(adjacencyMatrix, 1, 4);
    addEdge(adjacencyMatrix, 1, 5);
    addEdge(adjacencyMatrix, 1, 7);
    addEdge(adjacencyMatrix, 2, 5);
    addEdge(adjacencyMatrix, 2, 8);
    addEdge(adjacencyMatrix, 3, 4);
    addEdge(adjacencyMatrix, 3, 5);
    addEdge(adjacencyMatrix, 3, 9);
    addEdge(adjacencyMatrix, 4, 5);
    addEdge(adjacencyMatrix, 4, 8);
    addEdge(adjacencyMatrix, 4, 10);
    addEdge(adjacencyMatrix, 5, 11);
    addEdge(adjacencyMatrix, 6, 7);
    addEdge(adjacencyMatrix, 6, 11);
    addEdge(adjacencyMatrix, 7, 8);
    addEdge(adjacencyMatrix, 8, 9);
    addEdge(adjacencyMatrix, 9, 10);
    addEdge(adjacencyMatrix, 10, 11);
    */



    //print ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyVectors, orderedList);
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
    forwardAlgorithm(orderedList, adjacencyVectors, countTriangles);
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Time taken for forward algorithm: " << duration.count() << " microseconds" << endl;

    //cout << "Tot Max Theoretical Triangles: " << getTotTriangles(adjacencyVectors) << endl;
    cout << "Triangles found by forward algorithm: " << countTriangles << endl;

    return 0;
}