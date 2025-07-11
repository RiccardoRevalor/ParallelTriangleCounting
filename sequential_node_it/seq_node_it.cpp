#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>
#include <chrono>
#include "matrixMath.h"

using namespace std;

void addEdge(std::vector<std::vector<int>>& matrix, int u, int v) {
    
    matrix[u][v] = 1;
    matrix[v][u] = 1;
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


void forwardAlgorithm(const vector<int> &orderedList, const vector<vector<int>> &adjacencyMatrix, int &countTriangles) {
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
                    cout << "It's not possibile to form a triangle with vertexes: " << s << " and " << t << endl;
                } else {
                    cout << "Triangle formed by vertexes: " << s << ", " << t << " and ";
                    for (const auto &v : intersection) {
                        cout << v << " ";
                        ++countTriangles;

                    }
                    cout << endl;
                }

                //last step: update the set A[t]
                A[t].insert(s);

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
    // Il grafo ha 12 nodi, numerati da 0 a 11
    const int NUM_VERTICES = 100; //12;

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    vector<vector<int>> adjacencyMatrix(NUM_VERTICES, vector<int>(NUM_VERTICES, 0));


    addEdge(adjacencyMatrix, 0, 5);
addEdge(adjacencyMatrix, 0, 6);
addEdge(adjacencyMatrix, 0, 7);
addEdge(adjacencyMatrix, 0, 8);
addEdge(adjacencyMatrix, 0, 12);
addEdge(adjacencyMatrix, 0, 20);
addEdge(adjacencyMatrix, 0, 26);
addEdge(adjacencyMatrix, 0, 27);
addEdge(adjacencyMatrix, 0, 31);
addEdge(adjacencyMatrix, 0, 54);
addEdge(adjacencyMatrix, 0, 60);
addEdge(adjacencyMatrix, 0, 72);
addEdge(adjacencyMatrix, 0, 75);
addEdge(adjacencyMatrix, 0, 84);
addEdge(adjacencyMatrix, 1, 4);
addEdge(adjacencyMatrix, 1, 5);
addEdge(adjacencyMatrix, 1, 6);
addEdge(adjacencyMatrix, 1, 16);
addEdge(adjacencyMatrix, 1, 26);
addEdge(adjacencyMatrix, 1, 27);
addEdge(adjacencyMatrix, 1, 30);
addEdge(adjacencyMatrix, 1, 37);
addEdge(adjacencyMatrix, 1, 43);
addEdge(adjacencyMatrix, 1, 50);
addEdge(adjacencyMatrix, 1, 54);
addEdge(adjacencyMatrix, 1, 65);
addEdge(adjacencyMatrix, 1, 66);
addEdge(adjacencyMatrix, 1, 70);
addEdge(adjacencyMatrix, 1, 74);
addEdge(adjacencyMatrix, 1, 75);
addEdge(adjacencyMatrix, 1, 77);
addEdge(adjacencyMatrix, 1, 88);
addEdge(adjacencyMatrix, 1, 90);
addEdge(adjacencyMatrix, 1, 96);
addEdge(adjacencyMatrix, 1, 98);
addEdge(adjacencyMatrix, 1, 99);
addEdge(adjacencyMatrix, 2, 5);
addEdge(adjacencyMatrix, 2, 8);
addEdge(adjacencyMatrix, 2, 13);
addEdge(adjacencyMatrix, 2, 15);
addEdge(adjacencyMatrix, 2, 17);
addEdge(adjacencyMatrix, 2, 21);
addEdge(adjacencyMatrix, 2, 28);
addEdge(adjacencyMatrix, 2, 30);
addEdge(adjacencyMatrix, 2, 34);
addEdge(adjacencyMatrix, 2, 52);
addEdge(adjacencyMatrix, 2, 54);
addEdge(adjacencyMatrix, 2, 58);
addEdge(adjacencyMatrix, 2, 65);
addEdge(adjacencyMatrix, 2, 76);
addEdge(adjacencyMatrix, 2, 80);
addEdge(adjacencyMatrix, 3, 29);
addEdge(adjacencyMatrix, 3, 45);
addEdge(adjacencyMatrix, 3, 49);
addEdge(adjacencyMatrix, 3, 50);
addEdge(adjacencyMatrix, 3, 74);
addEdge(adjacencyMatrix, 3, 77);
addEdge(adjacencyMatrix, 3, 82);
addEdge(adjacencyMatrix, 3, 85);
addEdge(adjacencyMatrix, 3, 98);
addEdge(adjacencyMatrix, 4, 6);
addEdge(adjacencyMatrix, 4, 15);
addEdge(adjacencyMatrix, 4, 21);
addEdge(adjacencyMatrix, 4, 43);
addEdge(adjacencyMatrix, 4, 74);
addEdge(adjacencyMatrix, 4, 81);
addEdge(adjacencyMatrix, 4, 91);
addEdge(adjacencyMatrix, 4, 98);
addEdge(adjacencyMatrix, 5, 8);
addEdge(adjacencyMatrix, 5, 14);
addEdge(adjacencyMatrix, 5, 20);
addEdge(adjacencyMatrix, 5, 21);
addEdge(adjacencyMatrix, 5, 32);
addEdge(adjacencyMatrix, 5, 42);
addEdge(adjacencyMatrix, 5, 43);
addEdge(adjacencyMatrix, 5, 77);
addEdge(adjacencyMatrix, 5, 83);
addEdge(adjacencyMatrix, 5, 91);
addEdge(adjacencyMatrix, 6, 11);
addEdge(adjacencyMatrix, 6, 45);
addEdge(adjacencyMatrix, 6, 46);
addEdge(adjacencyMatrix, 6, 50);
addEdge(adjacencyMatrix, 6, 65);
addEdge(adjacencyMatrix, 6, 73);
addEdge(adjacencyMatrix, 6, 83);
addEdge(adjacencyMatrix, 6, 98);
addEdge(adjacencyMatrix, 6, 99);
addEdge(adjacencyMatrix, 7, 20);
addEdge(adjacencyMatrix, 7, 38);
addEdge(adjacencyMatrix, 7, 43);
addEdge(adjacencyMatrix, 7, 50);
addEdge(adjacencyMatrix, 7, 66);
addEdge(adjacencyMatrix, 7, 77);
addEdge(adjacencyMatrix, 7, 84);
addEdge(adjacencyMatrix, 7, 85);
addEdge(adjacencyMatrix, 8, 16);
addEdge(adjacencyMatrix, 8, 17);
addEdge(adjacencyMatrix, 8, 37);
addEdge(adjacencyMatrix, 8, 42);
addEdge(adjacencyMatrix, 8, 44);
addEdge(adjacencyMatrix, 8, 48);
addEdge(adjacencyMatrix, 8, 52);
addEdge(adjacencyMatrix, 8, 61);
addEdge(adjacencyMatrix, 8, 67);
addEdge(adjacencyMatrix, 8, 94);
addEdge(adjacencyMatrix, 9, 14);
addEdge(adjacencyMatrix, 9, 21);
addEdge(adjacencyMatrix, 9, 22);
addEdge(adjacencyMatrix, 9, 24);
addEdge(adjacencyMatrix, 9, 33);
addEdge(adjacencyMatrix, 9, 48);
addEdge(adjacencyMatrix, 9, 55);
addEdge(adjacencyMatrix, 9, 88);
addEdge(adjacencyMatrix, 9, 91);
addEdge(adjacencyMatrix, 10, 20);
addEdge(adjacencyMatrix, 10, 21);
addEdge(adjacencyMatrix, 10, 30);
addEdge(adjacencyMatrix, 10, 31);
addEdge(adjacencyMatrix, 10, 33);
addEdge(adjacencyMatrix, 10, 37);
addEdge(adjacencyMatrix, 10, 38);
addEdge(adjacencyMatrix, 10, 41);
addEdge(adjacencyMatrix, 10, 43);
addEdge(adjacencyMatrix, 10, 51);
addEdge(adjacencyMatrix, 10, 67);
addEdge(adjacencyMatrix, 10, 76);
addEdge(adjacencyMatrix, 10, 85);
addEdge(adjacencyMatrix, 10, 89);
addEdge(adjacencyMatrix, 11, 26);
addEdge(adjacencyMatrix, 11, 38);
addEdge(adjacencyMatrix, 11, 40);
addEdge(adjacencyMatrix, 11, 72);
addEdge(adjacencyMatrix, 11, 73);
addEdge(adjacencyMatrix, 11, 75);
addEdge(adjacencyMatrix, 11, 77);
addEdge(adjacencyMatrix, 11, 79);
addEdge(adjacencyMatrix, 11, 84);
addEdge(adjacencyMatrix, 11, 91);
addEdge(adjacencyMatrix, 12, 15);
addEdge(adjacencyMatrix, 12, 27);
addEdge(adjacencyMatrix, 12, 46);
addEdge(adjacencyMatrix, 12, 58);
addEdge(adjacencyMatrix, 12, 60);
addEdge(adjacencyMatrix, 12, 67);
addEdge(adjacencyMatrix, 12, 68);
addEdge(adjacencyMatrix, 12, 74);
addEdge(adjacencyMatrix, 12, 77);
addEdge(adjacencyMatrix, 12, 82);
addEdge(adjacencyMatrix, 13, 19);
addEdge(adjacencyMatrix, 13, 25);
addEdge(adjacencyMatrix, 13, 33);
addEdge(adjacencyMatrix, 13, 54);
addEdge(adjacencyMatrix, 13, 94);
addEdge(adjacencyMatrix, 13, 99);
addEdge(adjacencyMatrix, 14, 19);
addEdge(adjacencyMatrix, 14, 32);
addEdge(adjacencyMatrix, 14, 40);
addEdge(adjacencyMatrix, 14, 50);
addEdge(adjacencyMatrix, 14, 54);
addEdge(adjacencyMatrix, 14, 62);
addEdge(adjacencyMatrix, 14, 64);
addEdge(adjacencyMatrix, 14, 66);
addEdge(adjacencyMatrix, 14, 76);
addEdge(adjacencyMatrix, 15, 52);
addEdge(adjacencyMatrix, 15, 53);
addEdge(adjacencyMatrix, 15, 54);
addEdge(adjacencyMatrix, 15, 58);
addEdge(adjacencyMatrix, 15, 59);
addEdge(adjacencyMatrix, 15, 66);
addEdge(adjacencyMatrix, 15, 67);
addEdge(adjacencyMatrix, 15, 82);
addEdge(adjacencyMatrix, 15, 85);
addEdge(adjacencyMatrix, 15, 87);
addEdge(adjacencyMatrix, 16, 24);
addEdge(adjacencyMatrix, 16, 27);
addEdge(adjacencyMatrix, 16, 48);
addEdge(adjacencyMatrix, 16, 50);
addEdge(adjacencyMatrix, 16, 63);
addEdge(adjacencyMatrix, 16, 72);
addEdge(adjacencyMatrix, 16, 78);
addEdge(adjacencyMatrix, 16, 94);
addEdge(adjacencyMatrix, 17, 19);
addEdge(adjacencyMatrix, 17, 20);
addEdge(adjacencyMatrix, 17, 25);
addEdge(adjacencyMatrix, 17, 37);
addEdge(adjacencyMatrix, 17, 50);
addEdge(adjacencyMatrix, 17, 51);
addEdge(adjacencyMatrix, 17, 73);
addEdge(adjacencyMatrix, 17, 79);
addEdge(adjacencyMatrix, 17, 91);
addEdge(adjacencyMatrix, 18, 30);
addEdge(adjacencyMatrix, 18, 34);
addEdge(adjacencyMatrix, 18, 40);
addEdge(adjacencyMatrix, 18, 41);
addEdge(adjacencyMatrix, 18, 52);
addEdge(adjacencyMatrix, 18, 62);
addEdge(adjacencyMatrix, 18, 65);
addEdge(adjacencyMatrix, 18, 67);
addEdge(adjacencyMatrix, 18, 80);
addEdge(adjacencyMatrix, 18, 83);
addEdge(adjacencyMatrix, 18, 90);
addEdge(adjacencyMatrix, 18, 94);
addEdge(adjacencyMatrix, 19, 25);
addEdge(adjacencyMatrix, 19, 27);
addEdge(adjacencyMatrix, 19, 34);
addEdge(adjacencyMatrix, 19, 61);
addEdge(adjacencyMatrix, 19, 73);
addEdge(adjacencyMatrix, 19, 86);
addEdge(adjacencyMatrix, 19, 99);
addEdge(adjacencyMatrix, 20, 39);
addEdge(adjacencyMatrix, 20, 41);
addEdge(adjacencyMatrix, 20, 42);
addEdge(adjacencyMatrix, 20, 47);
addEdge(adjacencyMatrix, 20, 51);
addEdge(adjacencyMatrix, 20, 62);
addEdge(adjacencyMatrix, 20, 65);
addEdge(adjacencyMatrix, 20, 67);
addEdge(adjacencyMatrix, 20, 74);
addEdge(adjacencyMatrix, 20, 82);
addEdge(adjacencyMatrix, 20, 83);
addEdge(adjacencyMatrix, 20, 89);
addEdge(adjacencyMatrix, 20, 96);
addEdge(adjacencyMatrix, 20, 98);
addEdge(adjacencyMatrix, 21, 24);
addEdge(adjacencyMatrix, 21, 31);
addEdge(adjacencyMatrix, 21, 34);
addEdge(adjacencyMatrix, 21, 62);
addEdge(adjacencyMatrix, 21, 67);
addEdge(adjacencyMatrix, 21, 68);
addEdge(adjacencyMatrix, 21, 79);
addEdge(adjacencyMatrix, 21, 83);
addEdge(adjacencyMatrix, 21, 89);
addEdge(adjacencyMatrix, 21, 90);
addEdge(adjacencyMatrix, 21, 93);
addEdge(adjacencyMatrix, 21, 98);
addEdge(adjacencyMatrix, 22, 28);
addEdge(adjacencyMatrix, 22, 31);
addEdge(adjacencyMatrix, 22, 32);
addEdge(adjacencyMatrix, 22, 41);
addEdge(adjacencyMatrix, 22, 46);
addEdge(adjacencyMatrix, 22, 61);
addEdge(adjacencyMatrix, 22, 66);
addEdge(adjacencyMatrix, 22, 74);
addEdge(adjacencyMatrix, 22, 78);
addEdge(adjacencyMatrix, 22, 86);
addEdge(adjacencyMatrix, 22, 88);
addEdge(adjacencyMatrix, 22, 91);
addEdge(adjacencyMatrix, 22, 92);
addEdge(adjacencyMatrix, 22, 93);
addEdge(adjacencyMatrix, 22, 99);
addEdge(adjacencyMatrix, 23, 31);
addEdge(adjacencyMatrix, 23, 35);
addEdge(adjacencyMatrix, 23, 52);
addEdge(adjacencyMatrix, 23, 60);
addEdge(adjacencyMatrix, 23, 65);
addEdge(adjacencyMatrix, 23, 68);
addEdge(adjacencyMatrix, 23, 86);
addEdge(adjacencyMatrix, 23, 97);
addEdge(adjacencyMatrix, 24, 28);
addEdge(adjacencyMatrix, 24, 59);
addEdge(adjacencyMatrix, 24, 65);
addEdge(adjacencyMatrix, 24, 87);
addEdge(adjacencyMatrix, 25, 75);
addEdge(adjacencyMatrix, 25, 79);
addEdge(adjacencyMatrix, 25, 84);
addEdge(adjacencyMatrix, 25, 88);
addEdge(adjacencyMatrix, 26, 32);
addEdge(adjacencyMatrix, 26, 43);
addEdge(adjacencyMatrix, 26, 52);
addEdge(adjacencyMatrix, 26, 60);
addEdge(adjacencyMatrix, 26, 65);
addEdge(adjacencyMatrix, 26, 68);
addEdge(adjacencyMatrix, 26, 75);
addEdge(adjacencyMatrix, 26, 77);
addEdge(adjacencyMatrix, 26, 92);
addEdge(adjacencyMatrix, 27, 52);
addEdge(adjacencyMatrix, 27, 54);
addEdge(adjacencyMatrix, 27, 56);
addEdge(adjacencyMatrix, 27, 70);
addEdge(adjacencyMatrix, 27, 71);
addEdge(adjacencyMatrix, 27, 79);
addEdge(adjacencyMatrix, 27, 90);
addEdge(adjacencyMatrix, 28, 31);
addEdge(adjacencyMatrix, 28, 34);
addEdge(adjacencyMatrix, 28, 45);
addEdge(adjacencyMatrix, 28, 51);
addEdge(adjacencyMatrix, 28, 65);
addEdge(adjacencyMatrix, 28, 71);
addEdge(adjacencyMatrix, 28, 87);
addEdge(adjacencyMatrix, 29, 38);
addEdge(adjacencyMatrix, 29, 47);
addEdge(adjacencyMatrix, 29, 74);
addEdge(adjacencyMatrix, 29, 81);
addEdge(adjacencyMatrix, 30, 35);
addEdge(adjacencyMatrix, 30, 49);
addEdge(adjacencyMatrix, 30, 59);
addEdge(adjacencyMatrix, 30, 65);
addEdge(adjacencyMatrix, 30, 67);
addEdge(adjacencyMatrix, 30, 78);
addEdge(adjacencyMatrix, 30, 84);
addEdge(adjacencyMatrix, 30, 91);
addEdge(adjacencyMatrix, 30, 94);
addEdge(adjacencyMatrix, 30, 96);
addEdge(adjacencyMatrix, 30, 99);
addEdge(adjacencyMatrix, 31, 56);
addEdge(adjacencyMatrix, 31, 63);
addEdge(adjacencyMatrix, 31, 72);
addEdge(adjacencyMatrix, 31, 74);
addEdge(adjacencyMatrix, 31, 76);
addEdge(adjacencyMatrix, 31, 77);
addEdge(adjacencyMatrix, 31, 78);
addEdge(adjacencyMatrix, 31, 86);
addEdge(adjacencyMatrix, 31, 88);
addEdge(adjacencyMatrix, 31, 93);
addEdge(adjacencyMatrix, 31, 96);
addEdge(adjacencyMatrix, 32, 35);
addEdge(adjacencyMatrix, 32, 41);
addEdge(adjacencyMatrix, 32, 44);
addEdge(adjacencyMatrix, 32, 68);
addEdge(adjacencyMatrix, 32, 70);
addEdge(adjacencyMatrix, 32, 74);
addEdge(adjacencyMatrix, 32, 77);
addEdge(adjacencyMatrix, 32, 92);
addEdge(adjacencyMatrix, 33, 52);
addEdge(adjacencyMatrix, 33, 54);
addEdge(adjacencyMatrix, 33, 64);
addEdge(adjacencyMatrix, 33, 71);
addEdge(adjacencyMatrix, 33, 74);
addEdge(adjacencyMatrix, 33, 95);
addEdge(adjacencyMatrix, 34, 36);
addEdge(adjacencyMatrix, 34, 47);
addEdge(adjacencyMatrix, 34, 62);
addEdge(adjacencyMatrix, 34, 83);
addEdge(adjacencyMatrix, 34, 89);
addEdge(adjacencyMatrix, 35, 53);
addEdge(adjacencyMatrix, 35, 65);
addEdge(adjacencyMatrix, 35, 66);
addEdge(adjacencyMatrix, 35, 68);
addEdge(adjacencyMatrix, 35, 72);
addEdge(adjacencyMatrix, 35, 79);
addEdge(adjacencyMatrix, 35, 85);
addEdge(adjacencyMatrix, 35, 87);
addEdge(adjacencyMatrix, 35, 89);
addEdge(adjacencyMatrix, 35, 91);
addEdge(adjacencyMatrix, 36, 63);
addEdge(adjacencyMatrix, 36, 65);
addEdge(adjacencyMatrix, 36, 70);
addEdge(adjacencyMatrix, 36, 73);
addEdge(adjacencyMatrix, 36, 88);
addEdge(adjacencyMatrix, 37, 41);
addEdge(adjacencyMatrix, 37, 44);
addEdge(adjacencyMatrix, 37, 70);
addEdge(adjacencyMatrix, 37, 84);
addEdge(adjacencyMatrix, 37, 85);
addEdge(adjacencyMatrix, 37, 94);
addEdge(adjacencyMatrix, 38, 41);
addEdge(adjacencyMatrix, 38, 49);
addEdge(adjacencyMatrix, 38, 59);
addEdge(adjacencyMatrix, 38, 61);
addEdge(adjacencyMatrix, 38, 69);
addEdge(adjacencyMatrix, 38, 81);
addEdge(adjacencyMatrix, 38, 89);
addEdge(adjacencyMatrix, 38, 92);
addEdge(adjacencyMatrix, 39, 42);
addEdge(adjacencyMatrix, 39, 43);
addEdge(adjacencyMatrix, 39, 46);
addEdge(adjacencyMatrix, 39, 49);
addEdge(adjacencyMatrix, 39, 51);
addEdge(adjacencyMatrix, 39, 59);
addEdge(adjacencyMatrix, 39, 61);
addEdge(adjacencyMatrix, 39, 67);
addEdge(adjacencyMatrix, 39, 83);
addEdge(adjacencyMatrix, 39, 91);
addEdge(adjacencyMatrix, 40, 56);
addEdge(adjacencyMatrix, 40, 61);
addEdge(adjacencyMatrix, 40, 72);
addEdge(adjacencyMatrix, 40, 86);
addEdge(adjacencyMatrix, 41, 44);
addEdge(adjacencyMatrix, 41, 50);
addEdge(adjacencyMatrix, 41, 52);
addEdge(adjacencyMatrix, 41, 53);
addEdge(adjacencyMatrix, 41, 78);
addEdge(adjacencyMatrix, 41, 85);
addEdge(adjacencyMatrix, 41, 94);
addEdge(adjacencyMatrix, 42, 57);
addEdge(adjacencyMatrix, 42, 59);
addEdge(adjacencyMatrix, 42, 61);
addEdge(adjacencyMatrix, 42, 81);
addEdge(adjacencyMatrix, 42, 88);
addEdge(adjacencyMatrix, 42, 90);
addEdge(adjacencyMatrix, 42, 96);
addEdge(adjacencyMatrix, 43, 46);
addEdge(adjacencyMatrix, 43, 49);
addEdge(adjacencyMatrix, 43, 50);
addEdge(adjacencyMatrix, 43, 52);
addEdge(adjacencyMatrix, 43, 62);
addEdge(adjacencyMatrix, 43, 64);
addEdge(adjacencyMatrix, 43, 70);
addEdge(adjacencyMatrix, 43, 78);
addEdge(adjacencyMatrix, 43, 80);
addEdge(adjacencyMatrix, 43, 97);
addEdge(adjacencyMatrix, 44, 52);
addEdge(adjacencyMatrix, 44, 53);
addEdge(adjacencyMatrix, 44, 56);
addEdge(adjacencyMatrix, 44, 59);
addEdge(adjacencyMatrix, 44, 70);
addEdge(adjacencyMatrix, 44, 93);
addEdge(adjacencyMatrix, 45, 68);
addEdge(adjacencyMatrix, 45, 74);
addEdge(adjacencyMatrix, 45, 81);
addEdge(adjacencyMatrix, 46, 66);
addEdge(adjacencyMatrix, 46, 70);
addEdge(adjacencyMatrix, 46, 76);
addEdge(adjacencyMatrix, 46, 83);
addEdge(adjacencyMatrix, 46, 86);
addEdge(adjacencyMatrix, 46, 91);
addEdge(adjacencyMatrix, 47, 62);
addEdge(adjacencyMatrix, 47, 72);
addEdge(adjacencyMatrix, 47, 74);
addEdge(adjacencyMatrix, 47, 75);
addEdge(adjacencyMatrix, 47, 82);
addEdge(adjacencyMatrix, 48, 73);
addEdge(adjacencyMatrix, 48, 74);
addEdge(adjacencyMatrix, 48, 86);
addEdge(adjacencyMatrix, 48, 88);
addEdge(adjacencyMatrix, 48, 89);
addEdge(adjacencyMatrix, 48, 91);
addEdge(adjacencyMatrix, 49, 59);
addEdge(adjacencyMatrix, 49, 73);
addEdge(adjacencyMatrix, 49, 90);
addEdge(adjacencyMatrix, 49, 96);
addEdge(adjacencyMatrix, 50, 53);
addEdge(adjacencyMatrix, 50, 72);
addEdge(adjacencyMatrix, 50, 73);
addEdge(adjacencyMatrix, 50, 93);
addEdge(adjacencyMatrix, 51, 58);
addEdge(adjacencyMatrix, 51, 65);
addEdge(adjacencyMatrix, 51, 70);
addEdge(adjacencyMatrix, 51, 76);
addEdge(adjacencyMatrix, 51, 83);
addEdge(adjacencyMatrix, 51, 97);
addEdge(adjacencyMatrix, 52, 59);
addEdge(adjacencyMatrix, 52, 70);
addEdge(adjacencyMatrix, 52, 74);
addEdge(adjacencyMatrix, 52, 76);
addEdge(adjacencyMatrix, 52, 79);
addEdge(adjacencyMatrix, 52, 80);
addEdge(adjacencyMatrix, 52, 81);
addEdge(adjacencyMatrix, 52, 82);
addEdge(adjacencyMatrix, 52, 93);
addEdge(adjacencyMatrix, 53, 54);
addEdge(adjacencyMatrix, 53, 60);
addEdge(adjacencyMatrix, 53, 69);
addEdge(adjacencyMatrix, 53, 87);
addEdge(adjacencyMatrix, 53, 98);
addEdge(adjacencyMatrix, 54, 56);
addEdge(adjacencyMatrix, 54, 63);
addEdge(adjacencyMatrix, 54, 64);
addEdge(adjacencyMatrix, 54, 68);
addEdge(adjacencyMatrix, 54, 69);
addEdge(adjacencyMatrix, 54, 73);
addEdge(adjacencyMatrix, 54, 75);
addEdge(adjacencyMatrix, 54, 86);
addEdge(adjacencyMatrix, 54, 91);
addEdge(adjacencyMatrix, 54, 99);
addEdge(adjacencyMatrix, 55, 61);
addEdge(adjacencyMatrix, 55, 73);
addEdge(adjacencyMatrix, 55, 81);
addEdge(adjacencyMatrix, 55, 88);
addEdge(adjacencyMatrix, 56, 59);
addEdge(adjacencyMatrix, 56, 61);
addEdge(adjacencyMatrix, 56, 63);
addEdge(adjacencyMatrix, 56, 73);
addEdge(adjacencyMatrix, 56, 82);
addEdge(adjacencyMatrix, 56, 85);
addEdge(adjacencyMatrix, 57, 59);
addEdge(adjacencyMatrix, 57, 64);
addEdge(adjacencyMatrix, 57, 72);
addEdge(adjacencyMatrix, 58, 60);
addEdge(adjacencyMatrix, 58, 62);
addEdge(adjacencyMatrix, 58, 64);
addEdge(adjacencyMatrix, 58, 66);
addEdge(adjacencyMatrix, 58, 70);
addEdge(adjacencyMatrix, 58, 71);
addEdge(adjacencyMatrix, 58, 85);
addEdge(adjacencyMatrix, 58, 87);
addEdge(adjacencyMatrix, 59, 61);
addEdge(adjacencyMatrix, 59, 69);
addEdge(adjacencyMatrix, 60, 68);
addEdge(adjacencyMatrix, 60, 73);
addEdge(adjacencyMatrix, 60, 77);
addEdge(adjacencyMatrix, 60, 97);
addEdge(adjacencyMatrix, 61, 67);
addEdge(adjacencyMatrix, 61, 81);
addEdge(adjacencyMatrix, 62, 72);
addEdge(adjacencyMatrix, 62, 79);
addEdge(adjacencyMatrix, 62, 80);
addEdge(adjacencyMatrix, 62, 89);
addEdge(adjacencyMatrix, 62, 94);
addEdge(adjacencyMatrix, 62, 95);
addEdge(adjacencyMatrix, 63, 73);
addEdge(adjacencyMatrix, 63, 82);
addEdge(adjacencyMatrix, 63, 89);
addEdge(adjacencyMatrix, 63, 94);
addEdge(adjacencyMatrix, 64, 71);
addEdge(adjacencyMatrix, 64, 73);
addEdge(adjacencyMatrix, 64, 77);
addEdge(adjacencyMatrix, 64, 91);
addEdge(adjacencyMatrix, 64, 95);
addEdge(adjacencyMatrix, 64, 99);
addEdge(adjacencyMatrix, 65, 67);
addEdge(adjacencyMatrix, 65, 69);
addEdge(adjacencyMatrix, 65, 79);
addEdge(adjacencyMatrix, 65, 91);
addEdge(adjacencyMatrix, 65, 94);
addEdge(adjacencyMatrix, 65, 95);
addEdge(adjacencyMatrix, 66, 67);
addEdge(adjacencyMatrix, 66, 73);
addEdge(adjacencyMatrix, 66, 80);
addEdge(adjacencyMatrix, 66, 91);
addEdge(adjacencyMatrix, 66, 94);
addEdge(adjacencyMatrix, 66, 97);
addEdge(adjacencyMatrix, 66, 99);
addEdge(adjacencyMatrix, 67, 68);
addEdge(adjacencyMatrix, 67, 81);
addEdge(adjacencyMatrix, 67, 89);
addEdge(adjacencyMatrix, 68, 69);
addEdge(adjacencyMatrix, 68, 71);
addEdge(adjacencyMatrix, 68, 73);
addEdge(adjacencyMatrix, 68, 80);
addEdge(adjacencyMatrix, 68, 92);
addEdge(adjacencyMatrix, 69, 70);
addEdge(adjacencyMatrix, 69, 86);
addEdge(adjacencyMatrix, 69, 87);
addEdge(adjacencyMatrix, 69, 92);
addEdge(adjacencyMatrix, 69, 95);
addEdge(adjacencyMatrix, 70, 73);
addEdge(adjacencyMatrix, 70, 78);
addEdge(adjacencyMatrix, 70, 88);
addEdge(adjacencyMatrix, 70, 90);
addEdge(adjacencyMatrix, 70, 92);
addEdge(adjacencyMatrix, 71, 73);
addEdge(adjacencyMatrix, 71, 86);
addEdge(adjacencyMatrix, 71, 87);
addEdge(adjacencyMatrix, 71, 90);
addEdge(adjacencyMatrix, 71, 91);
addEdge(adjacencyMatrix, 72, 77);
addEdge(adjacencyMatrix, 72, 79);
addEdge(adjacencyMatrix, 72, 98);
addEdge(adjacencyMatrix, 73, 76);
addEdge(adjacencyMatrix, 73, 77);
addEdge(adjacencyMatrix, 73, 78);
addEdge(adjacencyMatrix, 73, 80);
addEdge(adjacencyMatrix, 73, 88);
addEdge(adjacencyMatrix, 73, 96);
addEdge(adjacencyMatrix, 74, 77);
addEdge(adjacencyMatrix, 74, 82);
addEdge(adjacencyMatrix, 74, 90);
addEdge(adjacencyMatrix, 74, 95);
addEdge(adjacencyMatrix, 75, 96);
addEdge(adjacencyMatrix, 76, 91);
addEdge(adjacencyMatrix, 76, 97);
addEdge(adjacencyMatrix, 77, 80);
addEdge(adjacencyMatrix, 77, 82);
addEdge(adjacencyMatrix, 77, 88);
addEdge(adjacencyMatrix, 78, 79);
addEdge(adjacencyMatrix, 78, 87);
addEdge(adjacencyMatrix, 78, 93);
addEdge(adjacencyMatrix, 79, 91);
addEdge(adjacencyMatrix, 80, 89);
addEdge(adjacencyMatrix, 80, 94);
addEdge(adjacencyMatrix, 80, 96);
addEdge(adjacencyMatrix, 80, 97);
addEdge(adjacencyMatrix, 80, 98);
addEdge(adjacencyMatrix, 81, 85);
addEdge(adjacencyMatrix, 81, 93);
addEdge(adjacencyMatrix, 81, 95);
addEdge(adjacencyMatrix, 82, 84);
addEdge(adjacencyMatrix, 82, 85);
addEdge(adjacencyMatrix, 82, 87);
addEdge(adjacencyMatrix, 82, 92);
addEdge(adjacencyMatrix, 82, 99);
addEdge(adjacencyMatrix, 83, 88);
addEdge(adjacencyMatrix, 84, 88);
addEdge(adjacencyMatrix, 84, 97);
addEdge(adjacencyMatrix, 84, 99);
addEdge(adjacencyMatrix, 85, 87);
addEdge(adjacencyMatrix, 85, 95);
addEdge(adjacencyMatrix, 85, 98);
addEdge(adjacencyMatrix, 86, 91);
addEdge(adjacencyMatrix, 86, 98);
addEdge(adjacencyMatrix, 87, 89);
addEdge(adjacencyMatrix, 88, 90);
addEdge(adjacencyMatrix, 89, 93);
addEdge(adjacencyMatrix, 89, 94);
addEdge(adjacencyMatrix, 90, 93);
addEdge(adjacencyMatrix, 91, 93);
addEdge(adjacencyMatrix, 91, 95);
addEdge(adjacencyMatrix, 91, 96);
addEdge(adjacencyMatrix, 95, 96);


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

    // Stampa la matrice risultante
    std::cout << "Matrice di Adiacenza per il grafo:\n\n";
    printMatrix(adjacencyMatrix);

    //print with Graphviz DOT format
    printDot(adjacencyMatrix);


    //print ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyMatrix, orderedList);
    cout << "Ordered list of nodes based on degree:\n";
    for (const auto &node : orderedList) {
        cout << node << " ";
    }
    cout << "\n";


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