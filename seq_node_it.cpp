#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>

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


void forwardAlgorithm(const vector<int> &orderedList, const vector<vector<int>> &adjacencyMatrix) {
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
                    }
                    cout << endl;
                }

                //last step: update the set A[t]
                A[t].insert(s);

            }
        }
    }
}

int main() {
    // Il grafo ha 12 nodi, numerati da 0 a 11
    const int NUM_VERTICES = 12;

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    vector<vector<int>> adjacencyMatrix(NUM_VERTICES, vector<int>(NUM_VERTICES, 0));

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

    // Stampa la matrice risultante
    //std::cout << "Matrice di Adiacenza per il grafo:\n\n";
    //printMatrix(adjacencyMatrix);

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
    // Run the forward algorithm
    forwardAlgorithm(orderedList, adjacencyMatrix);

    return 0;
}