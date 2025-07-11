#include "utils.h"
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

void addEdge(std::vector<std::vector<int>>& matrix, int u, int v) {
    
    matrix[u][v] = 1;
    matrix[v][u] = 1;
}

vector<vector<int>> populateAdjacencyMatrix(string fileName) {
    int NUM_VERTICES;
    
    fstream file;
    file.open(fileName, ios::in);
    
    string line;
    getline(file, line);
    NUM_VERTICES = stoi(line);
    vector<vector<int>> adjacencyMatrix(NUM_VERTICES, vector<int>(NUM_VERTICES, 0));

    int v1, v2;
    while(getline(file, line)) {
        // line in the file: 1 2 
        stringstream ss(line);
        ss >> v1 >> v2;
        addEdge(adjacencyMatrix, v1, v2);
    }

    return adjacencyMatrix;

}