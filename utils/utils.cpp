#include "utils.h"
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>

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


map<int, vector<int>> populateAdjacencyVectors(string fileName){
    int NUM_VERTICES;
    
    fstream file;
    file.open(fileName, ios::in);

    string line;
    getline(file, line);
    NUM_VERTICES = stoi(line);

    map<int, vector<int>> adjacencyVectors;

    while(getline(file, line)) {
        // line in the file: 1 2 
        stringstream ss(line);
        int v1, v2;
        ss >> v1 >> v2;

        adjacencyVectors[v1].push_back(v2);
        adjacencyVectors[v2].push_back(v1); //assuming undirected graph

    }
    
    return adjacencyVectors;
}



void convertToCRS(const std::map<int, std::vector<int>>& adjacencyVectors,
                  std::vector<int>& row_ptr,
                  std::vector<int>& col_idx,
                  int& num_nodes,
                  bool sortNeighbors) {
    // Determine the number of nodes from the adjacency map
    if (!adjacencyVectors.empty()) {
        num_nodes = adjacencyVectors.rbegin()->first + 1; // assumes 0-based indexing
    } else {
        num_nodes = 0;
        row_ptr.clear();
        col_idx.clear();
        return;
    }

    row_ptr.resize(num_nodes + 1, 0);

    int edge_counter = 0;
    for (int node = 0; node < num_nodes; ++node) {
        row_ptr[node] = edge_counter;

        auto it = adjacencyVectors.find(node);
        if (it != adjacencyVectors.end()) {

            std::vector<int> neighbors = it->second; 
            if (sortNeighbors) {
                std::sort(neighbors.begin(), neighbors.end()); //sort neighbors if required
            }
            col_idx.insert(col_idx.end(), neighbors.begin(), neighbors.end());
            edge_counter += neighbors.size();
        }
    }

    row_ptr[num_nodes] = edge_counter;
}