#include "utils.h"
#include <fstream>
#include <string>
#include <sstream>
#include <map>

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