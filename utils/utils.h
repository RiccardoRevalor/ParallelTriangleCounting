#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <map>

using namespace std;

void addEdge(vector<vector<int>>& matrix, int u, int v);
vector<vector<int>> populateAdjacencyMatrix(string fileName);
map<int, vector<int>> populateAdjacencyVectors(string fileName);
void convertToCRS(const std::map<int, std::vector<int>>& adjacencyVectors,
                    std::vector<int>& row_ptr,
                    std::vector<int>& col_idx,
                    int& num_nodes,
                    bool sortNeighbors = false);

#endif