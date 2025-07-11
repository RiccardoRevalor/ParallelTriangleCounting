#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

using namespace std;

void addEdge(vector<vector<int>>& matrix, int u, int v);
vector<vector<int>> populateAdjacencyMatrix(string fileName);

#endif