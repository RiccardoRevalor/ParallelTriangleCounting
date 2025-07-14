#include <vector>
#include "matrixMath.h"

std::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& A,
                                               const std::vector<std::vector<int>>& B) {
    int n = A.size();
    std::vector<std::vector<int>> result(n, std::vector<int>(n, 0));

    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            if (A[i][k] != 0)  // Optimization for sparse matrices
                for (int j = 0; j < n; ++j)
                    result[i][j] += A[i][k] * B[k][j];

    return result;
}

std::vector<std::vector<int>> cubeAdjacencyMatrix(const std::vector<std::vector<int>>& adjacencyMatrix) {
    // Compute A^2
    std::vector<std::vector<int>> A2 = multiplyMatrices(adjacencyMatrix, adjacencyMatrix);
    // Compute A^3 = A^2 * A
    std::vector<std::vector<int>> A3 = multiplyMatrices(A2, adjacencyMatrix);
    return A3;
}
