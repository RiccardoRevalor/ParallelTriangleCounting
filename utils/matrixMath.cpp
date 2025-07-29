#include <vector>
#include <thread>
#include <algorithm> // For std::min

#define NUM_THREADS 8

//each thread computes a specific range of rows for the result matrix C = A * B
void parallelMultiplyMatrices(const std::vector<std::vector<int>>& A,
                              const std::vector<std::vector<int>>& B,
                              std::vector<std::vector<int>>& C, 
                              int startRow, int endRow) {
    int n = A.size(); 
    for (int i = startRow; i < endRow; ++i) {
        for (int k = 0; k < n; ++k) {
            //optimization for sparse matrices: only multiply if A[i][k] is not zero
            if (A[i][k] != 0) {
                for (int j = 0; j < n; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
}

// Function to cube the adjacency matrix
std::vector<std::vector<int>> cubeAdjacencyMatrix(const std::vector<std::vector<int>>& adjacencyMatrix) {
    int n = adjacencyMatrix.size();
    if (n == 0) return {}; 

    // Initialize result matrices
    std::vector<std::vector<int>> A2(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> A3(n, std::vector<int>(n, 0));

    std::vector<std::thread> threads;
    int chunkSize = (n + NUM_THREADS - 1) / NUM_THREADS;

    // --- Stage 1: Compute A^2 = A * A ---
    threads.clear(); 
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * chunkSize;
        int endRow = std::min(startRow + chunkSize, n);
        if (startRow < endRow) { 
            threads.emplace_back(parallelMultiplyMatrices, std::cref(adjacencyMatrix),
                                  std::cref(adjacencyMatrix), std::ref(A2), startRow, endRow);
        }
    }

    
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // --- Stage 2: Compute A^3 = A^2 * A ---
    threads.clear(); 
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * chunkSize;
        int endRow = std::min(startRow + chunkSize, n);
        if (startRow < endRow) { 
            threads.emplace_back(parallelMultiplyMatrices, std::cref(A2),
                                  std::cref(adjacencyMatrix), std::ref(A3), startRow, endRow);
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return A3;
}