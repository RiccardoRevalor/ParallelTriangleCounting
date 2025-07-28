#include <vector>
#include <iostream>
#include <iomanip> // For better formatting
#include <vector>
#include <chrono>
#include <barrier>
#include <thread>
#include <fstream>
#include "../../utils/utils.h"
#define NUMTHREADS 32

#define DEBUG 0
using namespace std;

barrier multiplyMatrixBarrier(NUMTHREADS);

void multiplyMatrices(const vector<vector<int>>& A,
                                               const vector<vector<int>>& B,
                                               const int startRow, const int endRow,
                                               vector<vector<int>> &result) {
    int n = A.size();

    for (int i = startRow; i < endRow; ++i)
        for (int k = 0; k < n; ++k)
            if (A[i][k] != 0)  // Optimization for sparse matrices
                for (int j = 0; j < n; ++j)
                    result[i][j] += A[i][k] * B[k][j];

}

void cubeAdjacencyMatrix(const std::vector<std::vector<int>>& adjacencyMatrix, vector<vector<int>> &A2_result, vector<vector<int>> &A3_result, const int startRow, const int endRow) {
    // Compute A^2
    multiplyMatrices(adjacencyMatrix, adjacencyMatrix, startRow, endRow, A2_result); //A2_result = A^2

    //WAIT for all threads to finish computing A^2
    multiplyMatrixBarrier.arrive_and_wait();

    // Compute A^3 = A^2 * A
    multiplyMatrices(A2_result, adjacencyMatrix, startRow, endRow, A3_result);

}

float getTotTriangles(const vector<vector<int>> adjacencyMatrix) {
    int n = adjacencyMatrix.size();
    vector<vector<int>> A2(n, vector<int>(n, 0));
    vector<vector<int>> A3(n, vector<int>(n, 0));
    
    vector<thread> threads;
    int chunkSize = (n + NUMTHREADS - 1) / NUMTHREADS; // Calculate chunk size for each thread

    cout << "Starting parallel matrix multiplication..." << endl;
    auto startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i < NUMTHREADS; ++i) {
        int startRow = i * chunkSize;
        int endRow = min(startRow + chunkSize, n);
        threads.emplace_back(cubeAdjacencyMatrix, ref(adjacencyMatrix), ref(A2), ref(A3), startRow, endRow);
    }

    cout << "Waiting for threads to finish..." << endl;
    for (auto& t : threads) {
        t.join();
    }
    cout << "All threads finished." << endl;

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
    cout << "Time taken for parallel matrix multiplication: " << duration << " microseconds" << endl;

    const float factor = (float)1/ (float)6;

    // compute trace
    int trace = 0;
    startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i < A3.size(); i++) {
        trace += A3[i][i];          
    }
    endTime = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
    cout << "Time taken for trace computation: " << duration << " microseconds" << endl;

    return factor*trace;
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


int main(void){
    std::string input;
    while(true) {
        cout << "insert file name: ";
        std::getline(std::cin, input);
        input = "../../graph_file/" + input;
        
        // check whether file can be opened
        std::ifstream file(input);
        
        if (file.is_open())
            break;
        cout << input << " doesn't exist!" << endl; 
    }

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    vector<vector<int>> adjacencyMatrix = populateAdjacencyMatrix(input);

    // Stampa la matrice risultante
    if (DEBUG) {
        std::cout << "Matrice di Adiacenza per il grafo:\n\n";
        printMatrix(adjacencyMatrix);

        //print with Graphviz DOT format
        printDot(adjacencyMatrix);
    }


    int countTriangles = getTotTriangles(adjacencyMatrix);
    cout << "Tot Max Theoretical Triangles: " << countTriangles << endl;



    return 0;

}