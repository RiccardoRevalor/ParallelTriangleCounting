#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata


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
    std::cout << "graph G {\n";

    int numVertices = matrix.size();
    for (int i = 0; i < numVertices; ++i) {
        for (int j = i + 1; j < numVertices; ++j) {
            if (matrix[i][j] == 1) {
                std::cout << "  " << i << " -- " << j << ";\n";
            }
        }
    }

    std::cout << "}\n";
}

int main() {
    // Il grafo ha 12 nodi, numerati da 0 a 11
    const int NUM_VERTICES = 12;

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    std::vector<std::vector<int>> adjacencyMatrix(NUM_VERTICES, std::vector<int>(NUM_VERTICES, 0));

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
    
    return 0;
}