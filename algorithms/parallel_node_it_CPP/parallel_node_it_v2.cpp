#include <iostream>
#include <vector>
#include <iomanip> // Per una stampa più ordinata
#include <map>
#include <algorithm>
#include <set>
#include <chrono>
#include <thread>
#include <atomic>
#include <unordered_set>
#include <fstream>
#include "../../utils/utils.h"
#include "../../utils/matrixMath.h"
#include <string>

#define NUM_THREADS 16
#define DEBUG 0


using namespace std;


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



void forwardAlgorithmParallel(const vector<int> &orderedList, const vector<vector<int>> &adjacencyMatrix, atomic<int> &countTriangles, int start, int end) {

    //maps of ranks of vertices based on their degree on the graph, so their position in the ordered list
    map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }


    for (int i = start; i < end; ++i) {
        int s = orderedList[i];
        //get adjacency list of the current node
        vector<int> neighbors = getNeighbors(adjacencyMatrix, s);

        unordered_set<int> u_neighbors_set(neighbors.begin(), neighbors.end());

        for (int t : neighbors) {
            if (ranks.at(s) >= ranks.at(t)) {
                continue;
            }

            // Ora, per ogni vicino 't', controlla i suoi vicini 'w'
            std::vector<int> t_neighbors = getNeighbors(adjacencyMatrix, t);
            for (int v : t_neighbors) {
                // Se v ha rank maggiore di t E v è anche vicino di s,
                // abbiamo trovato un triangolo contato una sola volta.
                if (ranks.at(t) < ranks.at(v) && u_neighbors_set.count(v)) {
                    countTriangles++; // Incrementa il contatore locale del thread
                }
            }
        }
        
    }
}


float getTotTriangles(const vector<vector<int>> adjacencyMatrix) {
    vector<vector<int>> A3 = cubeAdjacencyMatrix(adjacencyMatrix);

    const float factor = (float)1/ (float)6;

    // compute tractiant
    int traciant = 0;
    for (int i = 0; i < A3.size(); i++) {
        for (int j = 0; j < A3[i].size(); j++) {
            if (i == j) {
                traciant += A3[i][j];
            }
        }
    }

    return factor*traciant;
}

int main(int argc, char **argv) {

    if (argc != 4){
        cerr << "Usage: " << argv[0] << " <input_file> <NUM_THREADS> <GPU_MODEL>" << endl;
        return 1;
    }

    //if filename is "i" then ask for input
    std::string input;
    if (argv[1] == "i") {
        while (true) {
            std::cout << "insert file name: ";
            std::getline(std::cin, input);
            input = "../../graph_file/" + input;

            std::ifstream file(input);
            if (file.is_open())
                break;
            std::cout << input << " doesn't exist!" << std::endl;
        }
    } else {
        //extract file name from command line arguments
        input = "../../graph_file/" + std::string(argv[1]);
    }

    std::string gpuModel = argv[3];
    int numThreads = std::stoi(argv[2]);

    // Crea la matrice di adiacenza NxN, inizializzata con tutti 0
    vector<vector<int>> adjacencyMatrix = populateAdjacencyMatrix(input);

    // Stampa la matrice risultante
    if (DEBUG){
        std::cout << "Matrice di Adiacenza per il grafo:\n\n";
        printMatrix(adjacencyMatrix);

        //print with Graphviz DOT format
        printDot(adjacencyMatrix);
    }


    //print ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyMatrix, orderedList);
    if (DEBUG) {
        cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : orderedList) {
            cout << node << " ";
        }
        cout << "\n";
    }



    cout << "-----------------------------------------------------------------" << endl;
    atomic<int> countTriangles = 0;
    vector<thread> threads;
    int chunkSize = (orderedList.size() + NUM_THREADS - 1) / NUM_THREADS; // Round up division

    auto startTime = chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * chunkSize;
        int end = min(start + chunkSize, (int)orderedList.size());

       if (start < end ) {
           threads.emplace_back(forwardAlgorithmParallel, ref(orderedList), ref(adjacencyMatrix), ref(countTriangles), start, end);
       }
    }

    for (auto &t : threads) {
        t.join();
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Time taken for forward algorithm: " << duration.count() << " microseconds" << endl;

    cout << "Tot Max Theoretical Triangles: " << getTotTriangles(adjacencyMatrix) << endl;
    cout << "Triangles found by forward algorithm: " << countTriangles << endl;


     // create cross validation output file
    std::ofstream crossValidationFile;
    //REMOVE .g extension from input file name
    size_t pos = input.find_last_of(".");
    if (pos != std::string::npos) {
        input = input.substr(0, pos);
    }
    //take just the file name without path
    pos = input.find_last_of("/");
    if (pos != std::string::npos) {
        input = input.substr(pos + 1);
    }
    string outputFileName("../../cross_validation_output/parallel_node_it_v2/" + input + "_" + gpuModel + ".csv");
    cout << "Output file name: " << outputFileName << endl;
    crossValidationFile.open(outputFileName, std::ios::app);
    if (!crossValidationFile.is_open()) { // Use is_open() for robust check
        std::cerr << "Error opening cross validation output file!" << std::endl;
        return -1;
    }

    // write parameters and final time to the file, CSV format
    // put header if file is empty
    // Check if the file is empty by seeking to end and checking position
    crossValidationFile.seekp(0, std::ios::end); // Move to end
    if (crossValidationFile.tellp() == 0) { // Check position
        crossValidationFile << "NUM_THREADS,GPU_MODEL,TOTAL_DURATION_US,TRIANGLES\n";
    }
    // Changed `duration` to `duration_mm` and added `duration_trace`
    crossValidationFile << numThreads << ","
                      << gpuModel << ","
                      << duration.count() << ","
                      << countTriangles << "\n";

    crossValidationFile.close();


    return 0;
}