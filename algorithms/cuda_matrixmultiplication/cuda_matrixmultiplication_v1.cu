#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include "../../utils/utils.h"

#define DEBUG 0

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

//row major matrix multiplication kernel
__global__ void multiplyMatricesKernel(
    const int *A,
    const int *B,
    int *C,
    int n
){

    //each thread is given a row and column to multiply together
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}


__global__ void traceKernel(
    const int *A,
    int *traceGlobal,
    int n
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(traceGlobal, A[idx * n + idx]);
    }
}


// Function to allocate and copy a 2D adjacency matrix to device memory
void allocateAndCopyToDevice(const std::vector<std::vector<int>>& hostMatrix, int** deviceMatrix, int n) {
    CUDA_CHECK(cudaMalloc(deviceMatrix, n * n * sizeof(int)));

    // Flatten the host matrix for copying
    std::vector<int> flatHostMatrix(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flatHostMatrix[i * n + j] = hostMatrix[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(*deviceMatrix, flatHostMatrix.data(), n * n * sizeof(int), cudaMemcpyHostToDevice));
}

//Function to allocate and zero-set a device matrix
void allocateDeviceMatrix(int** deviceMatrix, int n) {
    CUDA_CHECK(cudaMalloc(deviceMatrix, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemset(*deviceMatrix, 0, n * n * sizeof(int))); // Initialize to 0
}


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


int main(int argc, char *argv[]) {

    if (argc != 4){
        std::cerr << "Usage: " << argv[0] << " <input_file> <BLOCK_SIZE> <GPU_MODEL>\n";
        return 1;
    }

    //extract BLOCK_SIZE from command line arguments
    int blockSize = std::stoi(argv[2]);
    std::string gpuModel = argv[3]; 

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


    std::vector<std::vector<int>> adjacencyMatrix = populateAdjacencyMatrix(input);


    if (DEBUG) {
        std::cout << "Matrice di Adiacenza per il grafo:\n\n";
        printMatrix(adjacencyMatrix);
        printDot(adjacencyMatrix);
    }


    //device row major matrices for A, A^2, A^3
    int n = adjacencyMatrix.size();
    int *d_A, *d_A2, *d_A3, *d_trace;
    int h_trace = 0;

    //allocate and copy to device
    allocateAndCopyToDevice(adjacencyMatrix, &d_A, n);
    allocateDeviceMatrix(&d_A2, n);
    allocateDeviceMatrix(&d_A3, n);

    // Define grid and block dimensions
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    auto startTime = std::chrono::high_resolution_clock::now();

    //Step 2: compute A^2
    multiplyMatricesKernel<<<gridDim, blockDim>>>(d_A, d_A, d_A2, n);
    CUDA_CHECK(cudaGetLastError());

    //SYNC
    CUDA_CHECK(cudaDeviceSynchronize());

    //compute A^3
    multiplyMatricesKernel<<<gridDim, blockDim>>>(d_A2, d_A, d_A3, n);
    CUDA_CHECK(cudaGetLastError());

    //SYNC
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Time taken for matrix multiplication: " << duration << " microseconds" << std::endl;

    //compute trace
    //alloc 
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(int)));
    //init to 0
    CUDA_CHECK(cudaMemset(d_trace, 0, sizeof(int)));


    //launch kernel for trace compuation
    const int traceBlockSize = 256;
    const int traceGridSize = (n + traceBlockSize - 1) / traceBlockSize;
    startTime = std::chrono::high_resolution_clock::now();
    traceKernel<<<traceGridSize, traceBlockSize>>>(d_A3, d_trace, n);
    CUDA_CHECK(cudaGetLastError());

    //SYNC
    CUDA_CHECK(cudaDeviceSynchronize());

    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime -startTime).count();
    std::cout << "Time taken for trace computation: " << duration << " microseconds" << std::endl;

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(&h_trace, d_trace, sizeof(int), cudaMemcpyDeviceToHost));

    //Final Formula: tot number of traingles = trace(A^3) / 6
    std::cout << "Total number of triangles: " << h_trace / 6 << std::endl;

    //Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_A3));
    CUDA_CHECK(cudaFree(d_trace));

    CUDA_CHECK(cudaDeviceReset());


    // create cross validation output file
    std::ofstream crossValidationFile;
    // Corrected string concatenation for filename
    crossValidationFile.open("../../cross_validation_output/cuda_matrixmultiplication_v1/cross_validation_output_" + gpuModel + ".csv", std::ios::app);
    if (!crossValidationFile.is_open()) { // Use is_open() for robust check
        std::cerr << "Error opening cross validation output file!" << std::endl;
        return -1;
    }

    // write parameters and final time to the file, CSV format
    // put header if file is empty
    // Check if the file is empty by seeking to end and checking position
    crossValidationFile.seekp(0, std::ios::end); // Move to end
    if (crossValidationFile.tellp() == 0) { // Check position
        crossValidationFile << "BLOCK_SIZE,GPU_MODEL,TOTAL_DURATION_US,TRIANGLES\n";
    }
    // Changed `duration` to `duration_mm` and added `duration_trace`
    crossValidationFile << blockSize << "," << gpuModel << ","
                        << duration << "," 
                        << h_trace / 6 << "\n";

    crossValidationFile.close();

    return 0;
}