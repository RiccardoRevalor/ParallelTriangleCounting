#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <string> // Added for std::string operationst
#include "../../utils/utils.h"

#define DEBUG 0

// CUDA Error checking macro
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

// Row-major matrix multiplication kernel using dynamically allocated shared memory
// TILE_SIZE is now passed as an argument, but the shared memory is declared externally
__global__ void multiplyMatricesSharedMemoryKernel(
    const int *A,
    const int *B,
    int *C,
    int n,
    int TILE_SIZE // TILE_SIZE is passed as a value
) {
    // Declare dynamically allocated shared memory
    // The size in bytes is passed as the third argument in the kernel launch.
    // cudaMemset will initialize this memory to zero.
    extern __shared__ int sharedMemBuffer[]; 
    
    // Pointers to the shared memory tiles for A and B
    int* s_A = sharedMemBuffer; // s_A starts at the beginning of the buffer
    int* s_B = s_A + TILE_SIZE * TILE_SIZE; // s_B starts after s_A

    // Global row and column for the element in C that this thread will compute
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0; // Accumulator for C[globalRow][globalCol]

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles from global memory to shared memory
        // Each thread loads one element from A and one from B into shared memory.
        int aIdx = globalRow * n + (tile * TILE_SIZE + threadIdx.x); // Element from A for s_A
        int bIdx = (tile * TILE_SIZE + threadIdx.y) * n + globalCol; // Element from B for s_B

        // Calculate local indices for shared memory (linear access)
        int s_A_local_idx = threadIdx.y * TILE_SIZE + threadIdx.x;
        int s_B_local_idx = threadIdx.y * TILE_SIZE + threadIdx.x;

        // Bounds checking for loading from global memory
        if (globalRow < n && (tile * TILE_SIZE + threadIdx.x) < n) {
            s_A[s_A_local_idx] = A[aIdx];
        } else {
            s_A[s_A_local_idx] = 0; // Pad with zeros if out of bounds
        }

        if ((tile * TILE_SIZE + threadIdx.y) < n && globalCol < n) {
            s_B[s_B_local_idx] = B[bIdx];
        } else {
            s_B[s_B_local_idx] = 0; // Pad with zeros if out of bounds
        }

        __syncthreads(); // Ensure all data is loaded into shared memory

        // Perform dot product using shared memory tiles
        // Access shared memory using 2D logic with 1D array
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y * TILE_SIZE + k] * s_B[k * TILE_SIZE + threadIdx.x];
        }

        __syncthreads(); // Ensure all threads finished using current tile's data
    }

    // Write the accumulated sum to global memory
    if (globalRow < n && globalCol < n) {
        C[globalRow * n + globalCol] = sum;
    }
}


__global__ void traceKernel(
    const int *A,
    int *traceGlobal,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(traceGlobal, A[idx * n + idx]);
    }
}


// Function to allocate and copy a 2D adjacency matrix to device memory
void allocateAndCopyToDevice(const std::vector<std::vector<int>>& hostMatrix, int** deviceMatrix, int n) {
    CUDA_CHECK(cudaMalloc(deviceMatrix, n * n * sizeof(int)));
    std::vector<int> flatHostMatrix(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flatHostMatrix[i * n + j] = hostMatrix[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(*deviceMatrix, flatHostMatrix.data(), n * n * sizeof(int), cudaMemcpyHostToDevice));
}

// Function to allocate and zero-set a device matrix
void allocateDeviceMatrix(int** deviceMatrix, int n) {
    CUDA_CHECK(cudaMalloc(deviceMatrix, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemset(*deviceMatrix, 0, n * n * sizeof(int))); // Initialize to 0
}


void printMatrix(const std::vector<std::vector<int>>& matrix) {
    int numVertices = matrix.size();
    
    // Print column headers
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

    // Print matrix rows
    for (int i = 0; i < numVertices; ++i) {
        std::cout << std::setw(2) << i << "|";
        for (int j = 0; j < numVertices; ++j) {
            std::cout << std::setw(3) << matrix[i][j];
        }
        std::cout << "\n";
    }
}


// Graphviz DOT format for printing the graph
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


int main(int argc, char **argv){
    //argc == 4
    if (argc != 5){
        std::cerr << "Usage: " << argv[0] << " <input_file> <TILE_SIZE> <TRACE_BLOCKSIZE> <GPU_MODEL>\n";
        return 1;
    }

    //extract TILE_SIZE and TRACE_BLOCKSIZE from command line arguments
    int TILE_SIZE = std::stoi(argv[2]);
    int TRACE_BLOCKSIZE = std::stoi(argv[3]);
    std::string gpuModel = argv[4]; // Changed to std::string directly

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

    // Device row-major matrices for A, A^2, A^3
    int n = adjacencyMatrix.size();
    int *d_A, *d_A2, *d_A3, *d_trace;
    int h_trace = 0;

    // Allocate and copy to device
    allocateAndCopyToDevice(adjacencyMatrix, &d_A, n);
    allocateDeviceMatrix(&d_A2, n);
    allocateDeviceMatrix(&d_A3, n);

    // Define grid and block dimensions for shared memory kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Calculate dynamic shared memory size in bytes
    // Need space for s_A (TILE_SIZE * TILE_SIZE ints) and s_B (TILE_SIZE * TILE_SIZE ints)
    size_t sharedMemBytes = 2 * TILE_SIZE * TILE_SIZE * sizeof(int);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Step 2: compute A^2 = A * A
    // Pass sharedMemBytes as the third argument to the kernel launch
    multiplyMatricesSharedMemoryKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_A, d_A, d_A2, n, TILE_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // SYNC: Ensure A^2 computation is complete before starting A^3
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: compute A^3 = A^2 * A
    // Pass sharedMemBytes as the third argument to the kernel launch
    multiplyMatricesSharedMemoryKernel<<<gridDim, blockDim, sharedMemBytes>>>(d_A2, d_A, d_A3, n, TILE_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // SYNC: Ensure A^3 computation is complete before copying trace
    CUDA_CHECK(cudaDeviceSynchronize());

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration_mm = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Time taken for matrix multiplication (with shared memory): " << duration_mm << " microseconds" << std::endl;

    // Compute trace
    // Allocate device memory for trace result
    CUDA_CHECK(cudaMalloc(&d_trace, sizeof(int)));
    // Initialize to 0 on device
    CUDA_CHECK(cudaMemset(d_trace, 0, sizeof(int)));

    // Launch kernel for trace computation
    const int traceBlockSize = TRACE_BLOCKSIZE; // Use TRACE_BLOCKSIZE from args
    const int traceGridSize = (n + traceBlockSize - 1) / traceBlockSize;
    startTime = std::chrono::high_resolution_clock::now();
    traceKernel<<<traceGridSize, traceBlockSize>>>(d_A3, d_trace, n);
    CUDA_CHECK(cudaGetLastError());

    // SYNC: Ensure trace computation is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    endTime = std::chrono::high_resolution_clock::now();
    auto duration_trace = std::chrono::duration_cast<std::chrono::microseconds>(endTime -startTime).count();
    std::cout << "Time taken for trace computation: " << duration_trace << " microseconds" << std::endl;

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(&h_trace, d_trace, sizeof(int), cudaMemcpyDeviceToHost));

    // Final Formula: tot number of triangles = trace(A^3) / 6
    std::cout << "Total number of triangles: " << h_trace / 6 << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_A2));
    CUDA_CHECK(cudaFree(d_A3));
    CUDA_CHECK(cudaFree(d_trace));

    CUDA_CHECK(cudaDeviceReset()); // Resets the device, useful for cleanup

    // create cross validation output file
    std::ofstream crossValidationFile;
    // Corrected string concatenatitn for filename
    crossValidationFile.open("../../cross_validation_output/cuda_matrixmultiplication_v2/cross_validation_output_" + gpuModel + ".csv", std::ios::app);
    if (!crossValidationFile.is_open()) { // Use is_open() for robust check
        std::cerr << "Error opening cross validation output file!" << std::endl;
        return -1;
    }

    // write parameters and final time to the file, CSV format
    // put header if file is empty
    // Check if the file is empty by seeking to end and checking position
    crossValidationFile.seekp(0, std::ios::end); // Move to end
    if (crossValidationFile.tellp() == 0) { // Check position
        crossValidationFile << "TILE_SIZE,TRACE_BLOCKSIZE,GPU_MODEL,MM_DURATION_US,TRACE_DURATION_US,TOTAL_DURATION_US,TRIANGLES\n";
    }
    // Changed `duration` to `duration_mm` and added `duration_trace`
    crossValidationFile << TILE_SIZE << "," << TRACE_BLOCKSIZE << "," << gpuModel << ","
                        << duration_mm << "," << duration_trace << "," << (duration_mm + duration_trace) << "," 
                        << h_trace / 6 << "\n";

    crossValidationFile.close();

    return 0;
}