#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>       // For CSV line parsing
#include <limits>        // For numeric_limits
#include <stdexcept>     // For exceptions like stoi, stoll

// Define NOMINMAX to prevent windows.h from defining min/max macros,
// which can conflict with std::min/std::max or other identifiers.
#define NOMINMAX
#include <windows.h>     // For CreateProcess, etc.
#include <filesystem>    // For std::filesystem (requires C++17)

// Global constants for relative paths (will be resolved to absolute paths at runtime)
const std::string CUDA_MATRIXMULTIPLICATION_V2_RELATIVE = ".\\..\\cuda_matrixmultiprocess\\main_v2.exe";
const std::string CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_RELATIVE = ".\\..\\cross_validation_output\\cuda_matrixmultiprocess_v2\\";

// Global variables for resolved absolute paths
std::string CUDA_MATRIXMULTIPLICATION_V2_ABSOLUTE;
std::string CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_ABSOLUTE;

// Function to execute a system command using CreateProcessA
// It now takes the absolute application path and the arguments string separately.
// Returns the exit code of the launched process (0 for success, non-zero for error),
// or -1 if CreateProcess itself failed.
int executeCommand(const std::string& applicationPath, const std::string& arguments) {
    // Construct the full command line string.
    // It's good practice to quote the application path if it might contain spaces.
    std::string fullCommandLine = "\"" + applicationPath + "\" " + arguments;

    // Create a mutable buffer for the command line.
    // CreateProcessA might modify this buffer.
    std::vector<char> cmdLineBuf(fullCommandLine.begin(), fullCommandLine.end());
    cmdLineBuf.push_back('\0'); // Null-terminate the string

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    std::cout << "Executing: " << fullCommandLine << std::endl;

    // CreateProcessA arguments:
    // lpApplicationName: The module to be executed (absolute path recommended).
    // lpCommandLine: The command line to be executed. Must be a modifiable string.
    // lpCurrentDirectory: NULL means it uses the calling process's current directory.
    if (!CreateProcessA(
        applicationPath.c_str(), // lpApplicationName: Absolute path to the executable
        cmdLineBuf.data(),       // lpCommandLine: Modifiable command line string
        NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        
        std::cerr << "Error: CreateProcess failed with error code " << GetLastError() << std::endl;
        return -1; // Indicate failure to launch process
    }

    // Wait for the launched process to finish
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Get the exit code of the launched process
    DWORD exitCode;
    if (!GetExitCodeProcess(pi.hProcess, &exitCode)) {
        std::cerr << "Error: GetExitCodeProcess failed with error code " << GetLastError() << std::endl;
        exitCode = -1; // Indicate error getting exit code
    }

    // Close process and thread handles to free resources
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return static_cast<int>(exitCode); // Return the actual exit code of the launched process
}

// Structure to store a row of data from the CSV
struct ResultEntry {
    int tileSize;
    int traceBlockSize;
    std::string gpuModel;
    long long mmDurationUs;
    long long traceDurationUs;
    long long totalDurationUs;
    int triangles;
};

int main() {
    // Resolve absolute paths at program start for robustness
    try {
        CUDA_MATRIXMULTIPLICATION_V2_ABSOLUTE = std::filesystem::absolute(CUDA_MATRIXMULTIPLICATION_V2_RELATIVE).string();
        CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_ABSOLUTE = std::filesystem::absolute(CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_RELATIVE).string();
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error resolving paths: " << e.what() << std::endl;
        return 1; // Exit if paths cannot be resolved
    }

    std::cout << "--- CUDA Parameter Optimizer ---\n";
    std::cout << "Resolved CUDA Executable Path: " << CUDA_MATRIXMULTIPLICATION_V2_ABSOLUTE << std::endl;
    std::cout << "Resolved CV Base Path: " << CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_ABSOLUTE << std::endl;


    // --- Configuration for Cross-Validation ---
    // Use the resolved absolute path for the executable
    const std::string cuda_executable_to_launch = CUDA_MATRIXMULTIPLICATION_V2_ABSOLUTE; 
    
    const std::string input_graph_file = "graph_10k.g"; // This file should be in 'graph_file/'
    const std::string gpu_model = "RTX4060Laptop_RIC"; // Name of your GPU model

    // Parameters to test for TILE_SIZE
    std::vector<int> tile_sizes = {4, 6, 8, 16, 32}; 

    // Parameters to test for TRACE_BLOCKSIZE
    std::vector<int> trace_blocksizes = {32, 64, 128, 256, 512};

    // --- Execute Cross-Validation ---
    std::cout << "\nStarting parameter sweep...\n";

    for (int ts : tile_sizes) {
        // Optional: check maximum threads per block limits of your GPU (e.g., 1024)
        if (ts * ts > 1024) { 
            std::cout << "Skipping TILE_SIZE=" << ts << " (threads per block: " << ts*ts << ") as it exceeds typical GPU limits (1024).\n";
            continue;
        }
        
        for (int tbs : trace_blocksizes) {
            // Construct only the arguments string. The executable path is passed separately.
            std::string arguments = input_graph_file + " " +
                                    std::to_string(ts) + " " +
                                    std::to_string(tbs) + " " +
                                    gpu_model;
            
            // Call executeCommand with the absolute path and arguments
            int exit_code = executeCommand(cuda_executable_to_launch, arguments);
            if (exit_code != 0) {
                std::cerr << "Error: CUDA executable returned non-zero exit code (" << exit_code << ") for TILE_SIZE=" 
                          << ts << ", TRACE_BLOCKSIZE=" << tbs << ". Check CUDA code for errors.\n";
                // Consider if you want to abort or continue in case of an error here
            }
        }
    }

    std::cout << "\nParameter sweep completed. Analyzing results...\n";

    // --- Analyze CSV File and Find Best Parameters ---
    // Use the resolved absolute base path for the CSV file
    std::string output_csv_path = CUDA_MATRIXMULTIPLICATION_CV_BASEPATH_ABSOLUTE + "cross_validation_output_" + gpu_model + ".csv";
    std::ifstream input_csv_file(output_csv_path);
    if (!input_csv_file.is_open()) {
        std::cerr << "Error: Could not open CSV file for reading at " << output_csv_path << ".\n";
        return 1;
    }

    std::string line;
    std::getline(input_csv_file, line); // Skip the header

    ResultEntry best_result;
    best_result.totalDurationUs = std::numeric_limits<long long>::max(); // Initialize with a very large value

    while (std::getline(input_csv_file, line)) {
        std::stringstream ss(line);
        std::string segment;
        ResultEntry current_result;

        // Parsing the CSV line with error checking
        try {
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing TILE_SIZE"); current_result.tileSize = std::stoi(segment);
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing TRACE_BLOCKSIZE"); current_result.traceBlockSize = std::stoi(segment);
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing GPU_MODEL"); current_result.gpuModel = segment;
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing MM_DURATION_US"); current_result.mmDurationUs = std::stoll(segment);
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing TRACE_DURATION_US"); current_result.traceDurationUs = std::stoll(segment);
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing TOTAL_DURATION_US"); current_result.totalDurationUs = std::stoll(segment);
            if (!std::getline(ss, segment, ',')) throw std::runtime_error("Missing TRIANGLES"); current_result.triangles = std::stoi(segment);

            if (current_result.totalDurationUs < best_result.totalDurationUs) {
                best_result = current_result;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing CSV line: '" << line << "' - " << e.what() << std::endl;
        }
    }

    input_csv_file.close();

    std::cout << "\n--- Optimization Results ---\n";
    if (best_result.totalDurationUs == std::numeric_limits<long long>::max()) {
        std::cout << "No valid results found. Check CSV file and execution logs.\n";
    } else {
        std::cout << "Best TILE_SIZE: " << best_result.tileSize << std::endl;
        std::cout << "Best TRACE_BLOCKSIZE: " << best_result.traceBlockSize << std::endl;
        std::cout << "Minimum Total Duration: " << best_result.totalDurationUs << " microseconds\n";
        std::cout << "GPU Model: " << best_result.gpuModel << std::endl;
        std::cout << "Triangles Count (from best run): " << best_result.triangles << std::endl;
    }

    std::cout << "\nOptimization complete.\n";

    return 0;
}