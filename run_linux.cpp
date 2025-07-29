#include <iostream>
#include <cstdlib>
#include <unistd.h> 
#include <string>
#include <array>

// paths
const std::string PATH_SEQUENTIAL_NODE_IT = "algorithms/sequential_node_it";
const std::string PATH_SEQUENTIAL_EDGE_IT = "algorithms/sequential_edge_it";
const std::string PATH_PARALLEL_NODE_IT_CPP = "algorithms/parallel_node_it_CPP";
const std::string PATH_PARALLEL_MATRIXMULTIPLICATION_CPP = "algorithms/parallel_matrixmultiplication_CPP";
const std::string PATH_PARALLEL_EDGE_IT_OPENMP = "algorithms/parallel_edge_it_openmp";
const std::string PATH_PARALLEL_EDGE_IT_MANUAL_THREADS_CPP = "algorithms/parallel_edge_it_manual_threads_CPP";
const std::string PATH_CUDA_NODE_IT = "algorithms/cuda_node_it";
const std::string PATH_CUDA_MATRIXMULTIPLICATION = "algorithms/cuda_matrixmultiplication";
const std::string PATH_CUDA_EDGE_IT = "algorithms/cuda_edge_it";

// graphs
const std::string graph_11 = "graph_11.g";
const std::string graph_100 = "graph_100.g";
const std::string graph_10k = "graph_10k.g";
const std::string graph_100k = "graph_100k.g";
const std::string graph_1ml = "graph_1ml.g";
const std::string graph_2ml = "graph_2ml.g";
const std::string graph_5ml = "graph_5ml.g";
const std::string graph_10ml = "graph_10ml.g";
const std::string graph_100ml = "graph_100ml.g";

std::array<std::string, 9> graph_array = {graph_11, graph_100, graph_10k, graph_100k, graph_1ml, graph_2ml, graph_5ml, graph_10ml, graph_100ml};
std::array<std::string, 9> graph_array_cap_10k = {graph_11, graph_100, graph_10k};

// program versions
const std::string main_v1 = "main_v1";
const std::string main_v1_1 = "main_v1_1";
const std::string main_v1_2 = "main_v1_2";
const std::string main_v2 = "main_v2";
const std::string main_v2_1 = "main_v2_1";
const std::string main_v2_2 = "main_v2_2";

int run_program(std::string program_name, std::string dirPath, std::string graph_file, std::string gpu) {
    int res;
    std::string outDir = "../../";
    std::string command;

    if(chdir(dirPath.c_str()) != 0) {
        std::cerr << "Failed to enter in " << dirPath << '\n';
        return 1;
    } else {
        std::cout << "Enter in " << dirPath << '\n';
    }

    command = "./" + program_name + " " + graph_file + " " + gpu;
    res = std::system(command.c_str());
    if (res != 0) {
        std::cerr << dirPath << " " << program_name << " " << "failed!" << "\n";
        return 1;
    }
    
    if(chdir(outDir.c_str()) != 0) {
        std::cerr << "Failed to exit" << '\n';
        return 1;
    } else {
        std::cout << "Exit" << '\n';
        std::cout << '\n';
    } 

    return 0;

}

int main(int argc, char** argv) {
   
    std::string gpu;

    if (argc != 2) {
        std::cerr << "Wrong number of arguments" << '\n';
        return 1;
    }

    gpu = argv[1];
    
    // RUN
    for (std::string& graph : graph_array_cap_10k) {
        if (run_program(main_v1, PATH_SEQUENTIAL_NODE_IT, graph, gpu) == 1) {
           return 1;
        }
    }

    for (std::string& graph : graph_array) {
        if (run_program(main_v2, PATH_SEQUENTIAL_NODE_IT, graph, gpu) == 1) {
            return 1;
        }
    }

    for (std::string& graph : graph_array_cap_10k) {
        if (run_program(main_v1, PATH_SEQUENTIAL_EDGE_IT, graph, gpu) == 1) {
            return 1;
        }
    }

    for (std::string& graph : graph_array) {
        if (run_program(main_v2, PATH_SEQUENTIAL_EDGE_IT, graph, gpu) == 1) {
            return 1;
        }
    }
}