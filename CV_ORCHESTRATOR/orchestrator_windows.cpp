#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <locale>       // For std::locale
#include <codecvt>
#include <array>


using namespace std;

const std::string PATH_SEQUENTIAL_NODE_IT = "../algorithms/sequential_node_it";
const std::string PATH_SEQUENTIAL_EDGE_IT = "../algorithms/sequential_edge_it";
const std::string PATH_PARALLEL_NODE_IT_CPP = "../algorithms/parallel_node_it_CPP";
const std::string PATH_PARALLEL_MATRIXMULTIPLICATION_CPP = "../algorithms/parallel_matrixmultiplication_CPP";
const std::string PATH_PARALLEL_EDGE_IT_OPENMP = "../algorithms/parallel_edge_it_openmp";
const std::string PATH_PARALLEL_EDGE_IT_MANUAL_THREADS_CPP = "../algorithms/parallel_edge_it_manual_threads_CPP";
const std::string PATH_CUDA_NODE_IT = "../algorithms/cuda_node_it";
const std::string PATH_CUDA_MATRIXMULTIPLICATION = "../algorithms/cuda_matrixmultiplication";
const std::string PATH_CUDA_EDGE_IT = "../algorithms/cuda_edge_it";

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

std::array<std::string, 8> graph_array = {graph_100, graph_10k, graph_100k, graph_1ml, graph_2ml, graph_5ml, graph_10ml, graph_100ml};
std::array<std::string, 2> graph_array_cap_10k = {graph_100, graph_10k};

// program versions
const std::string main_v1 = "main_v1.exe";
const std::string main_v1_1 = "main_v1_1.exe";
const std::string main_v1_2 = "main_v1_2.exe";
const std::string main_v2 = "main_v2.exe";
const std::string main_v2_1 = "main_v2_1.exe";
const std::string main_v2_2 = "main_v2_2.exe";

//thread options array
std::array<int, 7> numThreads = { 2, 4, 6, 8, 10, 16, 24 };
//CUDA block sizes array
std::array<int, 7> blockSizes = { 8, 16, 32, 64, 128, 256, 512 };
//CUDA TRACE_BLOCKSIZE array
std::array<int, 7> TileSize = { 8, 16, 32, 64, 128, 256, 512 };
//CUDA MAX_SHARED_LIST_PER_EDGE_COMBINED array
std::array<int, 7> maxSharedListPerEdgeCombined = { 4, 8, 16, 32, 64, 128, 256 };
//CUDA Desired Launches for CUDA EDGE V1_1
std::array<int, 7> desiredLaunches = { 200, 400, 800, 1000, 1200, 1500, 2000 };



//function to convert std::string (UTF-8) to std::wstring (UTF-16)
//This conversion is crucial for Windows API functions when compiled for Unicode.
std::wstring s2ws(const std::string& str) {
    try {
        using convert_type = std::codecvt_utf8_utf16<wchar_t>;
        std::wstring_convert<convert_type, wchar_t> converter;
        return converter.from_bytes(str);
    } catch (const std::range_error& e) {
        // Handle conversion errors (e.g., invalid UTF-8 sequence)
        std::wcerr << L"Error converting string to wstring: " << e.what() << std::endl;
        return std::wstring(); // Return empty or handle as appropriate
    }
}



int executeWindowsProcess(const string &path, const string &args){

    //convert the path to wstring (UTF-16) for Windows API compatibility
    wstring wPath = s2ws(path);
    wstring wArgs = s2ws(args);

    //1st element is the path of theb executable
    //if it contains some spaces, it must be enclosed in quotes to escape them
    wstring fullCmd_w;
    if (wPath.find(L' ') != std::wstring::npos) { // Note L' ' for wide char space
        fullCmd_w = L"\"" + wPath + L"\""; // Note L"" for wide string literals
    } else {
        fullCmd_w = wPath;
    }

    // argv1,2.... are the params
    if (!wArgs.empty()) {
        fullCmd_w += L" " + wArgs;
    }

    //convert string to LPSTR (LPSTR is a pointer to a null-terminated string)
    //use a buffer since CreateProcess expects a writable string
    vector<wchar_t> fullCmdBuffer(fullCmd_w.begin(), fullCmd_w.end());
    //null-terminate the string
    fullCmdBuffer.push_back(L'\0');

    //create STARTUPINFO, PROCESS_INFORMATION structures for Windows API
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;

    //initialize the STARTUPINFO structure
    ZeroMemory(&si, sizeof(si));
    //set the size of the structure
    si.cb = sizeof(si);
    //initialize the PROCESS_INFORMATION structure
    ZeroMemory(&pi, sizeof(pi));

    // Extract the directory from wPath to set as current directory for the child process
    std::wstring workingDir_w = wPath;
    size_t lastSlashPos = workingDir_w.find_last_of(L"\\/");
    if (lastSlashPos != std::wstring::npos) {
        workingDir_w = workingDir_w.substr(0, lastSlashPos);
    } else {
        workingDir_w = L"."; // If no path separator, assume current directory
    }


    //create and start the process
    //current directory is set to NULL, so it gets set to the directory of the executable
    if (!CreateProcessW(      //CreateProcessW -> explicitly use the wide-character version
        NULL,                 // lpApplicationName: NULL meaning we pass the command line directly
        fullCmdBuffer.data(), // lpCommandLine
        NULL,                 // lpProcessAttributes
        NULL,                 // lpThreadAttributes
        FALSE,                // bInheritHandles: don't inherit handles
        0,                    // dwCreationFlags: creation flags (0 for default)
        NULL,                 // lpEnvironment: use the parent's environment
        workingDir_w.c_str(), // lpCurrentDirectory: SET WORKING DIRECTORY NOT OF ORCHESTRATOR BUT OF THE EXECUTABLE
        &si,                  // lpStartupInfo: Information about how to start the process
        &pi                   // lpProcessInformation: Information about the created process
    )) {
        std::wcerr << L"CreateProcessW failed (" << GetLastError() << L").\n";
        return 1;
    } 


    std::wcout << L"Started Process with PID: " << pi.dwProcessId << L" fullCmd: " << fullCmd_w << std::endl;
    //then, wait for completion of the started process
    WaitForSingleObject(pi.hProcess, INFINITE);

    //catch the exit code of the process
    DWORD exitCode;
    if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
        wcout << L"Process " << pi.dwProcessId << L" exited with code: " << exitCode << endl;
    } else {
        wcerr << L"Process " << pi.dwProcessId << L" failed to get exit code (" << GetLastError() << L").\n";
    }

    //close process and thread handles
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return 0;
}

int main(int argc, char** argv) {
    
    if (argc != 2) {
        std::cerr << "Wrong number of arguments. Correct Usage: " + std::string(argv[0]) + " <GPU_MODEL>" << '\n';
        return 1;
    }

    std::string gpu = argv[1];

    int numPhases = 0; //current phase

    /*

    //RUN SEQUENTIALLY EVERY PROGRAM WITH EVERY PARAM

    //Node V1
    for (const std::string& graph : graph_array_cap_10k) {
        if (executeWindowsProcess(PATH_SEQUENTIAL_NODE_IT + "/" + main_v1, graph + " " + gpu) != 0) {
            return 1;
        }
    }

    cout << "Phase " << ++numPhases << " completed: Sequential Node V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //Node V2
    for (const std::string& graph : graph_array) {
        if (executeWindowsProcess(PATH_SEQUENTIAL_NODE_IT + "/" + main_v2, graph + " " + gpu) != 0) {
            return 1;
        }
    }

    cout << "Phase " << ++numPhases << " completed: Sequential Node V2 Iteration with all graphs." << endl << endl;
    
    //Edge V1
    for (const std::string& graph : graph_array_cap_10k) {
        if (executeWindowsProcess(PATH_SEQUENTIAL_EDGE_IT + "/" + main_v1, graph + " " + gpu) != 0) {
            return 1;
        }
    }

    cout << "Phase " << ++numPhases << " completed: Sequential Edge V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //Edge V2
    for (const std::string& graph : graph_array) {
        if (executeWindowsProcess(PATH_SEQUENTIAL_EDGE_IT + "/" + main_v2, graph + " " + gpu) != 0) {
            return 1;
        }
    }

    cout << "Phase " << ++numPhases << " completed: Sequential Edge V2 Iteration with all graphs." << endl << endl;


    //Parallel Node CPP V1
    for (const std::string& graph : graph_array_cap_10k) {
        for (int threads : numThreads) {
            if (executeWindowsProcess(PATH_PARALLEL_NODE_IT_CPP + "/" + main_v1, graph + " " + std::to_string(threads) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: Parallel Node V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //Parallel Node CPP V2
    for (const std::string& graph : graph_array) {
        for (int threads : numThreads) {
            if (executeWindowsProcess(PATH_PARALLEL_NODE_IT_CPP + "/" + main_v2, graph + " " + std::to_string(threads) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: Parallel Node V2 Iteration with all graphs." << endl << endl;

    //Parallel Edge CPP V1
    for (const std::string& graph : graph_array_cap_10k) {
        for (int threads : numThreads) {
            if (executeWindowsProcess(PATH_PARALLEL_EDGE_IT_MANUAL_THREADS_CPP + "/" + main_v1, graph + " " + std::to_string(threads) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: Parallel Edge V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //Parallel Edge CPP V2
    for (const std::string& graph : graph_array) {
        for (int threads : numThreads) {
            if (executeWindowsProcess(PATH_PARALLEL_EDGE_IT_MANUAL_THREADS_CPP + "/" + main_v2, graph + " " + std::to_string(threads) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: Parallel Edge V2 Iteration with all graphs." << endl << endl;

    //parallel Edge OpenMP V1 -> TODO

    //Parallel Matrix Multiplication CPP V1
    for (const std::string& graph : graph_array_cap_10k) {
        for (int threads : numThreads) {
            if (executeWindowsProcess(PATH_PARALLEL_MATRIXMULTIPLICATION_CPP + "/" + main_v1, graph + " " + std::to_string(threads) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: Parallel Matrix Multiplication V1 Iteration with graphs up to 10k nodes." << endl << endl;


    
    //CUDA Node V1, JUST BLOCKSIZE
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            if (executeWindowsProcess(PATH_CUDA_NODE_IT + "/" + main_v1, graph + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Node V1 Iteration with graphs up to 10k nodes." << endl << endl;
    
    //CUDA NODE V2, JUST BLOCKSIZE
    for (const std::string& graph : graph_array) {
        for (int blockSize : blockSizes) {
            if (executeWindowsProcess(PATH_CUDA_NODE_IT + "/" + main_v2, graph + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Node V2 Iteration with all graphs." << endl << endl;
    

    //CUDA EDGE V1, JUST BLOCKSIZE
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v1, graph + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                return 1;
            }
        }
    }

    cout << "Phase " << ++numPhases << " completed: CUDA Edge V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //CUDA EDGE V1_1, BLOCKSIZE AND DESIRED LAUNCHES
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            for (int desiredLaunch : desiredLaunches) {
                if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v1_1, graph + " " + std::to_string(blockSize) + " " + std::to_string(desiredLaunch) + " " + gpu) != 0) {
                    return 1;
                }
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Edge V1_1 Iteration with graphs up to 10k nodes." << endl << endl;

    

    //CUDA EDGE V1_2, JUST BLOCKSIZE
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            for (int desiredLaunch : desiredLaunches) {
                if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v1_2, graph + " " + std::to_string(blockSize) + " " + std::to_string(desiredLaunch) + " " + gpu) != 0) {
                    return 1;
                }
            }
        }
    }

    cout << "Phase " << ++numPhases << " completed: CUDA Edge V1_2 Iteration with graphs up to 10k nodes." << endl << endl;

    */

    //CUDA EDGE V2, JUST BLOCKSIZE
    for (const std::string& graph : graph_array) {
        for (int blockSize : blockSizes) {
            if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v2, graph + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Edge V2 Iteration with all graphs." << endl << endl;

    //CUDA EDGE V2_1, BLOCKSIZE AND MAX_SHARED_LIST_PER_EDGE_COMBINED
    for (const std::string& graph : graph_array) {
        for (int blockSize : blockSizes) {
            for (int maxSharedList : maxSharedListPerEdgeCombined) {
                if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v2_1, graph + " " + std::to_string(blockSize) + " " + std::to_string(maxSharedList) + " " + gpu) != 0) {
                    return 1;
                }
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Edge V2_1 Iteration with all graphs." << endl << endl;

    //CUDA EDGE V2_2, BLOCKSIZE AND MAX_SHARED_LIST_PER_EDGE_COMBINED
    for (const std::string& graph : graph_array) {
        for (int blockSize : blockSizes) {
            for (int maxSharedList : maxSharedListPerEdgeCombined) {
                if (executeWindowsProcess(PATH_CUDA_EDGE_IT + "/" + main_v2_2, graph + " " + std::to_string(blockSize) + " " + std::to_string(maxSharedList) + " " + gpu) != 0) {
                    return 1;
                }
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Edge V2_2 Iteration with all graphs." << endl << endl;


    //CUDA Matrix Multiplication V1, JUST BLOCKSIZE
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            if (blockSize > 32)
                continue;
            if (executeWindowsProcess(PATH_CUDA_MATRIXMULTIPLICATION + "/" + main_v1, graph + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                return 1;
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Matrix Multiplication V1 Iteration with graphs up to 10k nodes." << endl << endl;

    //CUDA Matrix Multiplication V2, BLOCKSIZE AND TILESIZE
    for (const std::string& graph : graph_array_cap_10k) {
        for (int blockSize : blockSizes) {
            if (blockSize > 32)
                continue;
            for (int tileSize : TileSize) {
                if (tileSize > 32)
                    continue;
                if (executeWindowsProcess(PATH_CUDA_MATRIXMULTIPLICATION + "/" + main_v2, graph + " " + std::to_string(tileSize) + " " + std::to_string(blockSize) + " " + gpu) != 0) {
                    return 1;
                }
            }
        }
    }
    cout << "Phase " << ++numPhases << " completed: CUDA Matrix Multiplication V2 Iteration with graphs up to 10k nodes." << endl << endl;

    

    return 0;
}