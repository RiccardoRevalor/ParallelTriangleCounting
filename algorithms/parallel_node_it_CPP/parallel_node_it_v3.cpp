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
#include <unordered_map>
#include <future>

#define DEBUG 0


using namespace std;


void createOrderedList(const unordered_map<int, vector<int>> &adjacencyVectors, vector<int> &orderedList){
    //create a map to store the degree of each node, then sort it
    unordered_map<int, int> nodeDegree;
    for (const auto &keyvaluepair: adjacencyVectors) {
        int node = keyvaluepair.first;
        int degree = keyvaluepair.second.size();
        nodeDegree[node] = degree;
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


long long threadFunc(const vector<int>& orderedList, const unordered_map<int, vector<int>>& adj, const unordered_map<int, int>& ranks, int start, int end){
    //thread function to count triangles inside a partition of nodes of the graph
    //then return  the local count of triangles
    //with unordered map as adjancency list and ranks, we have an average complexity of O(1) for each access (O(n) in the worst case)
    long long localCount = 0;

    //cycle through each node s inside the partition
    for (int i = start; i < end; i++){
        int s = orderedList[i];
        
        //retruieve the neighbors of s
        const vector<int>& s_neighbors = adj.at(s);
        //optimization: skip nodes with no neighbors
        if (s_neighbors.empty()) continue;

        //cycle over each neighbor t of s
        for (int t : s_neighbors) {
            //Make sure we only form triangles (s, t, v) where rank(s) < rank(t) < rank(v)
            if (ranks.at(s) >= ranks.at(t)) {
                continue;
            }

            //neighbors of t
            const vector<int>& t_neighbors = adj.at(t);

            //Merge Like: fast intersection of two sorted neighbor lists (s_neighbors and t_neighbors)
            auto it_s = s_neighbors.begin();
            auto it_t = t_neighbors.begin();

            while (it_s != s_neighbors.end() && it_t != t_neighbors.end()) {
                if (*it_s < *it_t) {
                    ++it_s;
                } else if (*it_t < *it_s) {
                    ++it_t;
                } else { //if *it_s == *it_t, we found a common neighbor 'v'
                    int v = *it_s;
                    //make sure we only count triangles once by checking rank order
                    if (ranks.at(t) < ranks.at(v)) {
                        localCount++;
                    }
                    ++it_s;
                    ++it_t;
                }
            }
        }

    }


    return localCount;
    
}


void forwardAlgorithmParallel(const vector<int>& orderedList, unordered_map<int, vector<int>>& adj, int numThreads,long long &countTriangles, long long &duration) {
    //Optimization 1: initialize ranks only once, here, and use unordered_map for faster access
    unordered_map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }

    //Optimization 2: sort adjacency list to prepare it for fast intersection inside thread function (merge like)
    for (auto& keyvaluepair : adj) {
        sort(keyvaluepair.second.begin(), keyvaluepair.second.end());
    }

    //Use tasks and let the OS manage the scheduling
    const int num_threads = numThreads;
    vector<future<long long>> futures;
    int chunk_size = orderedList.size() / num_threads;


    auto startTime = chrono::high_resolution_clock::now();


    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = min(start + chunk_size, (int)orderedList.size());

        futures.emplace_back(async(launch::async, threadFunc,
                                ref(orderedList), ref(adj), ref(ranks), start, end));
    }

    countTriangles = 0;
    for (auto& f : futures) {
        countTriangles += f.get();
    }

    auto endTime = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();

}


int main(int argc, char **argv){

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

    //Optimization: unordered_map for adjacency list, faster access (O(1) on average meanwhile normal map has O(log n) on average)
    unordered_map<int, vector<int>> adjacencyVectors = populateAdjacencyVectorsUnordered(input);


    //create ordered list of nodes based on degree
    vector<int> orderedList;
    createOrderedList(adjacencyVectors, orderedList);
    if (DEBUG) {
        cout << "Ordered list of nodes based on degree:\n";
        for (const auto &node : orderedList) {
            cout << node << " ";
        }
        cout << "\n";
    }

    cout << "-----------------------------------------------------------------" << endl;
    long long countTriangles = 0;
    long long duration = 0;

    forwardAlgorithmParallel(orderedList, adjacencyVectors, numThreads, countTriangles, duration);

    cout << "Time taken for forward algorithm: " << duration<< " microseconds" << endl;
    cout << "Triangles found by forward algorithm: " << countTriangles << endl;

    //write to output file
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
    string outputFileName("../../cross_validation_output/parallel_node_it_v3/" + input + "_" + gpuModel + ".csv");
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
                      << duration << ","
                      << countTriangles << "\n";

    crossValidationFile.close();

    return 0;
}