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

struct Edge{
    int v0;
    int v1;
};

bool operator==(const Edge &e1, const Edge &e2) {
    return (e1.v0 == e2.v0 && e1.v1 == e2.v1) || (e1.v0 == e2.v1 && e1.v1 == e2.v0);
}

namespace std {
    template<>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            // Ordina i nodi per garantire che (u,v) e (v,u) abbiano lo stesso hash.
            int first = min(e.v0, e.v1);
            int second = max(e.v0, e.v1);

            size_t h1 = hash<int>{}(first);
            size_t h2 = hash<int>{}(second);
            
            return h1 ^ (h2 << 1); 
        }
    };
}


vector<Edge> createDirectedEdgeVector(const unordered_map<int, vector<int>>& adj,
    const unordered_map<int, int>& ranks);


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

/**
 * @brief Function to count triangles in a partition of the graph.
 * @param orderedList The ordered list of nodes based on degree.
 * @param adj The adjacency list of the graph.
 * @param ranks A map from node ID to its rank.
 * @param start The starting index for the partition.
 * @param end The ending index for the partition.
 */
long long threadFunction(const unordered_map<int, vector<int>> &adjacencyVectors, const vector<Edge> &edgeVector, const unordered_map<int, int>& ranks, int start, int end) {
    //Optimization: each task updates and returns a local count of triangles
    long long countTriangles = 0;

    //iteration through assigned chunk of edges
    for (int i = start; i < end; ++i) {
        const auto &edge = edgeVector[i];

        int v0 = edge.v0;
        int v1 = edge.v1;

        //no longer check is rank(v0) < rank(v1), because this is already guaranteed by the way we create the edge set
        // Get constant references to pre-sorted neighbor lists. NO COPIES.
        const vector<int>& v0_neighbors = adjacencyVectors.at(v0);
        const vector<int>& v1_neighbors = adjacencyVectors.at(v1);

        //Merge Like algorithm: fast 2 pointer intersection of two sorted neighbor lists
        auto it_v0 = v0_neighbors.begin();
        auto it_v1 = v1_neighbors.begin();

        while (it_v0 != v0_neighbors.end() && it_v1 != v1_neighbors.end()) {
            if (*it_v0 < *it_v1) {
                ++it_v0;
            } else if (*it_v1 < *it_v0) {
                ++it_v1;
            } else { //if *it_v0 == *it_v1, we found a common neighbor 'v'
                int v = *it_v0; //v is the common neighbor of v0 and v1, so the third vertex of the triangle (v0, v1, v)
                //make sure we only count triangles once by checking rank order
                if (ranks.at(v) > ranks.at(v0) && ranks.at(v) > ranks.at(v1)) {
                    countTriangles++;
                }
                ++it_v0;
                ++it_v1;
            }
        }
    }

    return countTriangles;
}

/**
 * @brief Function to run the edge iterator algorithm in parallel. It orchestrates the threads and aggregates results.
 * @param orderedList The ordered list of nodes based on degree.
 * @param adjacencyVectors The adjacency list of the graph.
 * @param numThreads The number of threads to use.
 * @param countTriangles Reference to the total count of triangles found.
 * @param duration Reference to the duration of the algorithm execution.
 */
void ForwardAlgorithmEdgeParallel(const vector<int> &orderedList, const unordered_map<int, vector<int>> &adjacencyVectors, int numThreads,long long &countTriangles, long long &duration){
    //Optimization 1: initialize ranks only once, here, and use unordered_map for faster access
    unordered_map<int, int> ranks;
    for (int i = 0; i < orderedList.size(); ++i) {
        ranks[orderedList[i]] = i;
    }

    //create edge vector
    vector<Edge> edgeVector = createDirectedEdgeVector(adjacencyVectors, ranks);

    //tasks
    int totEdges = edgeVector.size();
    int chunkSize = (totEdges + numThreads - 1) / numThreads; // Round up division
    vector<future<long long>> futures;
    auto startTime = chrono::high_resolution_clock::now();
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunkSize;
        int end = min(start + chunkSize, totEdges);
        //create a task for each thread
        futures.emplace_back(async(launch::async, threadFunction, ref(adjacencyVectors), ref(edgeVector), ref(ranks), start, end));
    }

    // Wait for all tasks to complete and aggregate the results
    for (auto &fut : futures) {
        countTriangles += fut.get();
    }

    auto endTime = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
}



/**
 * @brief Creates a vector of directed edges based on node ranks.
 * * @param adj The graph's adjacency list (use unordered_map for performance).
 * @param ranks A map from node ID to its rank.
 * @return A vector containing only the edges (s, t) where rank(s) < rank(t).
 */
vector<Edge> createDirectedEdgeVector(
    const unordered_map<int, vector<int>>& adj,
    const unordered_map<int, int>& ranks) 
{
    // Return a vector directly. This is more efficient than building a set 
    // and then converting it, as it avoids hashing overhead.
    vector<Edge> directedEdgeVector;

    // Iterate through each node and its neighbors
    for (const auto& pair : adj) {
        int u = pair.first;
        const vector<int>& neighbors = pair.second;

        for (int v : neighbors) {
            // THE CRITICAL PRUNING STEP:
            // Only add the edge if it goes from a lower-rank node to a higher-rank one.
            // This ensures each edge is processed only once and in the correct direction, so we de facto use DIRECTED EDGES.
            if (ranks.at(u) < ranks.at(v)) {
                // Because of the check, we now have a directed edge where the first
                // element has a lower rank than the second. This is the guarantee.
                directedEdgeVector.push_back({u, v});
            }
        }
    }

    return directedEdgeVector;
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

    ForwardAlgorithmEdgeParallel(orderedList, adjacencyVectors, numThreads, countTriangles, duration);

    cout << "Time taken for edge parallel algorithm: " << duration<< " microseconds" << endl;
    cout << "Triangles found by edge parallel algorithm: " << countTriangles << endl;

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
    string outputFileName("../../cross_validation_output/parallel_edge_it_manual_threads_v3/" + input + "_" + gpuModel + ".csv");
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
