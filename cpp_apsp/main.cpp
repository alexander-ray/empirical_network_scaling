#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <omp.h>
extern "C" {
   #include <igraph.h>
}
#include <functional>

#include "all_pairs_shortest_paths.h"
#include "read_graphs.h"

std::vector<std::string> get_filenames(std::string filename) {
    std::ifstream infile(filename);
    std::string name;

    std::string appendee = "/scratch/Users/alra7284/gmls/Anna/";
    //std::string appendee = "";
    std::vector<std::string> filenames;
    while(std::getline(infile, name)) {
        filenames.push_back(appendee + name);
    }
    return filenames;
}

std::map<int, double> get_distance_distribution(std::string filename, int num_threads) {
    std::map<int, std::vector<int>> g = read_gml(filename);
    //std::map<int, std::vector<int>> g = read_edgelist(filename, 1);
    std::cout << "read gml, num nodes: " << g.size() << std::endl;

    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::vector<int>>> components = get_components(g);
    
    std::cout << "num components: " << components.size() << std::endl;
    
    std::map<int, double> distance_distribution;

    unsigned long long sum_distances = 0;
    unsigned long long num_distances = 0;
    unsigned long long distance_index = 0;
    for (auto const &component : components) {
        num_distances += component.size() * component.size();
    }

    // Return early if empty network
    if (num_distances == 0) {
        std::cout << 0.0 << std::endl;
        distance_distribution[0] = 1;
        return distance_distribution;
    }

    unsigned long long print_threshold = num_distances/10;
    std::cout << "print thres " << print_threshold << std::endl;
    for (auto const &component : components) {
        auto comp_size = component.size();
        if (comp_size <= 1) {
            distance_index++;
            if (distance_distribution.count(0) == 0)
                distance_distribution[0] = 1;
            else
                distance_distribution[0]++;
            if (distance_index % print_threshold == 0)
                std::cout << distance_index << std::endl;
            continue;
        }
        std::vector<std::vector<unsigned long>> trackers(num_threads, std::vector<unsigned long>(comp_size, 0));
        std::vector<std::vector<unsigned long>> dists(num_threads, std::vector<unsigned long>(comp_size));
        std::vector<std::vector<unsigned long>> visiteds(num_threads, std::vector<unsigned long>(comp_size));
        #pragma omp parallel for shared(component, comp_size, sum_distances, print_threshold, distance_index, trackers, dists, visiteds, distance_distribution)
        for (unsigned long i = 0; i < comp_size; i++) {
            std::vector<unsigned long> tmp = bfs_distances(component, i, comp_size, 
                                                 dists[omp_get_thread_num()], 
                                                 visiteds[omp_get_thread_num()], 
                                                 trackers[omp_get_thread_num()]);
            #pragma omp critical 
            {
                for (auto const &j : tmp) {
                    sum_distances += j;
                    distance_index++;
                    if (distance_distribution.count(j) == 0)
                        distance_distribution[j] = 1;
                    else
                        distance_distribution[j]++;
                    if (distance_index % print_threshold == 0)
                        std::cout << distance_index << std::endl;
                }
            }
        }
    }
    double mgd = sum_distances/(double)num_distances;;
    std::cout << std::setprecision(6) << mgd << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double,std::milli> elapsed = end - start;
    //std::cout << elapsed.count()/1000 << '\n';

    return distance_distribution;
}

void append_distance_distribution_to_file(std::map<int, double> distance_distribution, std::string output_filename, std::string input_filename) {
    std::fstream fs;
    fs.open(output_filename, std::fstream::in | std::fstream::out | std::fstream::app);

    fs << input_filename;
    for (const auto &kv : distance_distribution) {
        auto dist = kv.first;
        auto value = kv.second;
        fs << std::fixed << ", " << dist << ", " << value;
    }
    fs << "\n";
    fs.close();
}

int main(int argc, char* argv[]) {
    // Command line args
    if (argc != 4) return -1;
    std::string list_filename = argv[1];
    std::string output_filename = argv[2];
    std::vector<std::string> filenames = get_filenames(list_filename);

    for (const auto &fn : filenames)
        std::cout << fn << std::endl;
    
    auto num_threads = std::stoi(argv[3]);
    omp_set_num_threads(num_threads);

    for (const auto &graph_fn : filenames) {
        auto distance_distribution = get_distance_distribution(graph_fn, num_threads);
        append_distance_distribution_to_file(distance_distribution, output_filename, graph_fn);
    }

    return 0;
}
