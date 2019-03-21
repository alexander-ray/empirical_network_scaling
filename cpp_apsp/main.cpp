#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include "all_pairs_shortest_paths.h"
#include "read_edgelist.h"

int main(int argc, char* argv[]) {
    // Command line args
    if (argc != 4) return -1;
    std::string edgelist_fn = argv[1];
    auto zero_indexed = std::strtoul(argv[2], NULL, 10);
    //auto num_nodes = std::stoi(argv[3]);
    auto num_threads = std::stoi(argv[3]);
    std::map<int, std::vector<int>> g = read_edgelist(edgelist_fn, zero_indexed);
    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::vector<int>>> components = get_components(g);

    unsigned long long sum_distances = 0;
    unsigned long long num_distances = 0;
    unsigned long long distance_index = 0;
    for (auto const &component : components) {
        num_distances += component.size() * component.size();
    }

    // Return early if empty network
    if (num_distances == 0) {
        std::cout << 0.0 << std::endl;
        return 0;
    }
    omp_set_num_threads(num_threads);

    unsigned long long print_threshold = num_distances/1000;
    //std::cout << "print thres " << print_threshold << std::endl;
    for (auto const &component : components) {
        auto comp_size = component.size();
        if (comp_size == 1) {
            distance_index++;
            //if (distance_index % print_threshold == 0)
            //    std::cout << distance_index << std::endl;
            continue;
        }
        std::vector<std::vector<unsigned long>> trackers(num_threads, std::vector<unsigned long>(comp_size, 0));
        std::vector<std::vector<unsigned long>> dists(num_threads, std::vector<unsigned long>(comp_size));
        std::vector<std::vector<unsigned long>> visiteds(num_threads, std::vector<unsigned long>(comp_size));
        #pragma omp parallel for shared(component, comp_size, sum_distances, print_threshold, distance_index, trackers, dists, visiteds)
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
                    //if (distance_index % print_threshold == 0)
                    //    std::cout << distance_index << std::endl;
                }
            }
        }
    }
    double mgd = sum_distances/(double)num_distances;;
    std::cout << std::setprecision(6) << mgd << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double,std::milli> elapsed = end - start;
    //std::cout << elapsed.count()/1000 << '\n';
    return 0;
}
