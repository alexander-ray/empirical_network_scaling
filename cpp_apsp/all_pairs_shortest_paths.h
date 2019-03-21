//
// Created by Alex Ray on 2018-12-16.
//
#ifndef UNTITLED_ALL_PAIRS_SHORTEST_PATHS_H
#define UNTITLED_ALL_PAIRS_SHORTEST_PATHS_H

#include <vector>
#include <map>

void dfs_helper(unsigned long u,
                std::vector<int> &visited,
                std::map<int, std::vector<int>> &component,
                std::map<int, std::vector<int>> &g);
std::vector<std::vector<int>> relabel_component(std::map<int, std::vector<int>> &g);
std::vector<std::vector<std::vector<int>>> get_components(std::map<int,
                                                             std::vector<int>> &g);
std::vector<unsigned long> bfs_distances(const std::vector<std::vector<int>> &g,
                               unsigned long starting_node,
                               unsigned long num_nodes,
                               std::vector<unsigned long> &visited,
                               std::vector<unsigned long> &dist,
                               std::vector<unsigned long> &tracker);
#endif //UNTITLED_ALL_PAIRS_SHORTEST_PATHS_H
