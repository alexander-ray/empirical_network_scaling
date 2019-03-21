//
// Created by Alex Ray on 2018-12-16.
//

#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <array>
#include <unordered_set>
#include "all_pairs_shortest_paths.h"

void dfs_helper(unsigned long u,
                std::vector<int> &visited,
                std::map<int, std::vector<int>> &component,
                std::map<int, std::vector<int>> &g) {
    std::stack<unsigned long> s;
    s.push(u);
    
    while (!s.empty()) {
        u = s.top();
        s.pop();
        
        if (!visited[u]) {
            visited[u] = 1;
            component[u] = g[u];
        }
        for (auto const &v : g[u]) {
            if (!visited[v]) {
                s.push(v);
            }
        }
    }
}

std::vector<std::vector<int>> relabel_component(std::map<int, std::vector<int>> &g) {
    std::map<int, int> mapping;
    std::vector<std::vector<int>> relabeled(g.size());
    auto label_counter = 0;
    for (const auto &kv : g) {
        auto u = kv.first;
        auto vs = kv.second;
        if (mapping.find(u) == mapping.end()) {
            mapping[u] = label_counter;
            label_counter++;
        }
        for (auto const &v : vs) {
            if (mapping.find(v) == mapping.end()) {
                mapping[v] = label_counter;
                label_counter++;
            }
            relabeled[mapping[u]].push_back(mapping[v]);
        }
    }
    return relabeled;
}

// Assumes node labels in range 0..g.size()-1
// In other words, edgelist should be generated from igraph (which uses indices instead of labels)
std::vector<std::vector<std::vector<int>>> get_components(std::map<int, std::vector<int>> &g) {
    std::vector<int> visited(g.size(), 0);
    std::vector<std::vector<std::vector<int>>> components;
    for (unsigned long i = 0; i < g.size(); i++) {
        if (!visited[i]) {
            std::map<int, std::vector<int>> component;
            dfs_helper(i, visited, component, g);
            components.push_back(relabel_component(component));
        }
    }
    return components;
}

std::vector<unsigned long> bfs_distances(const std::vector<std::vector<int>> &g,
                               unsigned long starting_node,
                               unsigned long num_nodes,
                               std::vector<unsigned long> &visited,
                               std::vector<unsigned long> &dist,
                               std::vector<unsigned long> &tracker) {
    auto pos = 0;
    auto end = 1;
    unsigned long num_visited = 0;
    for (unsigned int i = 0; i < num_nodes; i++) {
        visited[i] = 0;
        dist[i] = 0;
    }

    tracker[pos] = starting_node;

    visited[starting_node] = 1;
    dist[starting_node] = 0;
    num_visited++;

    while (pos <= end && num_visited < num_nodes) {
        unsigned long u = tracker[pos++];
        for (auto const &v : g[u]) {
            if (!visited[v]) {
                visited[v] = 1;
                num_visited += 1;
                dist[v] = dist[u] + 1;
                tracker[end++] = v;
            }
        }
    }

    return dist;
}
