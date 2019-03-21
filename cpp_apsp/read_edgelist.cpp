//
// Created by Alex Ray on 2018-12-16.
//

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "read_edgelist.h"

std::map<int, std::vector<int>> read_edgelist(std::string filename, int zero_indexed) {
    std::ifstream infile(filename);

    std::string num_nodes_str;
    std::getline(infile, num_nodes_str);
    long num_nodes = std::stoul(num_nodes_str, NULL, 10);

    int a, b;

    std::map<int, std::vector<int>> g;
    while (infile >> a >> b) {
        // FB100 doesn't include 0 as node label
        if (!zero_indexed) {
            a--;
            b--;
        }
        // Don't add self-loop
        if (a == b) continue;

        if (std::find(g[a].begin(), g[a].end(), b) == g[a].end())
            g[a].push_back(b);
        if (std::find(g[b].begin(), g[b].end(), a) == g[b].end())
            g[b].push_back(a);
    }
    if (num_nodes == -1) return g;
    
    // fill in singleton nodes
    for (long i = 0; i < num_nodes; i++) {
        if (!g.count(i)) {
            g[i] = std::vector<int>();
        }
    }
    return g;
}

