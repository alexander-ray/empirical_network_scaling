//
// Created by Alex Ray on 2018-12-16.
//

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdio.h>
extern "C" {
    #include <igraph.h>
}
#include "read_graphs.h"

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

std::map<int, std::vector<int>> read_gml(std::string filename) {
    igraph_t g;
    std::map<int, std::vector<int>> g_map;

    FILE *infile;
    infile = fopen(filename.c_str(), "r");

    igraph_read_graph_gml(&g, infile);
    fclose(infile);

    igraph_to_undirected(&g, IGRAPH_TO_UNDIRECTED_COLLAPSE, 0);
    /* CURRENTLY SEGFAULTS, shouldn't be needed anyways. 
    if (igraph_cattribute_has_attr(&g, IGRAPH_ATTRIBUTE_EDGE, "weight"))
        igraph_cattribute_remove_e(&g, "weight");
    */
    igraph_simplify(&g, 1, 1, 0);

    igraph_adjlist_t adj;
    igraph_adjlist_init(&g, &adj, IGRAPH_ALL);
    
    igraph_destroy(&g);

    long vec_size;
    for (long i = 0; i < adj.length; i++) {
        vec_size = igraph_vector_int_size(igraph_adjlist_get(&adj, i));
        for (long j = 0; j < vec_size; j++) {
            g_map[i].push_back(VECTOR(*igraph_adjlist_get(&adj, i))[j]);
        }
        if (!vec_size) {
            g_map[i] = std::vector<int>();
        }
    }
    igraph_adjlist_destroy(&adj);
    return g_map;
}
