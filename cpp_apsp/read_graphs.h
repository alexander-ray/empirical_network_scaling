//
// Created by Alex Ray on 2018-12-16.
//
#ifndef UNTITLED_READ_GRAPHS_H
#define UNTITLED_READ_GRAPHS_H

#include <vector>
#include <map>
std::map<int, std::vector<int>> read_edgelist(std::string filename, int zero_indexed);
std::map<int, std::vector<int>> read_gml(std::string filename);
#endif //UNTITLED_READ_EDGELIST_H
