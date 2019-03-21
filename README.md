# Empirical Network Scaling Analysis
## Overview
This is the repo for a large-scale analysis of empirical scaling behavior in networks. Primarily, this code includes methods for calculating mean geodesic distance of large networks, estimating mean geodesic distance through sampling methods, and uniformly sampling from the space of graphs with a fixed degree sequence. 
## Definitions
### Mean Geodesic Distance
We define mean geodesic distance as ![Alt text](images/mgd.png?raw=true "Mean Geodesic Distance"), which defines the average shortest path length even with disconnected components. 
## Code Components
### `cpp_apsp`
`cpp_apsp` includes the code, headers, and tests for a parallelized C++ implementation of all pairs shortest path for a given network. For simplicity this code takes an edgelist (with number of nodes on the first line) and processes the data into adjacency list graph representations for each connected component. Mean geodesic distance is then calculated, using [OpenMP](https://www.openmp.org/) to run breadth first searches in parallel.
