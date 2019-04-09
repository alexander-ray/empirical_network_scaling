# Empirical Network Scaling Analysis
## Overview
This is the repo for a large-scale analysis of empirical scaling behavior in networks. Primarily, this code includes methods for calculating mean geodesic distance of large networks, estimating mean geodesic distance through sampling methods, and uniformly sampling from the space of graphs with a fixed degree sequence. 
## Definitions
### Mean Geodesic Distance
We define mean geodesic distance as 

![Alt text](images/mgd.png?raw=true "Mean Geodesic Distance")

which defines the average shortest path length even with disconnected components. 
### Global Clustering Coefficient
We define global clustering coefficient as the ratio between three times the number of triangles in the network and the number of connected triples. We use iGraph's [implementation](https://igraph.org/python/doc/igraph.GraphBase-class.html#transitivity_undirected) to calculate this value.
## Code Components
### `cpp_apsp`
`cpp_apsp` includes the code, headers, and tests for a parallelized C++ implementation of all pairs shortest path for a given network. For simplicity this code takes an edgelist (with number of nodes on the first line) and processes the data into adjacency list graph representations for each connected component. Mean geodesic distance is then calculated, using [OpenMP](https://www.openmp.org/) to run breadth first searches in parallel.

Tests for this subpackage were the same as tests for python APSP implementation, adapted to run via a shell script and input files.
### `py_tools`
`py_tools` includes the code and tests for all the worker functions required for general computation. This includes MCMC methods, pairwise path length sampling, I/O and graph format conversion methods, and exact BFS implementations in a number of different forms. 

Tests are included in the `tests` subdirectory. Broadly, BFS tests rely on exact mathematical computations or matches to iGraph or Networkx implementations. MCMC tests revolve around setting a random seed and comparing our implementation with the [implementation](https://github.com/joelnish/double-edge-swap-mcmc/) of Fosdick et al. 
## Contact
Please direct questions to <Alexander.W.Ray@colorado.edu>. 
