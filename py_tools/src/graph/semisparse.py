import networkx as nx
import numpy as np
import numba as nb
jit = nb.jit


def all_pairs_shortest_paths(G):
    """
    Driver function for fast APSP algorithm on simple graph with multiple disconnected components

    :param G: nx.Graph
    :return: 1d ndarray of size \sum_{i}n_i^2 where n_i is the number of nodes in the ith connected component
    """
    if len(G) <= 1:
        return np.array([])
    if nx.is_connected(G):
        return _all_pairs_shortest_paths_preprocessor(G, rolling_sum=False)
    else:
        results = []
        gen = (G.subgraph(c) for c in nx.connected_components(G))
        for g in gen:
            res = _all_pairs_shortest_paths_preprocessor(g, rolling_sum=False)
            if res is not None:
                results.append(res)
        return np.concatenate(results)


def all_pairs_shortest_paths_rolling_sum(G):
    """
    Driver function for fast APSP algorithm on simple graph with multiple disconnected components.
    Does a rolling sum instead of maintaining an nxn distance matrix

    :param G: nx.Graph
    :return: Tuple, (num distances, sum distances)
    """
    if len(G) <= 1:
        return 1, 0
    if nx.is_connected(G):
        return len(G)**2, _all_pairs_shortest_paths_preprocessor(G, rolling_sum=True)
    else:
        num_distances = 0
        sum_distances = 0
        gen = (G.subgraph(c) for c in nx.connected_components(G))
        for g in gen:
            if len(g) == 1:
                # Add single node to denomenator
                num_distances += 1
                continue
            res = _all_pairs_shortest_paths_preprocessor(g, rolling_sum=True)
            # Increase num distances by size of comp squared
            num_distances += len(g) ** 2
            sum_distances += res
        return num_distances, sum_distances


def _all_pairs_shortest_paths_preprocessor(G, rolling_sum=True):
    """
    Helper function for APSP on single connected component

    :param G: nx.Graph
    :return: 1d ndarray of size n^2 where n is the number of nodes in the connected component
    """
    num_nodes = len(G)
    node_list = list(range(len(G)))
    max_degree = max(G.degree, key=lambda x: x[1])[1]

    tracker = np.full(num_nodes, -1, dtype=np.int64)
    G_numpy = _to_semisparse_matrix(G, num_nodes, max_degree)
    if rolling_sum:
        return _all_pairs_shortest_paths_rolling_sum_driver(G_numpy, num_nodes, node_list, max_degree, tracker)
    else:
        return _all_pairs_shortest_paths_driver(G_numpy, num_nodes, node_list, max_degree, tracker)


def _to_semisparse_matrix(G, num_nodes, max_degree):
    """
    Creates "semisparse" matrix as a compromise between adjacency matrix and adjacency list.
    Matrix is n x max_degree, where elements represent the node ids.

    :param G: Connected nx.Graph
    :param num_nodes: Number of nodes
    :param max_degree: Largest degree in connected component
    :return: n x max_degree ndarray
    """
    if num_nodes == 1:
        return np.array([0])

    G = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight=None, format='lil')

    def generate_rows(rows):
        for row in rows:
            yield np.pad(row, (0, max_degree - len(row)), 'constant', constant_values=-1)

    return np.array(np.stack(list(generate_rows(G.rows))), dtype=np.int64)


@jit(nopython=True, nogil=True)
def _all_pairs_shortest_paths_driver(G, num_nodes, node_list, max_degree, tracker):
    """
    Jitted driver function for single-component APSP

    :param G: Semisparse matrix
    :param num_nodes: Number of nodes in component
    :param node_list: List of nodes
    :param max_degree: Max degree of nodes
    :param tracker: Tracker ndarray to maintain neighbor
    :return: 1d ndarray of size n^2
    """
    results = np.zeros(num_nodes**2)
    for i in node_list:
        start = i * num_nodes
        results[start:start+num_nodes] = _bfs_distances(G, i, num_nodes, max_degree, tracker)
    return results


@jit(nopython=True, nogil=True)
def _all_pairs_shortest_paths_rolling_sum_driver(G, num_nodes, node_list, max_degree, tracker):
    """
    Jitted driver function for single-component APSP, returning sum of distances

    :param G: Semisparse matrix
    :param num_nodes: Number of nodes in component
    :param node_list: List of nodes
    :param max_degree: Max degree of nodes
    :param tracker: Tracker ndarray to maintain neighbor
    :return: float
    """
    sum_distances = 0.0
    for i in node_list:
        sum_distances += np.sum(_bfs_distances(G, i, num_nodes, max_degree, tracker))
    return sum_distances


@jit(nopython=True, nogil=True)
def _bfs_distances(G, starting_node, num_nodes, max_degree, tracker):
    """
    Jitted BFS using semisparse matrices

    :param G: Semisparse matrix
    :param starting_node: Source node
    :param num_nodes: Number of nodes in component
    :param max_degree: Max degree of nodes
    :param tracker: Tracker ndarray to maintain neighbor
    :return: 1d distance ndarrray of size n
    """
    pos = 0
    end = 1
    visited = np.zeros(num_nodes)
    dist = np.zeros(num_nodes)
    tracker[pos] = starting_node

    visited[starting_node] = 1
    num_visited = 1

    while not pos > end and num_visited < num_nodes:
        node = tracker[pos]
        pos += 1
        # iterating over neighbors of node
        i = 0
        # Don't go past end of row
        while i < max_degree and G[node, i] != -1:
            val = G[node, i]
            if visited[val] == 0:
                visited[val] = 1
                num_visited += 1
                dist[val] = dist[node] + 1
                tracker[end] = val
                end += 1
            i += 1
    return dist
