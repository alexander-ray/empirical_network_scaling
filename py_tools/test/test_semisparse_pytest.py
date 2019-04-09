import networkx as nx

from src.graph import semisparse as ss


#############################################
# Test: all_pairs_shortest_path_rolling_sum()
#############################################
# Number of distances
def test_empty_num_distances():
    G = nx.Graph()
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == 1  # convention


def test_singleton_num_distances():
    G = nx.Graph()
    G.add_node(1)
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == 1  # convention


def test_single_edge_num_distances():
    G = nx.Graph()
    G.add_edge(1, 2)
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == len(G)**2


def test_disconnected_triads_num_distances():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == 3**2 + 3**2


def test_three_components_num_distances():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_node(9)
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == 3**2 + 3**2 + 1


def test_three_components_num_distances_2():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_edge(9, 10)
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == 3**2 + 3**2 + 2**2


def test_zkc_num_distances():
    G = nx.karate_club_graph()
    num_distances, _ = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert num_distances == len(G)**2


# Sum of distances
def test_empty_sum_distances():
    G = nx.Graph()
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 0


def test_singleton_sum_distances():
    G = nx.Graph()
    G.add_node(1)
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 0


def test_single_edge_sum_distances():
    G = nx.Graph()
    G.add_edge(1, 2)
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 2


def test_disconnected_triads_sum_distances():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 12


def test_three_components_sum_distances():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_node(9)
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 12


def test_three_components_sum_distances_2():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_edge(9, 10)
    _, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances == 14


def test_zkc_sum_distances():
    G = nx.karate_club_graph()
    tup = ss.all_pairs_shortest_paths_rolling_sum(G)
    _, sum_distances = tup
    assert sum_distances == sum([sum(x.values()) for _, x in nx.all_pairs_shortest_path_length(G)])
