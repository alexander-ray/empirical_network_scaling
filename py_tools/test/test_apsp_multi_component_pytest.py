import igraph
import networkx as nx
from src.graph import semisparse as ss
from src.graph.apsp_multi_component import apsp_multi_component
from src.graph.empirical import networkx_to_igraph
#############################################
# Test: apsp multi component. Relies on already-tested semisparse module and networkx to igraph module
#############################################


# Number of distances
def test_empty_mgd():
    g = igraph.Graph()
    assert apsp_multi_component(g) == 0  # convention


def test_singleton_mgd():
    g = igraph.Graph()
    g.add_vertices(1)
    assert apsp_multi_component(g) == 0  # convention


def test_single_edge_mgd():
    g = igraph.Graph()
    g.add_vertices(2)
    g.add_edge(0, 1)
    assert apsp_multi_component(g) == 0.5


def test_disconnected_triads_mgd():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    print(apsp_multi_component(networkx_to_igraph(G)))

    num_distances, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    assert sum_distances/num_distances == apsp_multi_component(networkx_to_igraph(G))


def test_three_components_mgd():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_node(9)
    num_distances, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    print(apsp_multi_component(networkx_to_igraph(G)))
    assert sum_distances/num_distances == apsp_multi_component(networkx_to_igraph(G))


def test_three_components_mgd_2():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    # Second triad
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    G.add_edge(9, 10)
    num_distances, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    print(apsp_multi_component(networkx_to_igraph(G)))
    assert sum_distances / num_distances == apsp_multi_component(networkx_to_igraph(G))


def test_zkc_mgd():
    G = nx.karate_club_graph()
    num_distances, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
    print(apsp_multi_component(networkx_to_igraph(G)))
    assert sum_distances / num_distances == apsp_multi_component(networkx_to_igraph(G))
