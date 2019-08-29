from src.graph.empirical import *
from src.graph.apsp_multi_component import *
from src.graph.semisparse import *


#############################################
# Test: networkx to igraph
#############################################
def test_nx_to_igraph_empty():
    G = nx.Graph()
    g = networkx_to_igraph(G)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es])) \
           and s / n == igraph_mgd


def test_nx_to_igraph_1():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    g = networkx_to_igraph(G)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert nx.degree_assortativity_coefficient(G) == g.assortativity_degree(directed=False) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es])) \
           and s / n == igraph_mgd


def test_nx_to_igraph_2():
    G = nx.karate_club_graph()
    g = networkx_to_igraph(G)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es])) \
           and s / n == igraph_mgd


def test_nx_to_igraph_3():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edges_from([(3, 4), (4, 5)])
    G.add_node(8)
    g = networkx_to_igraph(G)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es])) \
           and s/n == igraph_mgd


#############################################
# Test: igraph to networkx
#############################################
def test_igraph_to_nx_empty():
    g = igraph.Graph(directed=False)
    G = igraph_to_networkx(g)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))\
           and s/n == igraph_mgd


def test_igraph_to_nx_1():
    g = igraph.Graph(directed=False)
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2)])
    G = igraph_to_networkx(g)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert nx.degree_assortativity_coefficient(G) == g.assortativity_degree(directed=False) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))\
           and s/n == igraph_mgd


def test_igraph_to_nx_2():
    g = igraph.Graph.Famous(name='Zachary')
    G = igraph_to_networkx(g)

    # Make sure nx edge list has edges reported s.t. e[0] <= e[1]
    nx_edges = [e if e[0] <= e[1] else (e[1], e[0]) for e in list(G.edges)]
    nx_edges = list(sorted(nx_edges, key=lambda x: (x[0], x[1])))

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and nx_edges == \
               list(sorted([e.tuple for e in g.es], key=lambda x: (x[0], x[1])))\
           and s/n == igraph_mgd


def test_igraph_to_nx_3():
    g = igraph.Graph(directed=False)
    g.add_vertices(6)
    # First triad
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.add_edges([(3, 4), (4, 5)])
    G = igraph_to_networkx(g)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))\
           and s/n == igraph_mgd


def test_igraph_to_nx_4():
    g = igraph.Graph(directed=False)
    # Add extra dangling nodes
    g.add_vertices(8)
    # First triad
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.add_edges([(3, 4), (4, 5)])
    G = igraph_to_networkx(g)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))\
           and s/n == igraph_mgd


#############################################
# Test: igraph to networkx and back to igraph
#############################################
def test_igraph_to_nx_to_igraph_1():
    g = igraph.Graph(directed=False)
    # Add extra dangling nodes
    g.add_vertices(8)
    g.add_vertices(10)
    # First triad
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.add_edges([(3, 4), (4, 5)])
    G = igraph_to_networkx(g)
    g2 = networkx_to_igraph(G)

    n, s = all_pairs_shortest_paths_rolling_sum(G)
    igraph_mgd = apsp_multi_component_rolling_sum(g)
    igraph2_mgd = apsp_multi_component_rolling_sum(g2)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(directed=False), 6) == round(g2.assortativity_degree(directed=False), 6) \
           and len(G) == g.vcount() == g2.vcount() and G.number_of_edges() == g.ecount() == g2.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es])) == list(sorted([e.tuple for e in g2.es])) \
           and s/n == igraph_mgd == igraph2_mgd


#############################################
# Test: reading simple and complex has same result
#############################################
def test_files_1():
    g = igraph_from_gml('three_components_1.gml')
    g2 = igraph_from_gml('three_components_1_complex.gml')

    igraph_mgd = apsp_multi_component_rolling_sum(g)
    igraph2_mgd = apsp_multi_component_rolling_sum(g2)
    assert g.assortativity_degree(directed=False) == g2.assortativity_degree(directed=False) \
           and g.vcount() == g2.vcount() and g.ecount() == g2.ecount() \
           and list(sorted([e.tuple for e in g.es])) == list(sorted([e.tuple for e in g2.es])) \
           and igraph_mgd == igraph2_mgd
