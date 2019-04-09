import igraph
import itertools
import networkx as nx
from src.graph.empirical import *


#############################################
# Test: networkx to igraph
#############################################
def test_nx_to_igraph_empty():
    G = nx.Graph()
    g = networkx_to_igraph(G)

    assert len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_nx_to_igraph_1():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    g = networkx_to_igraph(G)

    assert nx.degree_assortativity_coefficient(G) == g.assortativity_degree() \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_nx_to_igraph_2():
    G = nx.karate_club_graph()
    g = networkx_to_igraph(G)

    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_nx_to_igraph_3():
    G = nx.Graph()
    # First triad
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    G.add_edges_from([(4, 5), (5, 6)])
    G.add_node(9)
    g = networkx_to_igraph(G)

    # Account for G change in networkx_to_igraph
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


#############################################
# Test: igraph to networkx
#############################################
def test_igraph_to_nx_empty():
    g = igraph.Graph(directed=False)
    G = igraph_to_networkx(g)

    assert len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_igraph_to_nx_1():
    g = igraph.Graph(directed=False)
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2)])
    G = igraph_to_networkx(g)

    assert nx.degree_assortativity_coefficient(G) == g.assortativity_degree() \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_igraph_to_nx_2():
    g = igraph.Graph.Famous(name='Zachary')
    G = igraph_to_networkx(g)

    # Make sure nx edge list has edges reported s.t. e[0] <= e[1]
    nx_edges = [e if e[0] <= e[1] else (e[1], e[0]) for e in list(G.edges)]
    nx_edges = list(sorted(nx_edges, key=lambda x: (x[0], x[1])))

    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and nx_edges == \
               list(sorted([e.tuple for e in g.es], key=lambda x: (x[0], x[1])))


def test_igraph_to_nx_3():
    g = igraph.Graph(directed=False)
    g.add_vertices(6)
    # First triad
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.add_edges([(3, 4), (4, 5)])
    G = igraph_to_networkx(g)

    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))


def test_igraph_to_nx_4():
    g = igraph.Graph(directed=False)
    # Add extra dangling nodes
    g.add_vertices(8)
    # First triad
    g.add_edges([(0, 1), (1, 2), (2, 0)])
    g.add_edges([(3, 4), (4, 5)])
    G = igraph_to_networkx(g)

    assert round(nx.degree_assortativity_coefficient(G), 6) == round(g.assortativity_degree(), 6) \
           and len(G) == g.vcount() and G.number_of_edges() == g.ecount() \
           and list(sorted(list(G.edges))) == list(sorted([e.tuple for e in g.es]))
