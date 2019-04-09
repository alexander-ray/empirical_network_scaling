import networkx as nx
import igraph


def networkx_from_gml(filepath):
    """
    Faster method to read GML file into networkx graph using igraph.
    Reads GML with igraph, converts to undirected, and uses edge list to create corresponding nx.Graph

    :param filepath: Path of GML file
    :return: nx.Graph
    """
    G = igraph_from_gml(filepath)
    edges = [e.tuple for e in G.es]
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    return G_nx


def igraph_from_gml(filepath):
    """
    Helper method to read GML file into igraph.
    Converts to undirected network.

    :param filepath: Path of GML file
    :return: igraph.Graph
    """
    G = igraph.read(filepath)
    if 'weight' in set(G.es.attributes()):
        del G.es['weight']
    G.to_undirected(combine_edges=None)
    G.simplify(multiple=True, loops=True, combine_edges=None)
    return G


def networkx_to_igraph(G):
    """
    Helper method convert undirected networkx graph to igraph

    :param G: nx.Graph
    :return: igraph.Graph
    """
    # Check to make sure nx graph starts at zero
    # If not, force it too
    # Otherwise, the igraph graph will have dangling '0' node
    if not G.has_node(0):
        G = nx.convert_node_labels_to_integers(G, first_label=0)

    edges = [e for e in G.edges]
    # setting n insures dangling nodes are kept
    return igraph.Graph(n=len(G), edges=edges, directed=False)


def igraph_to_networkx(g):
    """
    Helper method convert undirected igraph graph to networkx

    :param g: igraph.Graph
    :return: nx.Graph
    """
    edges = [e.tuple for e in g.es]
    G = nx.Graph()
    G.add_nodes_from(range(g.vcount()))
    G.add_edges_from(edges)
    return G
