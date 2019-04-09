from src.runner import BruteMGDGenerator
import networkx as nx


#############################################
# Test: BruteMGDGenerator
#############################################
def test_singleton_mgd():
    G = nx.Graph()
    G.add_node(1)
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    assert mgd == 0


def test_single_edge_mgd():
    G = nx.Graph()
    G.add_edge(1, 2)
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    assert mgd == .5


def test_zkc_mgd():
    G = nx.karate_club_graph()
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    # For single component, nx uses denom of n(n-1)
    # Multiply by (n-1) and divide by n to get comparable answers
    assert mgd == (nx.average_shortest_path_length(G)*(len(G)-1))/len(G)


def test_zkc_with_extra_triad_mgd():
    G = nx.karate_club_graph()
    orig_size = len(G)
    orig_sum_distances = nx.average_shortest_path_length(G)*(len(G)-1)*len(G)
    G.add_edges_from([(100, 101), (101, 102), (102, 100)])
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    assert mgd == (orig_sum_distances+6)/(orig_size**2 + 9)


def test_florentine_families_mgd():
    G = nx.florentine_families_graph()
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    assert mgd == (nx.average_shortest_path_length(G)*(len(G)-1))/len(G)


def test_connected_watts_strogatz():
    G = nx.connected_watts_strogatz_graph(100, 4, 0.5, 10)
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    assert mgd == round((nx.average_shortest_path_length(G) * (len(G) - 1)) / len(G), 4)


def test_disconnected_barbell():
    G = nx.barbell_graph(50, 0, create_using=None)
    # Remove bridging edge
    G.remove_edge(50-1, 50)
    _, _, _, mgd = BruteMGDGenerator.generate(G)[0]
    # There are 50*49 paths of length 1 in each barbell
    # Denom should be sum of the squares of the sizes of each barbell
    assert mgd == ((50*49)*2)/((50**2)*2)
