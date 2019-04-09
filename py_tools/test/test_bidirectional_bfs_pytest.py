import igraph
import itertools
from src.graph.bidirectional_bfs import bidirectional_bfs_distance_igraph


#############################################
# Test: bidirectional_bfs_distance_igraph
#############################################
def test_distance_to_self():
    g = igraph.Graph()
    g.add_vertices(1)
    assert bidirectional_bfs_distance_igraph(g, 1, 1) == 0


def test_distance_to_self_w_selfloop():
    g = igraph.Graph()
    g.add_vertices(1)
    g.add_edges([(0, 0)])
    assert bidirectional_bfs_distance_igraph(g, 1, 1) == 0


def test_distance_to_neighbor():
    g = igraph.Graph()
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2), (1, 2)])
    assert bidirectional_bfs_distance_igraph(g, 0, 1) == 1


def test_distance_in_zkc():
    # Test bidirectional vs real for all paths in zachary karate club network
    g = igraph.Graph.Famous(name='Zachary')
    test_results = []
    real_results = []
    for i, j in list(itertools.combinations(range(g.vcount()), 2)):
        test_results.append(bidirectional_bfs_distance_igraph(g, i, j))
        real_results.append(g.shortest_paths_dijkstra(source=[i], target=[j], weights=None, mode='OUT')[0][0])
    assert test_results == real_results


