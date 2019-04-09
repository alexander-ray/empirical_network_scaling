import igraph
import numpy as np
from src.graph.pathsample import _component_probability_generator_igraph


#############################################
# Test: _component_probability_generator_igraph
#############################################
def test_probs_for_single_component_small():
    g = igraph.Graph()
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2)])
    _, probabilities = _component_probability_generator_igraph(g)
    assert probabilities == [1]


def test_probs_for_single_component_zkc():
    g = igraph.Graph.Famous(name='Zachary')
    _, probabilities = _component_probability_generator_igraph(g)
    assert probabilities == [1]


def test_probs_for_two_components():
    g = igraph.Graph.Famous(name='Zachary')
    old_size = g.vcount()
    g.add_vertex(name='one')
    g.add_vertex(name='two')
    g.add_edge('one', 'two')
    _, probabilities = _component_probability_generator_igraph(g)

    tmp = old_size**2 + 4
    assert probabilities == [(old_size**2)/tmp, 4/tmp]


def test_probs_for_three_components():
    g = igraph.Graph()
    g.add_vertices(6)
    # comp sizes: 1, 3, 2
    g.add_edges([(1, 2), (1, 3), (4, 5)])
    _, probabilities = _component_probability_generator_igraph(g)

    tmp = 1**2 + 3**2 + 2**2
    assert probabilities == [(1**2)/tmp, (3**2)/tmp, (2**2)/tmp]


#############################################
# Test: np.random.choice
#############################################
def test_np_random_choice():
    vals = []
    n = 10000
    for i in range(n):
        vals.append(np.random.choice(4, p=[0.1, 0.2, 0.4, 0.3]))
    # Test to make sure real sum is close to expectation
    assert sum(vals) - ((0.2*1*n) + (0.4*2*n) + (0.3*3*n)) in range(-500, 500)
