from src.graph.mcmc import *


#############################################
# Test: stepping configuration mode
#############################################
def test_equality_1():
    num_swaps = 100
    seed = 123
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])

    allow_loops = False
    allow_multi_edges = False
    is_vertex_labeled = False
    np.random.seed(seed)
    graph_of_graphs = MCMC_class(G.copy(), allow_loops, allow_multi_edges, is_vertex_labeled)
    for _ in range(num_swaps):
        G2 = graph_of_graphs.step_and_get_graph()

    np.random.seed(seed)
    sampler = MCMCSamplerNX(G, burn_swaps=0, mixing_swaps=num_swaps)
    G = sampler.get_new_sample()

    assert list(G.edges) == list(G2.edges) and \
           list(G.degree) == list(G2.degree)


def test_equality_zkc():
    num_swaps = 1000
    seed = 123
    G = nx.karate_club_graph()

    allow_loops = False
    allow_multi_edges = False
    is_vertex_labeled = False
    np.random.seed(seed)
    graph_of_graphs = MCMC_class(G.copy(), allow_loops, allow_multi_edges, is_vertex_labeled)
    for _ in range(num_swaps):
        G2 = graph_of_graphs.step_and_get_graph()

    np.random.seed(seed)
    sampler = MCMCSamplerNX(G, burn_swaps=0, mixing_swaps=num_swaps)
    G = sampler.get_new_sample()
    assert nx.is_isomorphic(G, G2) and \
           list(G.edges) == list(G2.edges) and \
           list(G.degree) == list(G2.degree) and \
           nx.degree_assortativity_coefficient(G) == nx.degree_assortativity_coefficient(G2)


def test_equality_er():
    num_swaps = 100
    seed = 123
    np.random.seed(seed)
    G = nx.gnm_random_graph(1000, 10000)

    allow_loops = False
    allow_multi_edges = False
    is_vertex_labeled = False
    np.random.seed(seed)
    graph_of_graphs = MCMC_class(G.copy(), allow_loops, allow_multi_edges, is_vertex_labeled)
    for _ in range(num_swaps):
        G2 = graph_of_graphs.step_and_get_graph()

    np.random.seed(seed)
    sampler = MCMCSamplerNX(G, burn_swaps=0, mixing_swaps=num_swaps)
    G = sampler.get_new_sample()
    assert nx.is_isomorphic(G, G2) and \
           list(G.edges) == list(G2.edges) and \
           list(G.degree) == list(G2.degree) and \
           nx.degree_assortativity_coefficient(G) == nx.degree_assortativity_coefficient(G2)


def test_equality_er_2():
    num_swaps = 1000
    seed = 234
    np.random.seed(seed)
    G = nx.gnm_random_graph(1000, 10000)

    allow_loops = False
    allow_multi_edges = False
    is_vertex_labeled = False
    np.random.seed(seed)
    graph_of_graphs = MCMC_class(G.copy(), allow_loops, allow_multi_edges, is_vertex_labeled)
    for _ in range(num_swaps):
        G2 = graph_of_graphs.step_and_get_graph()

    np.random.seed(seed)
    sampler = MCMCSamplerNX(G, burn_swaps=0, mixing_swaps=num_swaps)
    G = sampler.get_new_sample()
    assert nx.is_isomorphic(G, G2) and \
           list(G.edges) == list(G2.edges) and \
           list(G.degree) == list(G2.degree) and \
           nx.degree_assortativity_coefficient(G) == nx.degree_assortativity_coefficient(G2)


def test_equality_er_3():
    num_swaps = 10000
    seed = 234
    np.random.seed(seed)
    G = nx.gnm_random_graph(1000, 10000)

    allow_loops = False
    allow_multi_edges = False
    is_vertex_labeled = False
    np.random.seed(seed)
    graph_of_graphs = MCMC_class(G.copy(), allow_loops, allow_multi_edges, is_vertex_labeled)
    for _ in range(num_swaps):
        G2 = graph_of_graphs.step_and_get_graph()

    np.random.seed(seed)
    sampler = MCMCSamplerNX(G, burn_swaps=0, mixing_swaps=num_swaps)
    G = sampler.get_new_sample()
    assert nx.is_isomorphic(G, G2) and \
           list(G.edges) == list(G2.edges) and \
           list(G.degree) == list(G2.degree) and \
           nx.degree_assortativity_coefficient(G) == nx.degree_assortativity_coefficient(G2)
