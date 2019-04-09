import networkx as nx
from abc import ABC, abstractmethod
import igraph
from .empirical import networkx_to_igraph, igraph_to_networkx
import numpy as np
import numba as nb
from scipy.stats import ks_2samp
from scipy.stats import entropy
jit = nb.jit


class AbstractMCMCSampler(ABC):
    @abstractmethod
    def __init__(self, G, burn_swaps=None, convergence_threshold=0.05, mixing_swaps=None):
        """
        :param G: nx.Graph
        :param burn_swaps: Number of swaps for burn-in. If None, uses convergence threshold
        :param convergence_threshold: Threshold for KL-divergence for burn-in
        :param mixing_swaps: Number of swaps between samples. If falsey, default to 2m
        """
        self._G = G

        # Precomputations
        self._edges = list(self._G.edges)
        self._edge_indices = list(range(self._G.number_of_edges()))
        self._m = len(self._edges)

        if mixing_swaps:
            self._mixing_swaps = mixing_swaps
        else:
            self._mixing_swaps = 2*self._m

        if burn_swaps is not None:
            for _ in range(burn_swaps):
                self._swap()
        else:
            if not convergence_threshold:
                print('Convergence threshold not defined')
            # Burn-in until MC has converged on stationary distribution
            self._burn_to_convergence(convergence_threshold)

    @abstractmethod
    def get_new_sample(self):
        """
        Mix self._G for self._mixing_swaps and return sample
        """
        pass

    def _burn_to_convergence(self, threshold):
        """
        Function to determine number of swaps necessary to achieve convergence of markov chain
        Uses entropy of degree assortativity to assess convergence

        :param threshold: Convergence threshold for difference in entropy
        :return: Number of swaps
        """
        def populate_assortivaties():
            assort_arr = []
            for _ in range(samples_per_group):
                for _ in range(int(np.ceil(t/samples_per_group))):
                    self._swap()
                assort_arr.append(networkx_to_igraph(self._G).assortativity_degree(directed=False))
            return assort_arr

        t = self._m  # Number of swaps in a "group"
        samples_per_group = 30
        total_swaps = 0

        threshold_met = False
        while not threshold_met:
            assort1 = populate_assortivaties()
            assort2 = populate_assortivaties()
            total_swaps += 2*t
            if np.abs(np.mean(assort1) - np.mean(assort2)) < threshold:
                threshold_met = True
        # Return for debugging purposes
        return total_swaps, total_swaps/self._m

    def _swap(self):
        """
        Perform one stub-labeled double-edge swap as specified in Fosdick et. al.
        Modification of their code https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
        :return:
        """
        p1 = np.random.randint(self._m)
        p2 = np.random.randint(self._m - 1)
        if p1 == p2:  # Prevents picking the same edge twice
            p2 = self._m - 1

        u, v = self._edges[p1]
        r = np.random.rand()
        if r < 0.5:
            x, y = self._edges[p2]
        else:
            y, x = self._edges[p2]

        # ensure no multigraph
        if x in self._G[u] or y in self._G[v]:
            return
        if u == v and x == y:
            return

        # ensure no loops
        if u == x or u == y or v == x or v == y:
            return

        self._G.remove_edges_from([(u, v), (x, y)])
        self._G.add_edges_from([(u, x), (v, y)])
        self._edges[p1] = (u, x)
        self._edges[p2] = (v, y)


class MCMCSampler(AbstractMCMCSampler):
    def __init__(self, g, burn_swaps=None, convergence_threshold=0.05, mixing_swaps=None):
        """
        :param g: igraph.Graph
        :param burn_swaps: Number of swaps for burn-in. If falsey, uses convergence threshold
        :param convergence_threshold: Threshold for KL-divergence for burn-in
        :param mixing_swaps: Number of swaps between samples. If falsey, default to 2m
        """
        # Networkx version of graph
        super().__init__(igraph_to_networkx(g), burn_swaps=burn_swaps,
                         convergence_threshold=convergence_threshold, mixing_swaps=mixing_swaps)

    def get_new_sample(self):
        """
        Mix self._G for self._mixing_swaps and return sample
        """
        for _ in range(self._mixing_swaps):
            self._swap()
        return networkx_to_igraph(self._G)


class MCMCSamplerNX(AbstractMCMCSampler):
    def __init__(self, G, burn_swaps=None, convergence_threshold=0.05, mixing_swaps=None):
        """
        :param G: nx.Graph
        :param burn_swaps: Number of swaps for burn-in. If falsey, uses convergence threshold
        :param convergence_threshold: Threshold for KL-divergence for burn-in
        :param mixing_swaps: Number of swaps between samples. If falsey, default to 2m
        """
        super().__init__(G, burn_swaps=burn_swaps,
                         convergence_threshold=convergence_threshold, mixing_swaps=mixing_swaps)

    def get_new_sample(self):
        """
        Mix self._G for self._mixing_swaps and return sample
        """
        for _ in range(self._mixing_swaps):
            self._swap()
        return self._G


# Fosdick et. al. code for testing
#@jit(nopython=True, nogil=True)
def MCMC_step_stub(A, edge_list, swaps, allow_loops, allow_multi):
    '''

    Performs a stub-labeled double edge swap.

    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph space.

    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled.

    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this
    to work a sparse adjacency matrix simply requires removing the '@nb.jit'
    decorator. This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.

    '''
    # Choose two edges uniformly at random
    m = len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m - 1)
    if p1 == p2:  # Prevents picking the same edge twice
        p2 = m - 1

    u, v = edge_list[p1]
    r = np.random.rand()
    if r < 0.5:  # Pick either swap orientation 50% at random
        x, y = edge_list[p2]
    else:
        y, x = edge_list[p2]

    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_ux = A[u, x]
    w_vy = A[v, y]

    # If multiedges are not allowed, resample if swap would replicate an edge
    if not allow_multi:
        if (w_ux >= 1 or w_vy >= 1):
            return False

        if u == v and x == y:
            return False

    # If loops are not allowed then only swaps on 4 distinct nodes are possible
    if not allow_loops:
        if u == x or u == y or v == x or v == y:
            return False

    swaps[0] = u  # Numba currently is having trouble with slicing
    swaps[1] = v
    swaps[2] = x
    swaps[3] = y

    A[u, v] += -1
    A[v, u] += -1
    A[x, y] += -1
    A[y, x] += -1

    A[u, x] += 1
    A[x, u] += 1
    A[v, y] += 1
    A[y, v] += 1

    edge_list[p1, 0] = u
    edge_list[p1, 1] = x
    edge_list[p2, 0] = v
    edge_list[p2, 1] = y

    return True


class MCMC_class:
    '''

    MCMC_class stores the objects necessary for MCMC steps. This
    implementation maintains a networkx version of the graph, though at some
    cost in speed.

    | Args:
    |     G (networkx_class): This graph initializes the Markov chain. All
             sampled graphs will have the same degree sequence as G.
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph space.
    |     is_v_labeled (bool): True only if the graph space is vertex-labeled.
            True by default.

    | Returns:
    |     None

    Notes
    -----
    MCMC_class copies the instance of the graph used to initialize it. This
    class supports loopy graphs, but depending on the degree sequence, it may
    not be able to sample from all loopy graphs.

    '''

    def __init__(self, G, allow_loops, allow_multi, is_v_labeled=True):
        #self.G = flatten_graph(G, allow_loops, allow_multi)
        self.G = G
        self.allow_loops = allow_loops
        self.allow_multi = allow_multi
        self.is_v_labeled = is_v_labeled
        if self.is_v_labeled:
            raise NotImplementedError
        else:
            self.step = MCMC_step_stub

        self.A = nx.adjacency_matrix(self.G)
        self.A = self.A.toarray()
        self.A += np.diag(np.diag(self.A))
        self.edge_list = np.array(self.G.edges())
        self.swaps = np.zeros(4, dtype=np.int64)

    def step_and_get_graph(self):
        '''

        The Markov chains will attempt a double edge swap, after which the next
        graph/multigraph in the chain is returned.

        | Args:
        |     None

        | Returns:
        |     The Markov chain's current graph.

        Notes
        -----
        Modifying the returned graph will cause errors in repeated calls of
        this function.

        '''
        new = self.step(self.A, self.edge_list, self.swaps, self.allow_loops, self.allow_multi)
        if new:
            #            print swaps, new
            #            print A
            u, v, x, y = self.swaps
            self.G.add_edge(u, x)
            self.G.add_edge(v, y)
            self.G.remove_edge(u, v)
            self.G.remove_edge(x, y)

        return self.G
