import networkx as nx
import numpy as np
from collections import Counter
from .bidirectional_bfs import bidirectional_bfs_distance_networkx, bidirectional_bfs_distance_igraph


def threshold_sampler_igraph(g, threshold=0.1, batch_size=1000):
    """
    :param G: igraph.Graph
    :param threshold: Threshold value
    :param batch_size: Number of samples to take before re-evaluating
    :return: List of samples
    """
    s1 = []
    s2 = []
    half_batch = int(batch_size/2)
    threshold_met = False
    while not threshold_met:
        s1.extend(sampler_no_rejection_igraph(g, half_batch))
        s2.extend(sampler_no_rejection_igraph(g, half_batch))
        if np.abs(np.mean(s1) - np.mean(s2)) < threshold:
            threshold_met = True
    return s1 + s2


def sampler_no_rejection_igraph(g, num_samples):
    """
    igraph version of "no-rejection" sampler for pairwise distances.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.

    :param g: igraph.Graph
    :param num_samples: Number of samples
    :return: List of pairwise distances
    """
    # Return early if graph too small
    if g.vcount() == 0 or g.vcount() == 1:
        return

    tracker = []
    components, probabilities = _component_probability_generator_igraph(g)
    num_components = len(components)
    for x in range(num_samples):
        subgraph_index = np.random.choice(num_components, p=probabilities)
        i, j = np.random.choice(components[subgraph_index], 2, replace=True)
        tracker.append(bidirectional_bfs_distance_igraph(g, i, j))

    return tracker


def _component_probability_generator_igraph(g):
    """
    Provides connected component list and probabilities for each component
    Probability of component i is proportional to n_i^2.

    :param g: igraph.Graph
    :return: list of components, list of probabilities
    """
    # Get list of connected component subgraphs
    components = g.components()
    num_components = len(components)
    component_sizes = [s for s in components.sizes()]

    # Make probabilities proportional to n_i ** 2
    tmp = [s ** 2 for s in component_sizes]
    probabilities = [n / sum(tmp) for n in tmp]
    return components, probabilities
