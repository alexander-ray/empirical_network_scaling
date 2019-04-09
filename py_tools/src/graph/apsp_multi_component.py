import igraph
import numpy as np


def apsp_multi_component(g):
    """
    Driver function for igraph apsp with (potentially) disconnected components

    :param g: igraph.Graph
    :return: mgd
    """
    if g.vcount() <= 1:
        return 0
    if g.is_connected(mode='STRONG'):
        return np.mean(g.shortest_paths(source=None, target=None, weights=None, mode='OUT'))
    else:
        num = 0
        components = g.components()
        denom = np.sum([len(x)**2 for x in components])
        for c in components:
            num += np.sum(g.shortest_paths(source=c, target=c, weights=None, mode='OUT'))
        if denom == 0:
            return 0
        return num/denom
