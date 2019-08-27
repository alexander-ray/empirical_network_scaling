import igraph
import numpy as np
from collections import defaultdict


def apsp_multi_component(g):
    """
    Driver function for igraph apsp with (potentially) disconnected components

    :param g: igraph.Graph
    :return: distribution
    """
    results = defaultdict(int)
    if g.vcount() <= 1:
        results[0.0] += 1
        return results
    if g.is_connected(mode='STRONG'):
        res = g.shortest_paths(source=None, target=None, weights=None, mode='OUT')
        for i in res:
            for j in i:
                results[j] += 1
        return results
    else:
        components = g.components()
        for c in components:
            if len(c) <= 1:
                results[0.0] += 1
            else:
                res = g.shortest_paths(source=c, target=c, weights=None, mode='OUT')
                for i in res:
                    for j in i:
                        results[j] += 1
        return results


def apsp_multi_component_rolling_sum(g):
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
