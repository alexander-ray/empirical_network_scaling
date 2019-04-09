import networkx
import igraph


def bidirectional_bfs_distance_networkx(G, s, t):
    """
    Stripped version of networkx "bidirectional_shortest_path" function, removing unnecessary stuff.

    :param G: nx.Graph
    :param s: Source node
    :param t: Target node
    :return: Distance between nodes
    """
    if s == t:
        return 0

    # predecesssor and successors in search
    dist_pred = {s: 0}
    dist_succ = {t: 0}

    # initialize fringes, start with forward
    forward_fringe = [s]
    reverse_fringe = [t]

    while forward_fringe and reverse_fringe:
        # Decide which one to work on
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            # Iterate through level
            for v in this_level:
                for w in G.adj[v]:
                    if w not in dist_pred:
                        forward_fringe.append(w)
                        dist_pred[w] = dist_pred[v] + 1
                    if w in dist_succ:  # path found
                        return dist_pred[w] + dist_succ[w]
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in G.adj[v]:
                    if w not in dist_succ:
                        dist_succ[w] = dist_succ[v] + 1
                        reverse_fringe.append(w)
                    if w in dist_pred:  # found path
                        return dist_pred[w] + dist_succ[w]


def bidirectional_bfs_distance_igraph(G, s, t):
    """
    Stripped version of networkx "bidirectional_shortest_path" function, adapted to igraph.

    :param G: igraph.Graph
    :param s: Source node
    :param t: Target node
    :return: Distance between nodes
    """
    if s == t:
        return 0

    # predecesssor and successors in search
    dist_pred = {s: 0}
    dist_succ = {t: 0}

    # initialize fringes, start with forward
    forward_fringe = [s]
    reverse_fringe = [t]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in G.neighbors(v):
                    if w not in dist_pred:
                        forward_fringe.append(w)
                        dist_pred[w] = dist_pred[v] + 1
                    if w in dist_succ:  # path found
                        return dist_pred[w] + dist_succ[w]
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in G.neighbors(v):
                    if w not in dist_succ:
                        dist_succ[w] = dist_succ[v] + 1
                        reverse_fringe.append(w)
                    if w in dist_pred:  # found path
                        return dist_pred[w] + dist_succ[w]
