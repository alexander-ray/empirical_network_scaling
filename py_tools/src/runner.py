import sys
import numpy as np
from empirical_network_scaling.py_tools.src.graph import semisparse as ss
from empirical_network_scaling.py_tools.src.graph import empirical
from empirical_network_scaling.py_tools.src.graph import mcmc
from empirical_network_scaling.py_tools.src.multi import helper as mh
from empirical_network_scaling.py_tools.src.graph import pathsample as ps
from empirical_network_scaling.py_tools.src.graph.apsp_multi_component import apsp_multi_component
from empirical_network_scaling.py_tools.src.utils.dict_statistics import *
from abc import ABC, abstractclassmethod
import os
import igraph


class NetworkStatisticGenerator(ABC):
    @staticmethod
    @abstractclassmethod
    def generate(filename):
        pass


class BruteMGDGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = empirical.networkx_from_gml(source)
        else:
            G = source

        n = len(G)
        degrees = list(sorted([degree[1] for degree in G.degree]))
        c = sum(degrees) / len(G)

        num_distances, sum_distances = ss.all_pairs_shortest_paths_rolling_sum(G)
        if num_distances == 0:
            mgd = 0
        else:
            mgd = sum_distances / num_distances

        # Print info for tracking after calculation has been completed
        print([source, n, c, mgd])
        sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c, mgd]]
        else:
            return [['', n, c, mgd]]


class DistanceDistributionGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = empirical.networkx_from_gml(source)
        else:
            G = source

        print('Working on ' + source)
        sys.stdout.flush()

        n = len(G)
        degrees = list(sorted([degree[1] for degree in G.degree]))
        c = sum(degrees) / len(G)

        distribution = ss.all_pairs_shortest_paths(G)

        ret = [source, n, c]
        dist_list = list(sorted([(k, v) for k, v in distribution.items()], key=lambda x: x[0]))
        for k, v in dist_list:
            ret.append(k)
            ret.append(v)
        # Print info for tracking after calculation has been completed
        print(ret)
        sys.stdout.flush()

        if type(source) == str:
            return [ret]
        else:
            return [ret]


class GlobalClusteringGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = empirical.igraph_from_gml(source)
        else:
            G = source

        n = G.vcount()
        degrees = list(sorted([d for d in G.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        clustering = G.transitivity_undirected(mode='zero')
        # Print info for tracking after calculation has been completed
        print(source + '  ' + str(os.getpid()))
        sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c, clustering]]
        else:
            return [['', n, c, clustering]]


class DegreeAssortativityGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = empirical.igraph_from_gml(source)
        else:
            G = source

        n = G.vcount()
        degrees = list(sorted([d for d in G.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        assort = G.assortativity_degree(directed=False)
        # Print info for tracking after calculation has been completed
        print(source + '  ' + str(os.getpid()))
        sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c, assort]]
        else:
            return [['', n, c, assort]]


class MeanDegreeGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = empirical.igraph_from_gml(source)
        else:
            G = source

        n = G.vcount()
        degrees = list(sorted([d for d in G.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        # Print info for tracking after calculation has been completed
        print(source + '  ' + str(os.getpid()))
        sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c]]
        else:
            return [['', n, c]]


class ErdosRenyiGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        ret = []
        ret.append(source)

        print('graph: ' + source)
        sys.stdout.flush()

        g = empirical.igraph_from_gml(source)
        n = g.vcount()
        m = g.ecount()
        degrees = list(sorted([d for d in g.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        ret.append(n)
        ret.append(c)

        print(ret)
        sys.stdout.flush()

        for _ in range(20):
            g = igraph.Graph.Erdos_Renyi(n=n, m=m, directed=False, loops=False)
            if n < 1000:
                dist = apsp_multi_component(g)
                mgd = mean_of_dict(dist)
                median = median_of_dict(dist)
                max_val = max_key_of_dict(dist)
            else:
                samples = ps.threshold_sampler_igraph(g, threshold=0.1, batch_size=1000)
                mgd = np.mean(samples)
                median = np.median(samples)
                max_val = np.max(samples)
            ret.append(mgd)
            ret.append(median)
            ret.append(max_val)
            ret.append(g.transitivity_undirected(mode='zero'))
            ret.append(g.assortativity_degree(directed=False))

        print(ret)
        sys.stdout.flush()
        return [ret]


class MCMCGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        ret = []
        ret.append(source)

        print('graph: ' + source)
        sys.stdout.flush()

        g = empirical.igraph_from_gml(source)
        sampler = mcmc.MCMCSampler(g=g, burn_swaps=None, convergence_threshold=0.05, mixing_swaps=None, p=1)
        n = g.vcount()
        degrees = list(sorted([d for d in g.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        ret.append(n)
        ret.append(c)

        print(ret)
        sys.stdout.flush()

        for _ in range(10):
            g = sampler.get_new_sample()
            if n < 1000:
                dist = apsp_multi_component(g)
                mgd = mean_of_dict(dist)
                median = median_of_dict(dist)
                max_val = max_key_of_dict(dist)
            else:
                samples = ps.threshold_sampler_igraph(g, threshold=0.1, batch_size=1000)
                mgd = np.mean(samples)
                median = np.median(samples)
                max_val = np.max(samples)
            ret.append(mgd)
            ret.append(median)
            ret.append(max_val)
            ret.append(g.transitivity_undirected(mode='zero'))
            ret.append(g.assortativity_degree(directed=False))

        print(ret)
        sys.stdout.flush()
        return [ret]


path_prepend = '/scratch/Users/alra7284/gmls/Anna/'
paths = []
with open('/Users/alra7284/random_networks.txt', 'r') as f:
    for line in f:
        #if not line.strip().endswith('n6.gml') and not line.strip().endswith('n5.gml'):
        paths.append(path_prepend + line.strip())


print('Num networks: ' + str(len(paths)))
print('Num CPUs: ' + str(int(os.environ['SLURM_JOB_CPUS_PER_NODE'])))

mh.multiprocessing_to_csv(DegreeAssortativityGenerator.generate,
                          paths,
                          f'/Users/alra7284/assortativity_{len(paths)}.csv',
                          int(os.environ['SLURM_JOB_CPUS_PER_NODE']))
