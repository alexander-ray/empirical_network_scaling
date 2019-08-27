import sys
import time
import numpy as np
import csv
from src.graph import semisparse as ss
from src.graph import empirical
from src.graph import configuration_model_mcmc
from src.multi import helper as mh
from src.graph import pathsample as ps
from src.graph.apsp_multi_component import apsp_multi_component
from src.utils.dict_statistics import *
from abc import ABC, abstractclassmethod
import os
from scipy.stats import sem

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


class MCMCGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        ret = []
        ret.append(source)

        g = empirical.igraph_from_gml(source)
        sampler = configuration_model_mcmc.MCMCSampler(g=g, burn_swaps=None,
                                                       convergence_threshold=0.05, mixing_swaps=None)

        n = g.vcount()
        degrees = list(sorted([d for d in g.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        ret.append(n)
        ret.append(c)

        for _ in range(20):
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


class TestGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        ret = []
        ret.append(source)
        start = time.time()
        g = empirical.igraph_from_gml(source)
        end = time.time()
        network_read_time = end-start

        start = time.time()
        samples = ps.threshold_sampler_igraph(g, threshold=1, batch_size=10000)
        end = time.time()
        sampler_time = end-start

        ret.append(network_read_time)
        ret.append(sampler_time)
        ret.append(len(samples))
        ret.append(np.mean(samples))
        ret.append(np.std(samples, ddof=1))
        ret.append(sem(samples, ddof=1))

        print(ret)
        sys.stdout.flush()

        return [ret]

'''
path_prepend = '/scratch/Users/alra7284/gmls/Anna/'
paths = []
with open('random_networks.txt', 'r') as f:
    for line in f:
        if not line.strip().endswith('n6.gml') and not line.strip().endswith('n5.gml'):
            paths.append(path_prepend + line.strip())


print('Num networks: ' + str(len(paths)))
print('Num CPUs: ' + str(int(os.environ['SLURM_JOB_CPUS_PER_NODE'])))

mh.multiprocessing_to_csv(BruteMGDGenerator.generate,
                          paths,
                          'output_mgd_check_w_new_read_function.csv',
                          int(os.environ['SLURM_JOB_CPUS_PER_NODE']))
'''
