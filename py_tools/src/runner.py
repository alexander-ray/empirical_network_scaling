import sys
import time
import numpy as np
import csv
from src.graph.empirical import igraph_from_gml, networkx_to_igraph, networkx_from_gml
from src.graph.apsp_multi_component import apsp_multi_component
import src.graph.configuration_model_mcmc as configuration_model_mcmc
import src.graph.semisparse as ss
import src.graph.pathsample as ps
import src.multi.helper as mh
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
            G = networkx_from_gml(source)
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
        #print([source, n, c, mgd])
        #sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c, mgd]]
        else:
            return [['', n, c, mgd]]


class BruteMGDGeneratorLowMem(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = igraph_from_gml(source)
        else:
            G = source

        n = G.vcount()
        degrees = list(sorted([d for d in G.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        mgd = apsp_multi_component(G)

        # Print info for tracking after calculation has been completed
        print([source, n, c, mgd])
        sys.stdout.flush()

        if type(source) == str:
            return [[source, n, c, mgd]]
        else:
            return [['', n, c, mgd]]


class SampledMGDGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = networkx_from_gml(source)
        else:
            G = source

        t = 0.005
        n = len(G)
        degrees = list(sorted([degree[1] for degree in G.degree]))
        c = sum(degrees) / len(G)

        ret = [source, n, c]

        alphas = [0.1, 0.05, 0.01]
        for alpha in alphas:
            G = igraph_from_gml(source)
            proportions = ps.ew_sampler_igraph(G, alpha=alpha, t=t)
            ret.append(sum(k * v for k, v in proportions.items()))
            ret.append(np.sqrt(sum((k * t) ** 2 for k in proportions.keys())))

        # Print info for tracking after calculation has been completed
        print(source + '  ' + str(os.getpid()))
        sys.stdout.flush()

        if type(source) == str:
            return [ret]
        else:
            return [['', n, c]]


class GlobalClusteringGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = igraph_from_gml(source)
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


class MeanDegreeGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        if type(source) == str:
            G = igraph_from_gml(source)
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

        g = igraph_from_gml(source)
        start = time.time()
        sampler = configuration_model_mcmc.MCMCSampler(g=g, burn_swaps=None,
                                                       convergence_threshold=0.05, mixing_swaps=None)
        end = time.time()

        n = g.vcount()
        degrees = list(sorted([d for d in g.degree(mode='ALL', loops=False)]))
        c = sum(degrees) / n

        ret.append(n)
        ret.append(c)

        for _ in range(20):
            g = sampler.get_new_sample()
            if n < 1000:
                mgd = apsp_multi_component(g)
            else:
                samples = ps.threshold_sampler_igraph(g, threshold=0.1, batch_size=1000)
                mgd = np.mean(samples)
            ret.append(mgd)
            ret.append(g.transitivity_undirected(mode='zero'))

        print(ret)
        sys.stdout.flush()
        return [ret]


class TestGenerator(NetworkStatisticGenerator):
    @staticmethod
    def generate(source):
        ret = []
        ret.append(source)
        start = time.time()
        g = igraph_from_gml(source)
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

#paths = []
#mh.multiprocessing_to_csv(MCMCGenerator.generate,
#                          paths,
#                          'output.csv', 8)
