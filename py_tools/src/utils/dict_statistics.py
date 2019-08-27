import numpy as np
from collections import OrderedDict


def median_of_dict(d):
    """
    Returns median of dictionary that acts like histogram. Keys and values must be numbers.

    If odd, median is middle of sorted list of values. If even, median is the average of the two middle numbers.
    :param d: Dictionary
    :return: Median
    """
    num_values = sum(d.values())
    if num_values % 2 == 0:
        even = True
        target = num_values // 2
    else:
        even = False
        target = (num_values // 2)+ 1

    d_list = list(sorted(d.items(), key=lambda x: x[0]))

    if len(d_list) == 0:
        raise RuntimeError

    counter = 0
    for i, kv in enumerate(d_list):
        k, v = kv
        if counter + v > target:
            return k
        if counter + v == target:
            if even:
                return (k + d_list[i+1][0])/2
            else:
                return k
        if counter + v < target:
            counter += v


def max_key_of_dict(d):
    """
    Returns largest key in dictionary, corresponding to diameter of network

    :param d: Dictionary
    :return: Max key
    """
    if len(d.keys()) == 0:
        raise RuntimeError

    return max(d.keys())


def mean_of_dict(d):
    """
    Returns mean of dictionary, corresponding to MGD of network

    :param d: Dictionary
    :return: Mean value
    """
    if len(d.keys()) == 0:
        raise RuntimeError

    return sum([k*v for k, v in d.items()])/sum([v for _, v in d.items()])
