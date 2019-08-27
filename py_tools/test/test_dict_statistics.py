from src.utils.dict_statistics import median_of_dict, max_key_of_dict
import pytest
import numpy as np


#############################################
# Test: median_of_dict
#############################################
def test_empty():
    with pytest.raises(Exception):
        median_of_dict({})


def test_single():
    d = {0: 1}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 0


def test_single2():
    d = {0: 22}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 0


def test_multiple():
    d = {0: 22, 1: 1}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 0


def test_multiple2():
    d = {0: 22, 3: 56}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 3


def test_multiple3():
    d = {0.0: 1, 1.0: 1, 2.0: 1}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 1.0


def test_multiple4():
    d = {0.0: 1, 1.0: 1, 2.0: 1, 34.0: 1}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 1.5


def test_multiple5():
    d = {0.0: 1, 5.0: 1, 2.0: 1, 34.0: 1}
    assert np.median([i for sl in [[k]*v for k, v in d.items()] for i in sl]) == median_of_dict(d) == 3.5


#############################################
# Test: max_key_of_dict
#############################################
def test_empty_max():
    with pytest.raises(Exception):
        max_key_of_dict({})


def test_single_max():
    d = {0: 1}
    assert max_key_of_dict(d) == 0


def test_single2_max():
    d = {0: 22}
    assert max_key_of_dict(d) == 0


def test_multiple_max():
    d = {0: 22, 1: 1}
    assert max_key_of_dict(d) == 1


def test_multiple2_max():
    d = {0: 22, 3: 56}
    assert max_key_of_dict(d) == 3


def test_multiple3_max():
    d = {0.0: 1, 1.0: 1, 2.0: 1}
    assert max_key_of_dict(d) == 2.0


def test_multiple4_max():
    d = {34.0: 1, 0.0: 1, 1.0: 1, 2.0: 1}
    assert max_key_of_dict(d) == 34.0

