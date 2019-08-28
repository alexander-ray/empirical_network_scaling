import sys
sys.path.append('/Users/alexray/Dropbox/graph_scaling/empirical_network_scaling')
from py_tools.src.utils.dict_statistics import mean_of_dict

# https://stackoverflow.com/questions/5389507/
def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)

with open('answers.txt') as f1, open('output.txt') as f2:
    for x, y in zip(f1, f2):
        x = float(x)
        l = y.strip().split(',')
        name = l[0]
        dist_list = l[1:]
        dist = {}
        for a, b in pairwise(dist_list):
            dist[int(a)] = float(b)
        
        mgd = round(mean_of_dict(dist), 6)
        if mgd != x:
            print(f'FAILED {name}-- Expected: {x} Actual: {mgd}')
        else:
            print(f'PASSED {name}')

