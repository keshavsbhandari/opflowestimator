# SHOULD BE IN UTILS
from pathlib import Path
from itertools import chain
import re
import random


# getid = lambda x: [*map(int, re.findall(r'\d+', x.name))][0]
# getdirdata = lambda x: [*map(lambda y: y.as_posix(), sorted(x.glob('*.png'), key=getid))]
# alldata = chain.from_iterable(map(chunker, map(getdirdata, Path(sintelroot).glob('*'))))


def getSintelPairFrame(root, sample=None, test = False):
    def nsample(datalist):
        return random.sample(datalist, sample) if sample else datalist

    def chunker(lst):
        return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]

    def getid(x):
        return [*map(int, re.findall(r'\d+', x.name))][0]

    def getdirdata(x):
        return [*map(lambda y: y.as_posix(), sorted(x.glob('*.png'), key=getid))]

    def getflowpath(f):
        return f.replace('final', 'flow').replace('.png', '.flo')

    def getocclusionpath(f):
        return f.replace('final', 'occlusions')

    subroot = Path(root).glob('*')

    datalist = chain.from_iterable(map(chunker, map(getdirdata, subroot)))
    if test: return nsample([{'frame': (f1, f2)} for f1, f2 in datalist])
    datalist = [{'flow': getflowpath(f1), 'occlusion': getocclusionpath(f1), 'frame': (f1, f2)} for f1, f2 in datalist]
    return nsample(datalist)

def direct_read_n_sample_sintel(root, sample, test = False):
    if sample is None:
        raise Exception('Read only few sample')
    datapath = getSintelPairFrame(root, sample, test)
    if test:
        pass
    else:
        pass

def get_random_sintel(root,n = None):
    random.sample(dl,)




"""uses
sintelroot = "/data/keshav/sintel/training/final"
files = getSintelPairFrame(sintelroot)
"""
