import numpy as np
import math

def sampleEqualRanges(train, no_ranges):
    n_rows = train[0].shape[0]
    range_width = int(math.floor(n_rows / no_ranges))
    ranges = range(range_width, n_rows, range_width)
    print "Sample steps: %d" % range_width
    return ranges

def sampleLog2Ranges(train, no_ranges):
    n_rows = train[0].shape[0]
    no_ranges = int(math.floor(math.log(n_rows, 2)))
    ranges = [int(math.pow(2, i)) for i in xrange(no_ranges)]
    return ranges

def sampleLog2AndEqualRanges(train, no_ranges):
    equal = sampleEqualRanges(train, no_ranges)
    log2 = sampleLog2Ranges(train, no_ranges)
    equal.extend(log2)
    equal = list(set(equal))
    equal.sort()
    return equal

def samplePNNRanges(train, no_ranges):
    return [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 28000, 32000]

def sampleSimplePNNRanges(train, no_ranges):
    #return [20000]
    #sizes = [1000, 2000, 8000, 10000, 20000, 30000, 40000, 50000, train[0].shape[0]]
    sizes = [2000, 5000, 10000, 15000, 20000, 30000]
    #sizes = [8000, 10000]
    return filter(lambda size: size <= train[0].shape[0], sizes)