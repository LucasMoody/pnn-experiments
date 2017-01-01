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
    ranges = [math.pow(2, i) for i in xrange(no_ranges)]
    return ranges

def sampleLog2AndEqualRanges(train, no_ranges):
    equal = sampleEqualRanges(train, no_ranges)
    log2 = sampleLog2Ranges(train, no_ranges)
    equal.extend(log2)
    equal = list(set(equal))
    equal.sort()
    return equal