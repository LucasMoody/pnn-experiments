import numpy as np
import math

def sampleEqualRanges(train, no_ranges):
    n_rows = train[0].shape[0]
    range_width = int(math.floor(n_rows / no_ranges))
    ranges = range(range_width, n_rows, range_width)
    ranges.append(n_rows)
    print "Sample steps: %d" % range_width
    return ranges