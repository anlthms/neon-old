"""
Utility functions for gaussian operations.
"""

import math
import numpy


def gauss(x, mu, sigma):
    """
    Gaussian density function.
    """

    assert type(x) == float
    diff = x - mu
    denom = sigma * math.sqrt(2 * numpy.pi)
    return numpy.exp(-(diff * diff / (2 * sigma * sigma))) / denom


def gaussian_filter(shape):
    """
    Return a filter that follows the gaussian distribution, with
    the peak at the center of the filter.
    """

    assert len(shape) == 2
    height, width = shape
    assert height % 2 == 1
    assert width % 2 == 1
    midx = width / 2
    midy = height / 2
    # Setting this to a higher value causes the filter to
    # cover more of the gaussian curve.
    spread = 3.0
    sigma = math.sqrt(midx * midx + midy * midy) / spread

    filter = numpy.zeros(shape)
    for i, j in numpy.ndindex(height, width):
        # NOTE: This can be optimized by computing a quarter of the output
        # matrix and then mirroring to fill out the rest.
        xdiff = j - midx
        ydiff = i - midy
        dist = math.sqrt(xdiff * xdiff + ydiff * ydiff)
        filter[i, j] = gauss(dist, 0, sigma)
    return filter
