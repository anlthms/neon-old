# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Hyperbolic tangent transform functions and classes.
"""

import numpy

from neon.transforms.activation import Activation


def tanh(dataset):
    """
    Applies the hyperbolic tangent transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    if isinstance(dataset, (int, float, numpy.ndarray)):
        exp_ds = numpy.exp(-2 * dataset)
    else:
        exp_ds = (-2 * dataset).exp()
    return (1.0 - exp_ds) / (1.0 + exp_ds)


def tanh_derivative(dataset):
    """
    Applies derivative of the hyperbolic tangent transform to the dataset
    passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    res = tanh(dataset)
    return 1.0 - res * res


class Tanh(Activation):
    """
    Embodiment of a hyperbolic tangent activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the hyperbolic tangent activation function.
        """
        return tanh(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the hyperbolic tangent activation function derivative.
        """
        return tanh_derivative(dataset)
