"""
Logistic transform functions and classes.
"""

from math import log
import numpy

from mylearn.transforms.activation import Activation


def logistic(dataset):
    """
    Applies logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    if isinstance(dataset, (int, float, numpy.ndarray)):
        neg_exp = numpy.exp(- dataset)
    else:
        neg_exp = (- dataset).exp()
    return 1.0 / (1.0 + neg_exp)


def logistic_derivative(dataset):
    """
    Applies derivative of the logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    return logistic(dataset) * (1 - logistic(dataset))


def logistic_and_derivative(backend, inputs, outputs):
    """
    Applies logistic function and its derivative to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed. This also acts as
                             storage for the output of the derivative function.
        outputs (array_like): Storage for the transformed output.
    """
    # Apply the logistic function.
    # FIXME: unnecessay wrapping.
    backend.multiply(inputs, backend.wrap(-1.0), out=outputs)
    backend.exp(outputs, out=outputs)
    backend.add(outputs, backend.wrap(1.0), out=outputs)
    #backend.divide(backend.wrap(1.0), outputs, out=outputs)
    backend.reciprocal(outputs, out=outputs)

    # Apply the derivative of the logistic function.
    backend.subtract(backend.wrap(1.0), outputs, out=inputs)
    backend.multiply(inputs, outputs, out=inputs)


def pseudo_logistic(dataset):
    """
    Applies faster, approximate logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    return (1.0 / (1.0 + 2 ** (- dataset)))


def pseudo_logistic_derivative(dataset):
    """
    Applies derivative of the approximate logistic transform to the dataset
    passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    res = pseudo_logistic(dataset)
    return (log(2) * res * (1.0 - res))


class Logistic(Activation):
    """
    Embodiment of a logistic activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the logistic activation function.
        """
        return logistic(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the logistic activation function derivative.
        """
        return logistic_derivative(backend, dataset)

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Apply the logistic activation function and its derivative.
        """
        return logistic_and_derivative(backend, inputs, outputs)


class PseudoLogistic(Activation):
    """
    Embodiment of an approximate logistic activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the approximate logistic activation function.
        """
        return pseudo_logistic(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the approximate logistic activation function derivative.
        """
        return pseudo_logistic_derivative(dataset)
