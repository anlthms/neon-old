"""
Logistic transform functions and classes.
"""

from math import log
import numpy

from mylearn.transforms.activation import Activation


def logistic(backend, inputs, outputs):
    """
    Applies logistic transform to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """
    backend.multiply(inputs, backend.wrap(-1.0), out=outputs)
    backend.exp(outputs, out=outputs)
    backend.add(outputs, backend.wrap(1.0), out=outputs)
    backend.reciprocal(outputs, out=outputs)


def logistic_derivative(backend, inputs, outputs):
    """
    Applies derivative of the logistic transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    logistic_and_derivative(backend, inputs, outputs)

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
    backend.reciprocal(outputs, out=outputs)

    # Apply the derivative of the logistic function.
    backend.subtract(backend.wrap(1.0), outputs, out=inputs)
    backend.multiply(inputs, outputs, out=inputs)


class Logistic(Activation):
    """
    Embodiment of a logistic activation function.
    """

    @staticmethod
    def apply_function(backend, inputs, outputs):
        """
        Apply the logistic activation function.
        """
        return logistic(backend, inputs, outputs)

    @staticmethod
    def apply_derivative(backend, inputs, outputs):
        """
        Apply the logistic activation function derivative.
        """
        return logistic_derivative(backend, inputs, outputs)

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Apply the logistic activation function and its derivative.
        """
        return logistic_and_derivative(backend, inputs, outputs)

