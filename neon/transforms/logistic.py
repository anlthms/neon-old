# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Logistic transform functions and classes.
"""

from neon.transforms.activation import Activation


def logistic(backend, inputs, outputs):
    """
    Applies logistic transform to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """
    backend.logistic(inputs, out=outputs)


def logistic_derivative(backend, inputs, outputs):
    """
    Applies derivative of the logistic transform to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """
    tmp = backend.zeros(inputs.shape)
    backend.logistic(inputs, outputs)
    backend.subtract(backend.wrap(1.0), outputs, out=tmp)
    backend.multiply(outputs, tmp, outputs)


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
    logistic(backend, inputs, outputs)

    # Apply the derivative of the logistic function, storing the result in
    # inputs
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
