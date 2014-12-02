# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Tanh transform functions and classes.
"""

from neon.transforms.activation import Activation


def tanh(backend, inputs, outputs):
    """
    Applies tanh transform to the dataset passed.
    (1.0 - np.exp(-2.0*x)) / (1.0 + np.exp(-2.0*x))

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """

    tmp = backend.zeros(inputs.shape)
    backend.multiply(inputs, backend.wrap(-2.0), out=tmp)
    backend.exp(tmp, out=tmp)
    backend.subtract(backend.wrap(1.0), tmp, out=outputs)
    backend.add(backend.wrap(1.0), tmp, out=tmp)
    backend.divide(outputs, tmp, out=outputs)


def tanh_derivative(backend, tanh, outputs):
    """
    Applies derivative of the tanh transform to the dataset passed.
    1 - tanh**2

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """

    backend.multiply(tanh, tanh, out=outputs)
    backend.subtract(backend.wrap(1.0), outputs, out=outputs)


def tanh_and_derivative(backend, inputs, outputs):
    """
    Applies tanh function and its derivative to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed. This also acts as
                             storage for the output of the derivative function.
        outputs (array_like): Storage for the transformed output.
    """

    tanh(backend, inputs, outputs)
    tanh_derivative(backend, outputs, inputs)


class Tanh(Activation):

    """
    Embodiment of a tanh activation function.
    """

    @staticmethod
    def apply_function(backend, inputs, outputs):
        """
        Apply the tanh activation function.
        """
        return tanh(backend, inputs, outputs)

    @staticmethod
    def apply_derivative(backend, inputs, outputs):
        """
        Apply the tanh activation function derivative.
        """
        return tanh_derivative(backend, inputs, outputs)

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Apply the tanh activation function and its derivative.
        """
        return tanh_and_derivative(backend, inputs, outputs)
