"""
Simple linear transform functions and classes.
"""

from mylearn.transforms.activation import Activation


def identity(backend, inputs, outputs):
    """
    Applies identity (i.e. no) transform to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """
    outputs[:] = inputs


def identity_derivative(backend, inputs, outputs):
    """
    Applies derivative of the identity linear transform to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed
        outputs (array_like): Storage for the transformed output.
    """
    outputs[:] = 1.0


def identity_and_derivative(backend, inputs, outputs):
    """
    Applies identity function and its derivative to the dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed. This also acts as
                             storage for the output of the derivative function.
        outputs (array_like): Storage for the transformed output.
    """
    # Apply the identity function.
    identity(backend, inputs, outputs)

    # Apply the derivative of the identity function.
    inputs[:] = 1.0


class Identity(Activation):
    """
    Embodiment of an identity linear activation function.
    """

    @staticmethod
    def apply_function(backend, inputs, outputs):
        """
        Apply the identity linear activation function.
        """
        return identity(backend, inputs, outputs)

    @staticmethod
    def apply_derivative(backend, inputs, outputs):
        """
        Apply the identity linear activation function derivative.
        """
        return identity_derivative(backend, inputs, outputs)

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Apply the identity activation function and its derivative.
        """
        return identity_and_derivative(backend, inputs, outputs)
