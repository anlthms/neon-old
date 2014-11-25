# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Logistic transform functions and classes.
"""

from neon.transforms.activation import Activation


class Logistic(Activation):

    """
    Embodiment of a logistic activation function.
    """
    def __init__(self, use_binary=True, shortcut_deriv=False):
        self.tmp = None

    def apply_function(self, backend, inputs, outputs):
        """
        Applies logistic transform to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.logistic(inputs, out=outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Applies derivative of the logistic transform to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        if not self.tmp or self.tmp.shape != inputs.shape:
            self.tmp = backend.zeros(inputs.shape)

        backend.logistic(inputs, outputs)
        backend.subtract(backend.wrap(1.0), outputs, out=self.tmp)
        backend.multiply(outputs, self.tmp, outputs)

    def apply_both(self, backend, inputs, outputs):
        """
        Applies logistic function and its derivative to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.
        """
        # Apply the logistic function.
        backend.logistic(backend, inputs, outputs)

        # Apply the derivative of the logistic function, storing the result in
        # inputs
        backend.subtract(backend.wrap(1.0), outputs, out=inputs)
        backend.multiply(inputs, outputs, out=inputs)
