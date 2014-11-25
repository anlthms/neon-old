# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Rectified linear (ReLU) transform functions and classes.
"""

from neon.transforms.activation import Activation


class RectLin(Activation):
    """
    Embodiment of a rectified linear activation function.
    """

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin(inputs, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function derivative.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin_derivative(inputs, outputs)

    def apply_both(self, backend, inputs, outputs):
        """
        Applies the rectified linear transform and its derivative to the
        dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also acts
                                 as storage for the output of the derivative
                                 function.
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectlin(inputs, outputs)
        backend.rectlin_derivative(inputs, inputs)
