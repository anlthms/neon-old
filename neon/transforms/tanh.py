# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Tanh transform functions and classes.
"""

from neon.transforms.activation import Activation


class Tanh(Activation):

    """
    Embodiment of a tanh activation function.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.gain = 1.0

    def apply_function(self, backend, inputs, outputs):
        """
        Applies the hyperbolic tangent transform to the dataset passed.

        Arguments:
            inputs (array_like): Input data to be transformed

        Returns:
            array_like: Transformed copy of the inputs.  Will be in the same
                        format as the input inputs.
        """
        backend.clip(inputs, -10, 10, out=inputs)
        backend.tanh(inputs, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Applies derivative of the hyperbolic tangent transform to the inputs
        passed.

        Arguments:
            inputs (array_like): Input data to be transformed

        Returns:
            array_like: Transformed copy of the inputs.  Will be in the same
                        format as the input inputs.
        """
        backend.clip(inputs, -10, 10, out=inputs)
        backend.tanh(inputs, outputs)
        backend.multiply(outputs, outputs, outputs)
        backend.subtract(1.0, outputs, outputs)

    def apply_both(self, backend, inputs, outputs):
        backend.clip(inputs, -10, 10, out=inputs)
        backend.tanh(inputs, outputs)
        backend.multiply(outputs, outputs, inputs)
        backend.subtract(1.0, inputs, inputs)
