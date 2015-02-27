# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Leaky Rectified linear (LReLU) transform functions and classes.
"""

from neon.transforms.activation import Activation


class RectLeaky(Activation):

    """
    Embodiment of a leaky rectified linear activation function.
    """

    def __init__(self, slope=0.01, **kwargs):
        self.slope = slope
        self.__dict__.update(kwargs)

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the leaky rectified linear activation function.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky(inputs, self.slope, outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Apply the leaky rectified linear activation function derivative.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky_derivative(inputs, self.slope, outputs)

    def fprop_func(self, backend, inputs, outputs):
        """
        Function to apply during fprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also acts
                                 as storage for the output of the derivative
                                 function.
            outputs (array_like): Storage for the transformed output.
        """
        backend.rectleaky(inputs, self.slope, outputs)

    def pre_act_buffer(self, backend, output, dtype):
        """
        overrides the pre_act_buffer with output to save memory

        Arguments:
            backend (Backend): The backend class to use for computation.
            output (array_like): Output data buffer.
            dtype: dtype for pre_act_buffer
        """
        return output

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Function to perform during the bprop

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
            skip_act (Boolean): whether to skip the multiplication
        """
        super(RectLeaky, self).bprop_func(backend, pre_act, error, skip_act)
