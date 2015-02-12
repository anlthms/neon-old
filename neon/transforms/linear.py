# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Linear transform functions and classes.
"""

from neon.transforms.activation import Activation


class Linear(Activation):
    """
    Embodiment of a linear activation function.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        return

    def apply_derivative(self, backend, inputs, outputs):
        """
        Apply the rectified linear activation function derivative.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed
            outputs (array_like): Storage for the transformed output.
        """
        return

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
        return

    def pre_act_buffer(self, make_zbuf, output, dtype):
        """
        overrides the pre_act_buffer with output to save memory

        Arguments:
            make_zbuf (backend.zeros): Function to initialize pre_act_buffer.
            output (array_like): Output data buffer.
            dtype: dtype for pre_act_buffer
        """
        return output

    def bprop_func(self, pre_act, error):
        """
        Function to perform during the bprop

        Arguments:
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
        """
        return

