# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains activation function related code.
"""


class Activation(object):
    """
    Abstract activation function class.  Defines operations any concrete
    activation function child must support.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.gain = 1.0

    def apply_function(self, backend, inputs, outputs):
        """
        Computes the activation function value by applying it to each element
        of the dataset passed.

        Arguments:
            dataset (array_like): The dataset upon which to apply the
                                  activation function.

        Returns:
            array_like: A transformed copy of the input dataset with the same
                        type and shape.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_function should be" +
                                  "overridden in child class.")

    def apply_derivative(self, backend, inputs, outputs):
        """
        Computes the activation function derivative value by applying it to
        each element of the dataset passed.

        Arguments:
            dataset (array_like): The dataset upon which to apply the
                                  activation function derivative.

        Returns:
            array_like: A transformed copy of the input dataset with the same
                        type and shape.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_derivative should be" +
                                  "overridden in child class.")

    def pre_act_buffer(self, make_zbuf, output, dtype):
        """
        Creates the pre_act_buffer

        Arguments:
            make_zbuf (backend.zeros): Function to initialize pre_act_buffer.
            output (array_like): Output data buffer.
            dtype: dtype for pre_act_buffer
        """
        return make_zbuf(output.shape, dtype)

    def fprop_func(self, backend, inputs, outputs):
        """
        Function to apply during fprop
        Typically computes the activation function and its derivative by
        applying it to each element of the dataset passed, but there are
        exceptions (RectLin).

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("apply_both should be" +
                                  "overridden in child class.")

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        """
        Function to apply during bprop
        Typically empty, but can be used to compute derivative during bprop
        instead of storing it during fprop (used in RectLin).

        Arguments:
            backend (Backend): The backend class to use for computation.
            pre_act (array_like): pre_activation buffer
            error (array_like): error buffer
            skip_act (Boolean): whether to skip the multiplication
        """
        if skip_act is False:
            backend.multiply(error, pre_act, out=error)
