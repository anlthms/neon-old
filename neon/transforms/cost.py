# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains cost or loss function related code.
"""
from neon.util.param import opt_param, req_param


from neon.util.param import opt_param


class Cost(object):
    """
    Abstract cost function class.  Defines operations any concrete
    cost function child must support.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        opt_param(self, ['temp_dtype', 'outputbuf', 'temp'], None)
        opt_param(self, ['scale'], 1.0)

        opt_param(self, ['half_precision'], False)
        if self.half_precision:
            import numpy as np
            setattr(self, 'temp_dtype', np.float16)

    def initialize(self, kwargs):
        self.__dict__.update(kwargs)
        opt_param(self, ['backend'], self.olayer.backend)
        opt_param(self, ['batch_size'], self.olayer.batch_size)
        opt_param(self, ['olayer_data'], 'output')
        req_param(self.olayer, [self.olayer_data])
        # if not hasattr(self.olayer, self.olayer_data):
        #     raise ValueError("Layer %s does not have buffer %s" %
        #                      (self.olayer.name, self.olayer_data))
        # else:
        self.set_outputbuf(getattr(self.olayer, self.olayer_data))

    def set_outputbuf(self, databuf):
        """
        Called when we need to change the data that the cost function is
        operating on.
        In the derived costs, this will reallocate the temporary storage if
        the outputbuf shape changes (hopefully infrequently)
        """
        self.outputbuf = databuf

    def apply_function(self, targets):
        """
        Computes the cost function value by applying it pairwise against
        correspondsing elements of the outputs and targets datasets passed.
        Outputs and targets must have the same shape.

        Arguments:
            outputs (array_like): The dataset containing predicted values.
            targets (array_like): The dataset containing true outcome values.

        Returns:
            array_like: The cost values evaluated at each pair of the input
                        datasets.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("Should be overridden in child class.")

    def apply_derivative(self, targets):
        """
        Computes the cost function derivative value by applying it to
        each corresponding element of the predicted outputs and known
        target outcomes.  Outputs and targets must have the same shape.

        Arguments:
            outputs (array_like): The dataset containing predicted values.
            targets (array_like): The dataset containing true outcome values.
            temp (array_like): Storage for intermediate results.

        Returns:
            array_like: The derivative cost values evaluated at each pair of
                        the input datasets.

        Raises:
            NotImplementedError: Must be implemented in a child class.
        """
        raise NotImplementedError("Should be overridden in child class.")
