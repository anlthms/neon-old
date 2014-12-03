# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Contains cost or loss function related code.
"""


class Cost(object):
    """
    Abstract cost function class.  Defines operations any concrete
    cost function child must support.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for req_param in ['olayer']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

        if not hasattr(self, 'backend'):
            self.backend = self.olayer.backend

        if not hasattr(self, 'temp_dtype'):
            self.temp_dtype = None

        if not hasattr(self, 'batch_size'):
            self.batch_size = self.olayer.batch_size

        self.outputbuf = None
        if not hasattr(self, 'olayer_data'):
            self.set_outputbuf(getattr(self.olayer, 'output'))
        else:
            if not hasattr(self.olayer, self.olayer_data):
                raise ValueError("Layer %s does not have buffer %s" %
                                 (self.olayer.name, self.olayer_data))

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
