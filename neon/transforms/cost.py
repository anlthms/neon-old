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

    @staticmethod
    def apply_function(outputs, targets):
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

    @staticmethod
    def apply_derivative(outputs, targets, temp):
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
