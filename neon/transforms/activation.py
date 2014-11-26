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

    @staticmethod
    def apply_function(dataset):
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

    @staticmethod
    def apply_derivative(dataset):
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

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Computes the activation function and its derivative by applying it to
        each element of the dataset passed.

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
