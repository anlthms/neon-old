"""
Rectified linear (ReLU) transform functions and classes.
"""

from mylearn.transforms.activation import Activation


def rectlin(dataset):
    """
    Applies rectified linear transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    # negative elements should be set to 0, positive remain unchanged
    return (dataset * (dataset > 0))


def rectlin_derivative(dataset):
    """
    Applies derivative of the rectified linear transform to the dataset passed.

    Arguments:
        dataset (array_like): Input data to be transformed

    Returns:
        array_like: Transformed copy of the dataset.  Will be in the same
                    format as the input dataset.
    """
    # negative elements should be set to 0, positive to 1
    return (1 * (dataset > 0))


class RectLin(Activation):
    """
    Embodiment of a rectified linear activation function.
    """

    @staticmethod
    def apply_function(dataset):
        """
        Apply the rectified linear activation function.
        """
        return rectlin(dataset)

    @staticmethod
    def apply_derivative(dataset):
        """
        Apply the rectified linear activation function derivative.
        """
        return rectlin_derivative(dataset)
