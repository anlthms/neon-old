"""
Rectified linear (ReLU) transform functions and classes.
"""

from neon.transforms.activation import Activation


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


def rectlin_and_derivative(backend, inputs, outputs):
    """
    Applies the rectified linear transform and its derivative to the
    dataset passed.

    Arguments:
        backend (Backend): The backend class to use for computation.
        inputs (array_like): Input data to be transformed. This also acts as
                             storage for the output of the derivative function.
        outputs (array_like): Storage for the transformed output.
    """
    # Rectified linear.
    backend.greater(inputs, backend.wrap(0), outputs)
    backend.multiply(inputs, outputs, outputs)

    # The derivative of rectified linear.
    backend.greater(inputs, backend.wrap(0), inputs)


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

    @staticmethod
    def apply_both(backend, inputs, outputs):
        """
        Apply the rectified linear activation function and its derivative.
        """
        return rectlin_and_derivative(backend, inputs, outputs)
