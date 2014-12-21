"""
Logistic transform functions and classes.
"""

from neon.transforms.activation import Activation


class Softmax(Activation):

    """
    Embodiment of a softmax activation function.
    """
    def __init__(self, shortcut_deriv=False):
        self.shortcutDeriv = shortcut_deriv
        self.tmp = None
        self.gain = 1.0

    def apply_function(self, backend, inputs, outputs):
        """
        Apply the softmax activation function.
        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed (the x's)
            outputs (array_like): Storage for the transformed output.
        """
        backend.softmax(inputs, out=outputs)

    def apply_derivative(self, backend, inputs, outputs):
        """
        Applies derivative of the softmax transform to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed (the x's)
            outputs (array_like): Storage for the transformed output.
        """

        # First need to compute the actual softmax to get the y's
        backend.softmax(inputs, outputs)

        # Since gradient calculates the derivative incorporating the errors,
        # we need to include an error matrix of 1's to get the correct value
        if not self.tmp or self.tmp.shape != inputs.shape:
            self.tmp = backend.ones(inputs.shape)

        backend.softmax_gradient(outputs, err=self.tmp, out=outputs)

    def apply_both(self, backend, inputs, outputs):
        """
        Apply the softmax activation function and its derivative.
        (If we're using shortcut, then don't bother to compute the derivative)
        """
        self.apply_function(backend, inputs, outputs)
        if self.shortcutDeriv is False:
            if not self.tmp or self.tmp.shape != inputs.shape:
                self.tmp = backend.ones(inputs.shape)
            backend.softmax_gradient(outputs, err=self.tmp, out=inputs)
        else:
            inputs[:] = 1.0
