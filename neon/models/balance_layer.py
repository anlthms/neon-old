from neon.models.layer import Layer
import numpy as np
from neon.transforms.logistic import Logistic
from neon.transforms.tanh import Tanh
from neon.transforms.rectified import RectLin
from neon.transforms.linear import Identity
from neon.transforms.softmax import SoftMax


class BalanceLayer(Layer):
    """Balance Network Layer

    Parameters
    ----------
    n_input : int
        Dimensionality if input
    n_output : int
        Dimensionality of output
    backend : Backend
        neon backend for computation
    w_learning_rule : LearningRule
        Object to update W.
    b_learning_rule : LearningRule
        Object to update b.
    W : array-like, shape (n_input, n_output), optional
        Pre-specified weight matrix
    b : array-like, shape (n_output), optional
        Pre-specified bias
    max_norm : float, optional
        Constrain the L2 norm of input weight vectors to not exceed this value
    weight_clip : float, optional
        Constrain individual weight values to not exceed this value
    rng : numpy.random.RandomState, optional
        Random number generate to use
    dtype : numpy.dtype, optional
        Datatype of the layer (ex: np.float32)
    """
    def __init__(self, n_input, n_output, backend,
                 w_learning_rule, b_learning_rule, w=None, b=None,
                 max_norm=None, weight_clip=None,
                 rng=np.random.RandomState(817), dtype=np.float32):
        # Internal state variables for backward pass
        self.backend = backend

        # Need batch size info to preallocate these variables
        raise NotImplementedError
        self.X = None
        self.prestate = None
        self.state = None
        self.deriv = None
        self.dWupdate = None
        self.dbupdate = None
        init_alpha = 0.1

        if w is None:
            self.W = backend.array(np.sqrt(init_alpha / (n_input + n_output)) *
                                   rng.randn(n_input, n_output).astype(dtype))
        else:
            # No recast to preserve reference (for tieing autoencoder weights)
            self.W = w
        if b is None:
            self.b = backend.zeros(n_output, dtype)
        else:
            # No recast to preserve reference (for tieing autoencoder biases)
            self.b = b
        self.W_lr = w_learning_rule
        self.W_lr.allocate_state(self.W)
        self.b_lr = b_learning_rule
        self.b_lr.allocate_state(self.b)

        self.max_norm = max_norm
        self.weight_clip = weight_clip

        self.rng = rng
        self.dtype = dtype

        self.dlossdW = backend.zeros_like(self.W)
        self.dlossdb = backend.zeros_like(self.b)

    def forward(self, x):
        """Forward propagate a given input

        Parameters
        ----------
        x : array-like, shape (n_samples, n_inputs)
            Input from the layer below

        Returns
        -------
        hidden_state : array-like, shape (n_samples, n_outputs)
            Output state from the layer
        """
        self.X = x
        # inputs = self.backend.append_bias(self.b)
        self.backend.fprop_fc_dot(self.X, self.W, self.pre_state)
        self.backend.add(self.pre_state, self.b, self.pre_state)
        self.nonlinear.apply_function(self.backend, self.prestate, self.state)
        return self.state

    def backward(self, dlossdh, accumulate=False):
        """Given dlossdh of layer_i, computes dlossdh of layer_i-1. This
        function relies on several internal state variables and should
        only be called after forward() has been called to define these
        state variables.

        Parameters
        ----------
        dlossdh : array-like, shape (n_samples, n_outputs)
        Derivative of the loss w.r.t the hidden state
        accumulate : boolean, optional
        Accumulate onto already existing derivative

        Returns
        -------
        dlossdhb : array-like, shape (n_samples, n_inputs)
        Derivative of the loss w.r.t the hidden state below
        """
        self.nonlinear.apply_derivative(self.backend,
                                        self.prestate, self.deriv)
        self.backend.multiply(dlossdh, self.deriv, self.dlossdz)
        self.backend.dot(self.X.T, self.dlossdz, self.dWupdate)
        self.backend.sum(self.dlossdz, self.dbupdate, axis=0)
        if accumulate:
            self.backend.add(self.dlossdW, self.dWupdate, self.dlossdW)
            self.backend.add(self.dlossdb, self.dbupdate, self.dlossdb)
        else:
            # Is there a better way to do this value set?
            self.backend.add(self.dWupdate, 0., self.dlossdW)
            self.backend.add(self.dbupdate, 0., self.dlossdb)
        return self.backend.dot(self.dlossdz, self.W.T)

    def update(self, epoch):
        """Update layer parameters
        """
        self.b_lr.apply_rule(self.b, self.dlossdb, epoch)
        self.W_lr.apply_rule(self.W, self.dlossdW, epoch)
        if self.max_norm is not None:
            raise NotImplementedError
            w_norm = np.sqrt((self.W**2).sum(0, keepdims=True))
            w_norm[w_norm < self.max_norm] = self.max_norm
            self.W *= self.max_norm/w_norm
        if self.weight_clip is not None:
            raise NotImplementedError
            self.W[self.W > self.weight_clip] = self.weight_clip


class SoftmaxLayer(BalanceLayer):
    def __init__(self, *args):
        raise NotImplementedError
        self.nonlinear = SoftMax()
        super(SoftmaxLayer, self).__init__(*args)


class ReluLayer(BalanceLayer):
    def __init__(self, *args):
        self.nonlinear = RectLin()
        super(ReluLayer, self).__init__(*args)


class SigmoidLayer(BalanceLayer):
    def __init__(self, *args):
        self.nonlinear = Logistic()
        super(SigmoidLayer, self).__init__(*args)


class TanhLayer(BalanceLayer):
    def __init__(self, *args):
        self.nonlinear = Tanh()
        super(TanhLayer, self).__init__(*args)


class LinearLayer(BalanceLayer):
    def __init__(self, *args):
        self.nonlinear = Identity()
        super(LinearLayer, self).__init__(*args)


class RankOut(BalanceLayer):
    def __init__(self, *args):
        raise NotImplementedError

    def nonlinear(self, x):
        z = np.zeros_like(x)
        z[np.arange(x.shape[0]), np.argmax(x, axis=1)] = 1.
        return x*z

    def dnonlinear(self, x):
        return 1.


# class LazyRelu(Relu):
#     def __init__(self, *args):
#         raise NotImplementedError

#     def dnonlinear(self, x):
#         return 1.


# class LazySigmoid(Sigmoid):
#     def __init__(self, *args):
#         raise NotImplementedError

#     def dnonlinear(self, x):
#         return 1.


# class ReluSigmoid(Sigmoid):
#     def __init__(self, *args):
#         raise NotImplementedError

#     def nonlinear(self, x):
#         return x*(x > 0.)
