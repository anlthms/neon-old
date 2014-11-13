#from neon.models.layer import Layer
import numpy as np
from neon.transforms.logistic import Logistic as neonLogistic
from neon.transforms.tanh import Tanh as neonTanh
from neon.transforms.rectified import RectLin as neonRectlin
from neon.transforms.linear import Identity and neonIdentity


class Layer:
    """Generic Layer class

    Ensures inheriting class implement a
    forward() method.
    """
    def forward(self, X):
        raise NotImplementedError()

class NeuralLayer(Layer):
    """Balance Network Layer

    n_input : int
        Dimensionality if input

    n_output : int
        Dimensionality of output

    backend : Backend
        neon backend for computation

    W : array-like, shape (n_input, n_output), optional
        Pre-specified weight matrix

    b : array-like, shape (n_output), optional
        Pre-specified bias

    max_norm : float, optional
        Constrain the L2 norm of input weight vectors to not exceed this value

    weight_clip : float, optional
        Constrain individual weight values to not exceed this value

    adadelta_eps : float, optional
        Epsilon term in Adadelta, set to some small positive value
        to turn on adadelta

    adadelta_rho : float, optional
        Rho term in Adadelta

    momentum : float, optional
        Momentum factor

    learning_rate : float, optional
        Learning rate parameter in gradient descent

    rng : numpy.random.RandomState, optional
        Random number generate to use

    dtype : numpy.dtype, optional
        Datatype of the layer (ex: np.float32)
    """
    def __init__(self, n_input, n_output, backend
        W=None, b=None, max_norm=None, weight_clip=None,
        adadelta_eps=None, adadelta_rho=0.95,
        momentum=None, learning_rate=None,
        init_alpha = .1, rng=np.random.RandomState(817), dtype=np.float32,
        learning_rule, lr_params):
        # Internal state variables for backward pass
        self.backend = backend
        self.X = None
        self.prestate = None
        self.state = None

        if W is None:
            self.W = backend.array(np.sqrt(init_alpha/(n_input+n_output))*
                                   rng.randn(n_input,n_output).astype(dtype))
        else:
            # No recast to preserve reference (for tieing autoencoder weights)
            self.W = W
        if b is None:
            self.b = backend.zeros(n_output, dtype)
        else:
            # No recast to preserve reference (for tieing autoencoder biases)
            self.b = b

        self.max_norm = max_norm
        self.weight_clip = weight_clip

        assert (adadelta_eps is None) or (momentum is None)
        if adadelta_eps is not None:
            self.adadelta_eps = backend.wrap(adadelta_eps)
            assert adadelta_rho is not None
            self.adadelta_rho = backend.wrap(adadelta_rho)
        else:
            self.adadelta_eps = None
        if momentum is not None:
            self.momentum = backend.wrap(momentum)
        else:
            self.momentum = None
        if learning_rate is not None:
            self.learning_rate = backend.wrap(learning_rate)
        else:
            self.learning_rate = None

        self.rng = rng
        self.dtype = dtype

        self.dlossdW = backend.zeros_like(self.W)
        self.dlossdb = backend.zeros_like(self.b)
        # For momentum
        self.velocitydW = backend.zeros_like(self.W)
        self.velocitydb = backend.zeros_like(self.b)
        # For adadelta
        self.dlossdW2 = backend.zeros_like(self.W)
        self.dlossdb2 = backend.zeros_like(self.b)
        self.dW2 = backend.zeros_like(self.W)
        self.db2 = backend.zeros_like(self.b)

    def forward(self, X):
        """Forward propagate a given input

        Parameters
        ----------
        X : array-like, shape (n_samples, n_inputs)
            Input from the layer below

        Returns
        -------
        hidden_state : array-like, shape (n_samples, n_outputs)
            Output state from the layer
        """
        self.X = X
        inputs = self.backend.append_bias(self.b)
        self.backend.dot(self.X, self.W, self.pre_state)
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
        deriv = self.nonlinear.apply_derivative(self.backend, self.prestate)
        dlossdz = dlossdh*deriv
        dWupdate = self.backend.dot(self.X.T, dlossdz)
        dbupdate = self.backend.sum(dlossdz, axis=0)
        if accumulate:
            self.backend.add(self.dlossdW, dWupdate, self.dlossdW)
            self.backend.add(self.dlossdb, dbupdate, self.dlossdb)
        else:
            self.dlossdW = dWupdate
            self.dlossdb = dbupdate
        return self.backend.dot(dlossdz, self.W.T)

    def update_b(self):
        """Update bias parameters
        """
        if self.adadelta_eps is not None:
            old = self.adadelta_rho*self.dlossdb2
            new = (1.-self.adadelta_rho)*self.dlossdb*self.dlossdb
            self.backend.add(old, new, self.dlossdb2)
            rms_dlossdb2 = self.dlossdb2+self.adadelta_eps
            self.backend.sqrt(rms_dlossdb2, rms_dlossdb2)
            rms_db2 = self.db2+self.adadelta_eps
            rms_db2 = self.backend.sqrt(rms_db2, rms_db2)
            learning_rate_b = rms_db2/rms_dlossdb2
        else:
            learning_rate_b = self.learning_rate

        if self.momentum is not None:
            self.backend.subtract(self.momentum*self.velocitydb, learning_rate_b*self.dlossdb, self.velocitydb)
            deltab = self.velocitydb
        else:
            deltab = -learning_rate_b*self.dlossdb

        if self.adadelta_eps is not None:
            self.backend.add(self.adadelta_rho*self.db2, (1.-self.adadelta_rho)*(deltab**2), self.db2)
        
        self.b += deltab

    def update_W(self):
        """Update weight parameters
        TODO : neon integration
        """
        raise NotImplementedError
        if self.adadelta_eps is not None:
            # Fix dlossdW2 to initialize to first sample instead of zeros
            self.dlossdW2 = self.adadelta_rho*self.dlossdW2 + (1.-self.adadelta_rho)*(self.dlossdW**2)
            rms_dlossdW2 = np.sqrt(self.dlossdW2 + self.adadelta_eps)
            rms_dW2 = np.sqrt(self.dW2 + self.adadelta_eps)
            learning_rate_W = rms_dW2/rms_dlossdW2
        else:
            learning_rate_W = self.learning_rate

        if self.momentum is not None:
            self.velocitydW = self.momentum*self.velocitydW - learning_rate_W*self.dlossdW
            deltaW = self.velocitydW
        else:
            deltaW = -learning_rate_W*self.dlossdW

        if self.adadelta_eps is not None:
            self.dW2 = self.adadelta_rho*self.dW2 + (1.-self.adadelta_rho)*(deltaW**2)

        self.W += deltaW

        if self.max_norm is not None:
            w_norm = np.sqrt((self.W**2).sum(0,keepdims=True))
            w_norm[w_norm < self.max_norm] = self.max_norm
            self.W *= self.max_norm/w_norm
        if self.weight_clip is not None:
            self.W[self.W > self.weight_clip] = self.weight_clip

class Softmax(NeuralLayer):
    def __init__(self, *args):
        raise NotImplementedError
        self.nonlinear = neonSoftMax()
        super(Softmax, self).__init__(*args)

class Relu(NeuralLayer):
    def __init__(self, *args):
        self.nonlinear = neonRectLin()
        super(Relu, self).__init__(*args)

class Sigmoid(NeuralLayer):
    def __init__(self, *args):
        self.nonlinear = neonLogistic()
        super(Sigmoid, self).__init__(*args)

class Tanh(NeuralLayer):
    def __init__(self, *args):
        self.nonlinear = neonTanh()
        super(Tahn, self).__init__(*args)

class Linear(NeuralLayer):
    def __init__(self, *args):
        self.nonlinear = neonIdentity()
        super(Tahn, self).__init__(*args)

class RankOut(NeuralLayer):
    def __init__(self, *args):
        raise NotImplementedError
    def nonlinear(self,X):
        z = np.zeros_like(X)
        z[np.arange(X.shape[0]),np.argmax(X,axis=1)] = 1.
        return X*z

    def dnonlinear(self,X):
        return 1.

class LazyRelu(Relu):
    def __init__(self, *args):
        raise NotImplementedError
    def dnonlinear(self,X):
        return 1.

class LazySigmoid(Sigmoid):
    def __init__(self, *args):
        raise NotImplementedError
    def dnonlinear(self,X):
        return 1.

class ReluSigmoid(Sigmoid):
    def __init__(self, *args):
        raise NotImplementedError
    def nonlinear(self,X):
        return X*(X>0.)
