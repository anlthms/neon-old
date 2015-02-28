# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Batch normalization transform functions and classes.
"""

from neon.transforms.activation import Activation
from neon.util.param import req_param, opt_param


class BatchNorm(Activation):

    """
    Embodiment of a BatchNormalization Transform
    """
    def initialize(self, kwargs):
        self.__dict__.update(kwargs)
        opt_param(self, ['_iscale', 'ishift'])
        opt_param(self, ['_eps'], 1e-6)
        req_param(self, ['in_shape', 'update_list', 'param_list'])
        self.nin = self.in_shape[0]
        self._eps = 1e-6
        self.train_mode = True

        self._xhat = self.backend.zeros(self.in_shape, dtype='float32')

        self._mean = self.backend.zeros((self.nin, 1), dtype='float32')
        self._vars = self.backend.zeros((self.nin, 1), dtype='float32')
        self._beta = self.backend.zeros((self.nin, 1), dtype='float32')
        self._gamma = self.backend.ones((self.nin, 1), dtype='float32')

        # Global mean and var to be used during inference
        self._gmean = self.backend.zeros((self.nin, 1), dtype='float32')
        self._gvars = self.backend.zeros((self.nin, 1), dtype='float32')

        # learned params and their update buffers
        self._beta = self.backend.zeros((self.nin, 1), dtype='float32')
        self._gamma = self.backend.ones((self.nin, 1), dtype='float32')
        self.param_list.extend([self._beta, self._gamma])

        self._beta_updates = self.backend.zeros((self.nin, 1), dtype='float32')
        self._gamma_updates = self.backend.zeros((self.nin, 1), dtype='float32')
        self.update_list.extend([self._beta_updates, self._gamma_updates])

    def set_inference_mode(self):
        self.train_mode = False
        if self._iscale is None:
            self.backend.add(self._gvars, self._eps, self._gvars)
            self.backend.sqrt(self._gvars, out=self._gvars)
            self.backend.divide(self._gamma, self._gvars, self._gvars)
            self._iscale = self._gvars

            self.backend.multiply(self._gmean, self._gvars, self._gmean)
            self.backend.subtract(self._beta, self._gmean, self._gmean)
            self._ishift = self._gmean

    def apply_function(self, backend, inputs, outputs):
        pass

    def apply_derivative(self, backend, inputs, outputs):
        pass

    def fprop_func(self, backend, inputs, outputs):
        """
        Applies BatchNorm function and its derivative to the dataset passed.

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.
        """
        if self.train_mode:
            # Calc batch statistics
            print inputs.shape, self._mean.shape
            backend.mean(inputs, axes=1, out=self._mean)
            backend.var(inputs, axes=1, out=self._vars)

            # adjust the global estimates
            backend.add(self._gvars, self._vars, self._gvars)
            backend.add(self._gmean, self._vars, self._gmean)

            # Just store sqrt(vars + eps) since it's used as a unit
            backend.add(self._vars, self._eps, self._vars)
            backend.sqrt(self._vars, out=self._vars)

            # Every operation below uses broadcasting over minibatch dim
            backend.subtract(inputs, self._mean, out=self._xhat)
            backend.divide(self._xhat, self._vars, out=self._xhat)
            backend.multiply(self._xhat, self._gamma, out=outputs)
            backend.add(outputs, self._beta, out=outputs)
        else:
            backend.multiply(inputs, self._iscale, out=outputs)
            backend.add(outputs, self._ishift, out=outputs)

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        backend.multiply(self._xhat, error, out=pre_act)
        backend.sum(pre_act, axes=1, out=self._gamma_updates)
        backend.sum(error, axes=1, out=self._beta_updates)

        # Compute the backpropagated error into _xhat
        backend.multiply(self._xhat, self._gamma_updates, out=self._xhat)
        backend.add(self._xhat, self._beta_updates, out=self._xhat)
        backend.divide(self._xhat, self._xhat.shape[1], out=self._xhat)
        backend.subtract(error, self._xhat, out=self._xhat)
        backend.multiply(self._xhat, self._gamma, out=self._xhat)
        backend.multiply(self._xhat, self._vars, out=self._xhat)
