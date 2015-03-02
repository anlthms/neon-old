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
        req_param(self, ['layer'])

        self.backend = self.layer.backend
        self.is_local = self.layer.is_local

        if self.is_local:
            self.in_shape = (self.layer.nofm,
                             self.layer.ofmsize * self.layer.batch_size)
            self.in1d = (self.layer.nofm, 1)
        else:
            self.in_shape = self.layer.output.shape
            self.in1d = (self.nin, 1)

        self.train_mode = True
        self.nbatches = 0

        self._xhat = self.backend.zeros(self.in_shape, dtype='float32')

        self._mean = self.backend.zeros(self.in1d, dtype='float32')
        self._vars = self.backend.zeros(self.in1d, dtype='float32')

        # Global mean and var to be used during inference
        self._gmean = self.backend.zeros(self.in1d, dtype='float32')
        self._gvars = self.backend.zeros(self.in1d, dtype='float32')

        # learned params and their update buffers
        self._beta = self.backend.zeros(self.in1d, dtype='float32')
        self._gamma = self.backend.ones(self.in1d, dtype='float32')
        self.layer.params.extend([self._beta, self._gamma])

        self._beta_updates = self.backend.zeros(self.in1d, dtype='float32')
        self._gamma_updates = self.backend.zeros(self.in1d, dtype='float32')
        self.layer.updates.extend([self._beta_updates, self._gamma_updates])

    def set_inference_mode(self):
        self.train_mode = False
        if self._iscale is None:
            self.backend.divide(self._gvars, self.nbatches, self._gvars)
            unbiaser = self.layer.batch_size / (self.layer.batch_size - 1.)
            self.backend.multiply(self._gvars, unbiaser, self._gvars)
            self.backend.add(self._gvars, self._eps, self._gvars)
            self.backend.sqrt(self._gvars, out=self._gvars)
            self.backend.divide(self._gamma, self._gvars, self._gvars)
            self._iscale = self._gvars

            self.backend.divide(self._gmean, self.nbatches, self._gmean)
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
            backend.mean(inputs, axes=1, out=self._mean)
            backend.var(inputs, self._mean, axes=1, out=self._vars)

            # increment the global estimates (TODO: stop after an epoch)
            backend.add(self._gvars, self._vars, self._gvars)
            backend.add(self._gmean, self._vars, self._gmean)
            self.nbatches += 1

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

        # Compute the backpropagated error into error
        backend.multiply(self._xhat, self._gamma_updates, out=self._xhat)
        backend.add(self._xhat, self._beta_updates, out=self._xhat)
        backend.divide(self._xhat, self._xhat.shape[1], out=self._xhat)
        backend.subtract(error, self._xhat, out=error)
        backend.multiply(error, self._gamma, out=error)
        backend.multiply(error, self._vars, out=error)
