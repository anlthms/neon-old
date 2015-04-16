# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Batch normalization transform functions and classes.
"""

import logging
from neon.transforms.activation import Activation
from neon.util.param import req_param, opt_param
import numpy as np


logger = logging.getLogger(__name__)


class BatchNorm(Activation):

    """
    Embodiment of a BatchNormalization Transform

    Forward pass: (gamma/beta are scalar parameters for each unit.)
    x' <- (x-mean)/sqrt(var+eps)
    y <- gamma * x' + beta

    Backward pass:
    dy/dx = dy/dx' * dx'/dx
          = gamma * [ 1*(var+eps)^-1/2 + (x-mean) * (var+eps)^-3/2 * (2x)^-1/2]
          = gamma * [ 1*(var+eps)^-1/2 + (x-mean) * (var+eps)^-3/2 * (2x)^-1/2]
    but this simplifies a lot.
    """
    def initialize(self, kwargs):
        """
        Is called from WeightLayer.initialize
        with a reference to the layer.
        """
        self.__dict__.update(kwargs)
        self.dtype = self.layer.weight_dtype
        self.bigtype = np.float32 if self.dtype is np.float16 else self.dtype
        opt_param(self, ['_iscale', '_ishift'])
        opt_param(self, ['_eps'], 1e-6)
        req_param(self, ['layer'])

        self.backend = self.layer.backend
        self.is_local = self.layer.is_local
        self.batch_size = self.layer.batch_size
        if self.is_local:
            self.in1d = (self.layer.nofm, 1)
            self.ofmsize = self.layer.ofmsize
            self.orig_shape = (self.layer.nofm * self.ofmsize, self.batch_size)
            self.in_shape = (self.layer.nofm, self.ofmsize * self.batch_size)
        else:
            self.in_shape = (self.layer.nout, self.batch_size)
            self.in1d = (self.layer.nout, 1)

        self.train_mode = True
        logger.info("BatchNormalization set to train mode")
        self.nbatches = 0

        self._xhat = self.backend.zeros(self.in_shape, dtype=self.dtype)

        self._mean = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._vars = self.backend.zeros(self.in1d, dtype=self.bigtype)

        # Global mean and var to be used during inference
        self._gmean = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gvars = self.backend.zeros(self.in1d, dtype=self.bigtype)

        # learned params and their update buffers
        self._beta = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gamma = self.backend.ones(self.in1d, dtype=self.bigtype)
        self.layer.params.extend([self._beta, self._gamma])

        self._beta_updates = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self._gamma_updates = self.backend.zeros(self.in1d, dtype=self.bigtype)
        self.layer.updates.extend([self._beta_updates, self._gamma_updates])

    def set_inference_mode(self):
        """
        Appears to have a bug. Urs went through the code and everything matches
        the paper.
        """
        self.train_mode = False
        logger.info("BatchNormalization set to inference mode")
        if self._iscale is None:
            # normalize global variance -- inference scaling factor
            self.backend.divide(self._gvars, self.nbatches, self._gvars)
            m = self.batch_size
            if self.is_local:
                m *= self.ofmsize
            unbiaser = float(m / (m - 1.))
            self.backend.multiply(self._gvars, unbiaser, self._gvars)
            self.backend.add(self._gvars, self._eps, self._gvars)
            self.backend.sqrt(self._gvars, out=self._gvars)
            self.backend.divide(self._gamma, self._gvars, self._gvars)
            self._iscale = self._gvars

            # normalize global mean -- inference shiting factor
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
        For a fully connected layer, this is done by computing the mean and
        variance of the `inputs` over the mini-batch dimension,
        Mean and variance are also accumulated into a global estimate that is
        used for

        Arguments:
            backend (Backend): The backend class to use for computation.
            inputs (array_like): Input data to be transformed. This also
                                 acts as storage for the output of the
                                 derivative function.
            outputs (array_like): Storage for the transformed output.
        """
        if self.is_local:
            inputs = inputs.reshape(self.in_shape)
            outputs = outputs.reshape(self.in_shape)

        if self.train_mode:
            # Calc batch statistics
            backend.mean(inputs, axes=1, out=self._mean)
            backend.variance(inputs, axes=1, out=self._vars, mean=self._mean)
            # increment the global estimates (TODO: stop after an epoch)
            backend.add(self._gvars, self._vars, self._gvars)
            backend.add(self._gmean, self._mean, self._gmean)
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
            # Inference mode: Using accumulated scale and shift
            backend.multiply(inputs, self._iscale, out=outputs)
            backend.add(outputs, self._ishift, out=outputs)

        if self.is_local:
            inputs = inputs.reshape(self.orig_shape)
            outputs = outputs.reshape(self.orig_shape)

    def bprop_func(self, backend, pre_act, error, skip_act=False):
        if self.is_local:
            pre_act = pre_act.reshape(self.in_shape)
            error = error.reshape(self.in_shape)

        backend.multiply(self._xhat, error, out=pre_act)
        backend.sum(pre_act, axes=1, out=self._gamma_updates)
        backend.sum(error, axes=1, out=self._beta_updates)

        # Compute the backpropagated error into error
        backend.multiply(self._xhat, self._gamma_updates, out=self._xhat)
        backend.add(self._xhat, self._beta_updates, out=self._xhat)
        backend.divide(self._xhat, float(self._xhat.shape[1]), out=self._xhat)
        backend.subtract(error, self._xhat, out=error)
        backend.multiply(error, self._gamma, out=error)
        backend.divide(error, self._vars, out=error)

        if self.is_local:
            pre_act = pre_act.reshape(self.orig_shape)
            error = error.reshape(self.orig_shape)
