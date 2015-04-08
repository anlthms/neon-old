# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Neon backend wrapper for the NervanaGPU library. Most functions are thin
wrappers around functions from the NervanaGPU class, the GPUTensor is taken
directly from NervanaGPU as well.
NervanaGPU is available at `<https://github.com/NervanaSystems/nervanagpu>`
"""
import logging

from neon.backends.backend import Backend
from nervanagpu import NervanaGPU, GPUTensor
from neon.diagnostics.timing_decorators import FlopsDecorator
import pycuda.driver as drv
import numpy as np


logger = logging.getLogger(__name__)

class GPU(Backend):
    """
    Sets up a NervanaGPU based backend for matrix operations.
    Note that some functions defined in the generic Backend class such as are
    cross-map pooling and normalization and adaDelta are not implemented for
    this backend.
    """
    default_dtype = None  # set this to np.float32 for less strict typing

    def __init__(self, rng_seed, stochastic_round=False, device_id=0):
        self.ng = NervanaGPU(stochastic_round=stochastic_round)
        logger.info("Initialized NervanaGPU with stochastic_round=%s",
                    stochastic_round)
        self.rng_seed = rng_seed
        self.rng_init()
        self.device_id = device_id if device_id is not None else 0

    def init_mempool(self, shape, dtype=default_dtype):
        """
        Allocates a memory pool for temporary storage
        """
        self.mem_pool = self.ng.empty(shape, dtype=dtype)

    def rng_init(self):
        """
        Initialize and seed the pseudo random number genrator. Random numbers
        are generated on the host using numpy, then transfered to device.
        """
        seed = None
        if 'rng_seed' in self.__dict__:
            seed = self.rng_seed
            logger.info("Seeding random number generator with: %s", str(seed))
        np.random.seed(seed)

    def flop_timing_init(self, decorate_fc, decorate_conv, decorate_ew):
        """
        Initialize FLOP timing.  Wraps the specified MOP calls via a decorator
        to record elapsed time and number of operations.

        Arguments:
           decorate_fc (list): string giving the function names of fully
                               connected layer forward/backward/update calls
                               to time.
           decorate_conv (list): string giving the function names of
                                 convolutional layer forward/backward/update
                                 calls to time.
           decorate_ew (list): string giving the function names of element-wise
                               calls to time.

        Notes:
            Must be called prior to first flop_timing_start call
        """
        self.start = drv.Event()
        self.end = drv.Event()
        self.flop_timer = FlopsDecorator(self)
        self.flop_timer.decorate(decorate_fc=decorate_fc,
                                 decorate_conv=decorate_conv,
                                 decorate_ew=decorate_ew)

    def flop_timinig_start(self):
        """
        Start a new FLOP timer.
        Returns:
            None: dummy value (not used)
        """
        return self.start.record()

    def flop_timing_finish(self, start_time):
        """
        Complete current FLOP timing.

        Arguments:
            start_time (unused): ignored.

        Returns:
            float: elapsed time in seconds since prior flop_timing_start call.
        """
        self.end.record()
        self.end.synchronize()
        return self.end.time_since(self.start)

    def uniform(self, low=0.0, high=1.0, shape=1, dtype=default_dtype, name=None,
                allocator=drv.mem_alloc):
        """
        generate numpy random number and convert to a GPUTensor.
        If called with dype=None it will probably explode
        """
        ary = np.random.uniform(low, high, shape)
        return GPUTensor(ary.shape, dtype, allocator=allocator, name=name,
                         rounding=self.ng.round_mode).set(ary)

    def normal(self, loc=0.0, scale=1.0, size=1, dtype=default_dtype, name=None,
               allocator=drv.mem_alloc):
        """
        Gaussian/Normal random number sample generation
        """
        ary = np.random.normal(loc, scale, size)
        return GPUTensor(ary.shape, dtype, allocator=allocator, name=name,
                         rounding=self.ng.round_mode).set(ary)

    def fprop_fc(self, out, inputs, weights, layer=None):
        """
        Forward propagate the inputs of a fully connected network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            layer (Layer): The layer object.
        """
        self.ng.dot(weights, inputs, out)

    def bprop_fc(self, out, weights, deltas, layer=None):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        self.ng.dot(weights.T, deltas, out)

    def update_fc(self, out, inputs, deltas, layer=None):
        """
        Compute the updated gradient for a fully connected network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            layer (Layer): The layer object.
        """
        self.ng.dot(deltas, inputs.T, out)

    def fprop_conv(self, out, inputs, weights, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, nifm, padding, stride, ngroups, fpropbuf,
                   local=False):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element
                                  in each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).  For this
                              backend we expect these values to be square.
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  convolved outputs for a single receptive
                                  field.  Not used for this backend.
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """

        '''
        N: Number of images in mini-batch
        C: Number of input feature maps
        K: Number of output feature maps

        D: Depth  of input image
        H: Height of input image
        W: Width  of input image

        T: Depth  of filter kernel
        R: Height of filter kernel
        S: Width  of filter kernel
        '''
        self.ng.fprop_conv(layer=fpropbuf, I=inputs, F=weights, O=out,
                           alpha=1.0, repeat=1)

    def bprop_conv(self, out, weights, deltas, ofmshape, ofmsize, ofmlocs,
                   ifmshape, links, padding, stride, nifm, ngroups, bpropbuf,
                   local=False):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            bpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
        """
        self.ng.bprop_conv(layer=bpropbuf, F=weights, E=deltas, grad_I=out,
                           alpha=1.0, repeat=1)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride, ngroups,
                    fwidth, updatebuf, local=False, layer=None):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            out (GPUTensor): Where to store the updated gradient value.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            weights (GPUTensor): The weight coefficient values for this layer.
            deltas (GPUTensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fwidth (int): Filter width.
            updatebuf (GPUTensor): Temporary storage buffer used to hold the
                                   updated gradient for a single receptive
                                   field
            local (bool, optional): Whether to do local filtering (True) or
                                    convolution (False, the default)
            layer (Layer): The layer object.
        """
        self.ng.update_conv(layer=updatebuf, I=inputs, E=deltas, grad_F=out,
                            alpha=1.0, repeat=1)

    def fprop_pool(self, out, inputs, op, ofmshape, ofmsize, ofmlocs, fshape,
                   ifmshape, links, nifm, padding, stride, fpropbuf):
        """
        Forward propagate the inputs of a Pooling network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (GPUTensor): Where to store the forward propagated results.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                                 each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            fpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  pooled outputs for a single receptive field.
        """
        op = op.lower()
        if op == "max":
            self.ng.fprop_pool(layer=fpropbuf, I=inputs, O=out, repeat=1)
        else:
            raise AttributeError("unexpected pooling op type: %s", op)

    def bprop_pool(self, out, fouts, inputs, deltas, op, ofmshape, ofmsize,
                   ofmlocs, fshape, fpsize, ifmshape, links, nifm, padding,
                   stride, bpropbuf):
        """
        Backward propagate the error through a pooling network layer.

        Arguments:
            out (GPUTensor): Where to store the backward propagated errors.
            fouts (GPUTensor): Forward propagated outputs from the previous
                               layer.
            inputs (GPUTensor): Will be either the dataset input values (first
                                layer), or the outputs from the previous layer.
            deltas (GPUTensor): The error values for this layer
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmsize (int): Total size of each output feature map.
            ofmlocs (GPUTensor): Indices giving the location of each element in
                              each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            fpsize (int): The size of each filter.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (GPUTensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            bpropbuf (GPUTensor): Temporary storage buffer used to hold the
                                  backpropagated error for a single receptive
                                  field
        """
        op = op.lower()
        if op == "max":
            self.ng.bprop_pool(layer=bpropbuf, I=inputs, E=deltas, grad_I=out,
                               repeat=1)
        else:
            raise AttributeError("unexpected pooling op type: %s", op)

    def logistic(self, x, out):
        """
        Logistic sigmoid nonlinearity, 1/(1+exp(-x))

        Arguments:
            x (GPUTensor): Input tensor
            out (GPUTensor): Output tensor

        """
        self.ng.sig(x, out=out)

        return out

    def rectlin(self, x, out):
        """
        Rectified Linear nonlinearity

        Arguments:
            x (GPUTensor): Input tensor
            out (GPUTensor): Output tensor

        """
        self.ng.maximum(x, 0., out=out)
        return out

    def sum(self, tsr, axes, out):
        """
        Sum

        Arguments:
            tsr  (GPUTensor): Input tensor
            axes (int): Axis along which the reduction is performed. If axes
                        is None,  the tensor is flattened and reduced over
                        both dimensions.
            out (GPUTensor): Output tensor
        """
        if axes is None:
            sze = tsr.shape[0]*tsr.shape[1]
            self.ng.sum(tsr.reshape(sze, 1), axis=0, out=out)
        else:
            self.ng.sum(tsr, axis=axes, out=out)
        return out

    def mean(self, tsr, axes, out):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            tsr (GPUTensor): Input tensor
            axes (int): Axis along which the reduction is performed. If axes
                        is None,  the tensor is flattened and reduced over
                        both dimensions.
            out (GPUTensor): Output tensor
        """
        if axes is None:
            sze = tsr.shape[0]*tsr.shape[1]
            self.ng.mean(tsr.reshape(sze, 1), axis=0, out=out)
        else:
            self.ng.mean(tsr, axis=axes, out=out)
        return out

    def min(self, tsr, axes, out):
        """
        Calculates the minimum of the elements along the specified
        axes.

        Arguments:
            tsr (GPUTensor): Input tensor
            axes (int): Axis along which the reduction is performed. If axes
                        is None,  the tensor is flattened and reduced over
                        both dimensions.
            out (GPUTensor): Output tensor
        """
        if axes is None:
            sze = tsr.shape[0]*tsr.shape[1]
            self.ng.min(tsr.reshape(sze, 1), axis=0, out=out)
        else:
            self.ng.min(tsr, axis=axes, out=out)
        return out

    def max(self, tsr, axes, out):
        """
        Calculates the maximum of the elements along the specified
        axes.

        Arguments:
            tsr (GPUTensor): Input tensor
            axes (int): Axis along which the reduction is performed. If axes
                        is None,  the tensor is flattened and reduced over
                        both dimensions.
            out (GPUTensor): Output tensor
        """
        if axes is None:
            sze = tsr.shape[0]*tsr.shape[1]
            self.ng.max(tsr.reshape(sze, 1), axis=0, out=out)
        else:
            self.ng.max(tsr, axis=axes, out=out)
        return out

    def variance(self, tsr, axes, out, mean=None, dtype=default_dtype):
        """
        Calculates the variance of the elements along the specified
        axes.

        Arguments:
            tsr (GPUTensor): the tensor on which to compute the variance
            axes (int, list, optional): the dimension(s) along which to
                                        variance.  If set to None, we will
                                        variance over all dimensions.
            out (GPUTensor): where the result will be stored.
            mean (GPUTensor): the tensor containing mean of tsr

        Returns:
            GPUTensor: reference to out
        """
        if mean is None:
            logger.error("GPUTensor requires mean to be specified.")
            raise ValueError("mean not specified")
        self.ng.mean(self.ng.square(tsr-mean),  axis=axes, out=out)
        return out

    def sqrt(self, x, out, dtype=default_dtype):
        """
        Calculates square root of the elements in a tensor

        Arguments:
            x (GPUTensor): Input tensor
            out (GPUTensor): Output tensor

        Returns:
            GPUTensor: reference to out
        """
        self.ng.sqrt(x, out=out)
        return out

    def zeros(self, shape, dtype=default_dtype):
        """
        Allocate a new GPUTensor and fill it with zeros.

        Arguments:
            shape (tupel): Shape of the desired GPUTensor
            dtype (dtype): Optional datatype

        Returns:
            GPUTensor: output
        """
        return self.ng.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=default_dtype):
        """
        Allocate a new GPUTensor and fill it with ones.

        Arguments:
            shape (tupel): Shape of the desired GPUTensor
            dtype (dtype): Optional datatype

        Returns:
            GPUTensor: output
        """
        return self.ng.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=default_dtype):
        """
        Allocate a new GPUTensor.

        Arguments:
            shape (tupel): Shape of the desired GPUTensor
            dtype (dtype): Optional datatype

        Returns:
            GPUTensor: output
        """
        return self.ng.empty(shape, dtype=dtype)

    def array(self, ary, dtype=default_dtype, name=None, allocator=drv.mem_alloc):
        """
        Allocate a new GPUTensor and fill it with supplied numpy array.

        Arguments:
            ary (ndarray): Numpy array with source data
            dtype (dtype): Optional datatype
            name (string): Name for the GPUTensor
            allocator (pycuda): Pycuda memory allocator

        Returns:
            GPUTensor: output
        """
        return GPUTensor(ary.shape, dtype, allocator=allocator, name=name,
                         rounding=self.ng.round_mode).set(ary)

    def add(self, left, right, out):
        """
        Elementwise addition

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.add(left, right, out=out)
        return out

    def subtract(self, left, right, out):
        """
        Elementwise subtraction

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.subtract(left, right, out=out)
        return out

    def multiply(self, left, right, out):
        """
        Elementwise multiplication

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.multiply(left, right, out=out)
        return out

    def divide(self, left, right, out):
        """
        Elementwise division

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.divide(left, right, out=out)
        return out

    def greater(self, left, right, out):
        """
        Elementwise greater than testing

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.greater(left, right, out=out)
        return out

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.equal(left, right, out=out)
        return out

    def not_equal(self, left, right, out):
        """
        Elementwise not equal testing

        Arguments:
            left (GPUTensor, numeric): left-hand side operand.
            right (GPUTensor, numeric): right-hand side operand.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.not_equal(left, right, out=out)
        return out

    def clip(self, a, a_min, a_max, out):
        """
        Elementwise clipping between a range of specified values

        Arguments:
            a (GPUTensor): input tensor.
            a_min (float): floor value.
            a_max (float): ceiling value.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.clip(a, a_min, a_max, out=out)
        return out

    def log(self, a, out):
        """
        Elementwise base-e logarithm

        Arguments:
            a (GPUTensor): input tensor.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        self.ng.log(a, out=out)
        return out

    def argmax(self, a, out, axis=0):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the elements of
        the first are returned.

        Arguments:
            tsr (GPUTensor): The GPUTensor on which to find the maximum indices
            axis (int): The dimension along which to find the maximum.  If set
                        to None, find the overall maximum index of a flattened
                        representation of tsr.
            out (GPUTensor): Where to store the result.  Should be of the
                             appropriate type and expected shape

        Returns:
            GPUTensor: reference to out
        """
        self.ng.argmax(a, out=out, axis=axis)
        return out

    def softmax(self, x, out):
        """
        Softmax nonlinearity. Computes exp(x-max(x)) / sum_i exp(x_i-max(x_i))

        Arguments:
            x (GPUTensor): input tensor.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """

        vecbuf = self.mem_pool
        assert vecbuf.shape == (1, x.shape[1])
        self.ng.max(x, axis=0, out=vecbuf)    # reduction over classes
        self.ng.exp(x - vecbuf, out=out)      # followed by ew
        self.ng.sum(out, axis=0, out=vecbuf)  # reduction over classes
        out[:] = out / vecbuf                 # followed by ew

        return out

    def softmax_gradient(self, y, err, out):
        """
        Gradient of the softmax nonlinearity.

        Arguments:
            y (GPUTensor): input tensor.
            err (GPUTensor): backpropagated error.
            out (GPUTensor): where the result will be stored.

        Returns:
            GPUTensor: reference to out
        """
        raise NotImplementedError("Softmax gradient should use shortcut")
        return out

    def make_binary_mask(self, tsr, keepthresh=0.5, dtype=default_dtype):
        """
        Create a binary mask for dropout layers.

        Arguments:
            tsr (GPUTensor): Output tensor
            keepthresh (float): fraction of ones
        """
        self.ng.dropout(keep=keepthresh, out=tsr)

    def gdm_compound(self, ps_item, us_item, vs_item, momentum_coef,
                     learning_rate, epoch):
        """
        Perform gradient descent update with momentum.

        Arguments:
            ps_item (GPUTensor): parameter tensor (e.g. a weight matrix)
            us_item (GPUTensor): update tensor, contains gradient wrt. weights
            vs_item (GPUTensor): velocity tensor.
            momentum_coef (float): momentum coefficient.
            learning_rate (float): learning rate.
            epoch (int): epoch (used in conjunction with diagnostics).

        Outputs are written to vs_item (updated velocity)
        and ps_item (updated weights)
        """
        self.ng.subtract(self.ng.multiply(vs_item, momentum_coef),
                         self.ng.multiply(us_item, learning_rate),
                         out=vs_item)
        self.ng.add(ps_item, vs_item, out=ps_item)

    def gdmwd_compound(self, ps_item, us_item, vs_item, momentum_coef,
                       learning_rate, wd, epoch):
        """
        Perform gradient descent update with momentum and weight decay.

        Arguments:
            ps_item (GPUTensor): parameter tensor (e.g. a weight matrix)
            us_item (GPUTensor): update tensor, contains gradient wrt. weights
            vs_item (GPUTensor): velocity tensor.
            momentum_coef (float): momentum coefficient.
            learning_rate (float): learning rate.
            wd (float): weight decay parameter.
            epoch (int): epoch (used in conjunction with diagnostics).

        Outputs:
            ps_item, the updated weights.
            vs_item, the updated velocity.
            us_item, used as a temp buffer.
        """
        self.ng.subtract(self.ng.multiply(vs_item, momentum_coef),
                         self.ng.multiply(us_item, learning_rate),
                         out=vs_item)

        # weight decay
        self.ng.multiply(self.ng.multiply(ps_item, wd),
                         learning_rate, out=us_item)
        self.ng.subtract(vs_item, us_item, out=vs_item)

        self.ng.add(ps_item, vs_item, out=ps_item)
