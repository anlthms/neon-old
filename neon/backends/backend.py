# pylint: disable = R0904, R0913, C0103
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Houses low-level code for performing underlying data manipulation operations.
"""

from neon.util.persist import YAMLable


class Backend(YAMLable):
    """
    backend interface used to manipulate Tensor data.  This abstract
    base class defines what operations each concrete backend must support.
    Inherits configuration file handling via `yaml.YAMLObject
    <http://pyyaml.org/wiki/PyYAMLDocumentation#YAMLObject>`_

    Notes:
        See the list of `implemented backends </backends.html>`_
    """

    def empty(self, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, without
        initializing element values.  This is slightly faster than
        :py:func:`~neon.backends.backend.Backend.array`,
        :py:func:`~neon.backends.backend.Backend.ones`,
        :py:func:`~neon.backends.backend.Backend.zero`, but the values will be
        random.

        Arguments:
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.array`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.ones`
        """
        raise NotImplementedError()

    def array(self, obj, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        elements based on obj values.

        Arguments:
            obj (array_like): input array object to construct from.  Can be
                              built-in python scalar or list (of lists), or a
                              numpy.ndarray
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.ones`
        """
        raise NotImplementedError()

    def zeros(self, shape, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 0.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.ones`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    def ones(self, shape, dtype=None):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        each element with a value of 1.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
        Returns:
            Tensor: array object

        Raises:
            NotImplmentedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.empty`,
            :py:func:`~neon.backends.backend.Backend.zeros`,
            :py:func:`~neon.backends.backend.Backend.array`
        """
        raise NotImplementedError()

    def copy(self, tsr):
        """
        Construct and return a deep copy of the Tensor passed.

        Arguments:
            tsr (Tensor): the object to copy

        Returns:
            Tensor: new array object with the same values as tsr.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def uniform(self, low=0.0, high=1.0, size=1):
        """
        Uniform random number generation of samples in range [low, high).

        Arguments:
            low (numeric, optional): Minimal sample value.  Defaults to 0.0
            high (numeric, optional): Maximal sample value (open-ended range).
                                      Defaults to 1.0.
            size (int, list, optional): The shape of the samples to return.
                                        Defaults to 1

        Returns:
            Tensor: of shape size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.normal`,
        """
        raise NotImplementedError("Can't create direct instances of Backend")

    def normal(self, loc=0.0, scale=1.0, size=1):
        """
        Gaussian/Normal random number generation of samples centered around
        mean loc, and with standard deviation scale.

        Arguments:
            loc (numeric, optional): Central value for Gaussian.  Defaults to
                                     0.0
            scale (numeric, optional): Standard deviation for samples.
                                       Defaults to 1.0
            size (int, list, optional): The shape of the samples to return.
                                        Defaults to 1

        Returns:
            Tensor: of shape size filled with these random numbers.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Backend.uniform`,
        """
        raise NotImplementedError("Can't create direct instances of Backend")

    def add(self, left, right, out):
        """
        Perform element-wise addition on the operands, storing the resultant
        values in the out Tensor.  Each operand and out must have identical
        shape or be broadcastable as such.

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def subtract(self, left, right, out):
        """
        Perform element-wise subtraction on the operands, storing the resultant
        values in the out Tensor.  Each operand and out must have identical
        shape or be broadcastable as such.

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def multiply(self, left, right, out):
        """
        Perform element-wise multiplication on the operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def divide(self, left, right, out):
        """
        Perform element-wise division on the operands, storing the
        resultant values in the out Tensor.  Each operand and out must have
        identical shape or be broadcastable as such.

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def log(self, tsr, out):
        """
        Perform element-wise natural logarithm transformation on Tensor tsr,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def exp(self, tsr, out):
        """
        Perform element-wise exponential transformation on Tensor tsr,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def power(self, tsr, power, out):
        """
        Perform element-wise raise of tsr values to specified power,
        storing the result in Tensor out.  Both Tensor's should have identical
        shape.

        Arguments:
            tsr (Tensor): input to be transformed.
            power (Tensor, numeric): exponentiated value to be applied to
                                     element.  Examples include 2 (square),
                                     0.5 (sqaure root).
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def dot(self, left, right, out, alpha=1, beta=0):
        """
        Perform sum product between the last axis of left and the second last
        axis of right, storing the result in out.  Note that this dot product
        is equivalent to the inner product if operands are vectors, and matrix
        multiplication if both operands are matrices.  We support BLAS Level 3
        general matrix multiplication (GEMM) functionality by including
        additional scalars alpha and beta.  The general form of the multiply
        is: out <- alpha * left * right + beta * out, but will be
        short-circuited to: out <- alpha * left * right if beta has value 0
        (the default).  All Tensor's should have commensurate shape or be
        broadcastable as such.

        Arguments:
            left (Tensor): left-hand side operand.
            right (Tensor): right-hand side operand.
            out (Tensor): where the result will be stored.  Note that this
                          object should differ from left and right.
            alpha (numeric, optional): scalar to multiply the resultant sum
                                       product by.  Defaults to 1.
            beta (numeric, optional): scalar to pre-multiply out values by
                                      prior to adding to sum product.  Defaults
                                      to 0, which implies no such addition of
                                      prior out values.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def equal(self, left, right, out):
        """
        Performs element-wise equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def not_equal(self, left, right, out):
        """
        Performs element-wise non-equality testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def greater(self, left, right, out):
        """
        Performs element-wise greater than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def greater_equal(self, left, right, out):
        """
        Performs element-wise greater than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def less(self, left, right, out):
        """
        Performs element-wise less than testing on each element of left and
        right, storing the result in out.  Each operand is assumed to be the
        same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def less_equal(self, left, right, out):
        """
        Performs element-wise less than or equal testing on each element of
        left and right, storing the result in out.  Each operand is assumed to
        be the same shape (or broadcastable as such).

        Arguments:
            left (Tensor, numeric): left-hand side operand.
            right (Tensor, numeric): right-hand side operand.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def sum(self, tsr, axes, out):
        """
        Calculates the summation of the elements along the specified axes.

        Arguments:
            tsr (Tensor): the Tensor on which to perform the sum
            axes (int, list, optional): the dimension(s) along which to sum.
                                        If set to None, we will sum over all
                                        dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def mean(self, tsr, axes, out):
        """
        Calculates the arithmetic mean of the elements along the specified
        axes.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the average
            axes (int, list, optional): the dimension(s) along which to
                                        average.  If set to None, we will
                                        average over all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def min(self, tsr, axes, out):
        """
        Calculates the minimal element value along the specified axes.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the minimum
            axes (int, list, optional): the dimension(s) along which to find
                                        the minimum.  If set to None, we will
                                        compute the overall minimal value
                                        across all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def max(self, tsr, axes, out):
        """
        Calculates the maximal element value along the specified axes.

        Arguments:
            tsr (Tensor): the Tensor on which to compute the maximum
            axes (int, list, optional): the dimension(s) along which to find
                                        the maximum.  If set to None, we will
                                        compute the overall maximal value
                                        across all dimensions.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def argmin(self, tsr, axis, out):
        """
        Calculates the indices of the minimal element value along the specified
        axis.  If multiple elements contain the minimum, only the indices of
        the first are returned.

        Arguments:
            tsr (Tensor): the Tensor on which to find the minimum indices
            axis (int): the dimension along which to find the minimum.  If set
                        to None, we will return the index relative to the 1-D
                        flattened version of the tensor.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def argmax(self, tsr, axis, out):
        """
        Calculates the indices of the maximal element value along the specified
        axis.  If multiple elements contain the maximum, only the indices of
        the first are returned.

        Arguments:
            tsr (Tensor): the Tensor on which to find the maximum index
            axis (int): the dimension along which to find the maximum.  If set
                        to None, we will return the index relative to the 1-D
                        flattened version of the tensor.
            out (Tensor): where the result will be stored.

        Returns:
            Tensor: reference to out

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def norm(self, tsr, order=None, axis=None, out=None):
        """
        Calculates and returns the vector p-norms of the Tensor along the
        specified axis.  The p-norm is defined on vector A as
        :math:`||A||_p = \sum_i(|A_i|^p)^{1/p}`.

        Arguments:
            tsr (Tensor): the Tensor on which to find the norms
            order (int): The order or p upon which the norm is calculated.
                         Valid values include:
                         None, inf, -inf, 0, 1, -1, 2, -2, ...
            axis (int): The axis along which to compute vector norms.
            out (Tensor, optional): where to write the results to.  Must be
                                    of the expected result shape.  If not
                                    specified, a new buffer is created and
                                    returned.

        Returns:
            Tensor: p-norms of tsr along the specified axis.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def rng_init(self):
        """
        Perform random number initialization.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError("Can't create direct instances of Backend")

    def err_init(self):
        """
        Perform error handling initialization.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fprop_fc(self, out, inputs, weights):
        """
        Forward propagate the inputs of a fully connected network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (Tensor): The weight coefficient values for this layer.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def bprop_fc(self, out, weights, deltas):
        """
        Backward propagate the error through a fully connected network layer.

        Arguments:
            out (Tensor): Where to store the backward propagated errors.
            weights (Tensor): The weight coefficient values for this layer.
            deltas (Tensor): The error values for this layer

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def update_fc(self, out, inputs, deltas):
        """
        Compute the updated gradient for a fully connected network layer.

        Arguments:
            out (Tensor): Where to store the updated gradient value.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            deltas (Tensor): The error values for this layer

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fprop_conv(self, out, inputs, weights, ofmshape, ofmlocs, ifmshape,
                   links, nifm, padding, stride, ngroups, fpropbuf):
        """
        Forward propagate the inputs of a convolutional network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (Tensor): The weight coefficient values for this layer.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmlocs (Tensor): Indices giving the location of each element in
                              each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (Tensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fpropbuf (Tensor): Temporary storage buffer used to hold the
                               convolved outputs for a single receptive field.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def bprop_conv(self, out, weights, deltas, ofmshape, ofmlocs, ifmshape,
                   links, padding, stride, nifm, ngroups, bpropbuf):
        """
        Backward propagate the error through a convolutional network layer.

        Arguments:
            out (Tensor): Where to store the backward propagated errors.
            weights (Tensor): The weight coefficient values for this layer.
            deltas (Tensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmlocs (Tensor): Indices giving the location of each element in
                              each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (Tensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            bpropbuf (Tensor): Temporary storage buffer used to hold the
                               backpropagated error for a single receptive
                               field

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmlocs,
                    ifmshape, links, nifm, padding, stride, ngroups, fwidth,
                    updatebuf):
        """
        Compute the updated gradient for a convolutional network layer.

        Arguments:
            out (Tensor): Where to store the updated gradient value.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (Tensor): The weight coefficient values for this layer.
            deltas (Tensor): The error values for this layer
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmlocs (Tensor): Indices giving the location of each element in
                              each output feature map stored in out.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (Tensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           convolution operation.
            stride (int): Number of neurons to shift the filter at each step.
            ngroups (int): Number of groups.
            fwidth (int): Filter width.
            updatebuf (Tensor): Temporary storage buffer used to hold the
                                updated gradient for a single receptive
                                field

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fprop_pool(self, out, inputs, op, ofmshape, ofmlocs, fshape, ifmshape,
                   links, nifm, padding, stride, fpropbuf):
        """
        Forward propagate the inputs of a Pooling network layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              number of height and width neurons).
            ofmlocs (Tensor): Indices giving the location of each element in
                              each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            links (Tensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            fpropbuf (Tensor): Temporary storage buffer used to hold the
                               pooled outputs for a single receptive field.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def bprop_pool(self, out, fouts, inputs, deltas, op, ofmshape, ofmlocs,
                   fshape, ifmshape, links, nifm, padding, stride, bpropbuf):
        """
        Backward propagate the error through a pooling network layer.

        Arguments:
            out (Tensor): Where to store the backward propagated errors.
            fouts (Tensor): Forward propagated outputs from the previous layer.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            deltas (Tensor): The error values for this layer
            op (string): The type of pooling operation to apply.  We support
                         "max", "avg", "l2" currently.
            ofmshape (tuple): Dimensions of each output feature map (typically
                              height and width).
            ofmlocs (Tensor): Indices giving the location of each element in
                              each output feature map stored in out.
            fshape (tuple): Dimensions of each filter (typically height and
                            width).
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            links (Tensor): Input receptive field indices.
            nifm (int): Total number of input feature maps.
            padding (int): Number of additional elements to include along each
                           dimension of each local receptive field during the
                           pooling operation.
            stride (int): Number of neurons to shift the filter at each step.
            bpropbuf (Tensor): Temporary storage buffer used to hold the
                               backpropagated error for a single receptive
                               field

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fprop_cmrnorm(self, out, inputs, ifmshape, nifm, ksize, alpha, beta):
        """
        Forward propagate the inputs of a CrossMap response normalization layer
        to produce output pre-activations (ready for transformation by an
        activation function).  The normalization is computed across feature
        maps at each pixel point.  The output will be same size as input.

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            fpropbuf (Tensor): Temporary storage buffer used to hold the
                               normalized outputs for a single receptive field.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def bprop_cmrnorm(self, out, fouts, inputs, deltas, ifmshape, nifm, ksize,
                      alpha, beta, bpropbuf):
        """
        Backward propagate the error through a CrossMap response normalization
        layer.

        Arguments:
            out (Tensor): Where to store the backward propagated errors.
            fouts (Tensor): The forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            deltas (Tensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).
            nifm (int): Total number of input feature maps.
            ksize (int): Kernel size. This defines the channel indices to sum
                         over.
            alpha (int): scalar multiplier to multiply the normalization
                         denominator by.
            beta (int): scalar power to raise the normalization denominator by
            bpropbuf (Tensor): Temporary storage buffer used to hold the
                               normalized outputs for a single receptive field.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def fprop_cmpool(self, out, inputs, weights, ifmshape):
        """
        Forward propagate the inputs of a CrossMap Pooling layer to
        produce output pre-activations (ready for transformation by an
        activation function).

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            weights (Tensor): The weight coefficient values for this layer.
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def bprop_cmpool(self, out, weights, deltas, ifmshape):
        """
        Backward propagate the error through a CrossMap pooling layer.

        Arguments:
            out (Tensor): Where to store the forward propagated results.
            weights (Tensor): The weight coefficient values for this layer.
            deltas (Tensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              number of height and width neurons).

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def update_cmpool(self, out, inputs, deltas, ifmshape, updatebuf):
        """
        Compute the updated gradient for a CrossMap pooling layer.

        Arguments:
            out (Tensor): Where to store the updated gradient value.
            inputs (Tensor): Will be either the dataset input values (first
                             layer), or the outputs from the previous layer.
            deltas (Tensor): The error values for this layer
            ifmshape (tuple): Dimensions of each input feature map (typically
                              height and width).
            updatebuf (Tensor): Temporary storage buffer used to hold the
                                updated gradient for a single receptive
                                field

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()


class Tensor(object):
    """
    Represents an arbitrary n-dimensional array data structure.

    Arguments:
        object (numpy.ndarray): An n-dimensional array containing the actual
                                data values.  Python scalars, and lists (of
                                lists) are also supported in addition to
                                :py:class:`numpy.ndarray` objects
        dtype (numpy.dtype, optional): The underlying type of each element

    Attributes:
        shape (list): array specifying the length of each dimension
        dtype (numpy.dtype): the underlying type given to each element.
        raw (object): the underlying backend specific data structure.  Could be
                      numpy.ndarray, cudamat.CUDAMatrix, etc. depending on the
                      backend.

    Raises:
        NotImplmentedError: Can't be instantiated directly.
    """
    shape = None
    dtype = None
    raw = None

    def __init__(self, object, dtype=None):
        raise NotImplementedError()

    def __getitem__(self, key):
        """
        Extract a subset view of the items via slice style indexing
        along each dimension. e.g. A[5:10, :].  Each slice consists of
        start_idx:stop_idx:step_size triplets.  If step_size isn't specified it
        defaults to 1.  If start_idx isn't specified it defaults to 0.  If
        stop_idx isn't specified it defaults to the total number of elements
        along that dimension.  As such a slice value of ':' allows one to
        select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.

        Returns:
            Tensor: view of self corresponding to the subset items.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Tensor.take`,
        """
        raise NotImplementedError()

    def __setitem__(self, key, value):
        """
        Assign the specified value to a subset of elements found via slice
        style indexing along each dimension. e.g. A[5:10, :] = 4.5.
        Each slice consists of start_idx:stop_idx:step_size triplets.  If
        step_size isn't specified it defaults to 1.  If start_idx isn't
        specified it defaults to 0.  If stop_idx isn't specified it defaults
        to the total number of elements along that dimension.  As such a slice
        value of ':' allows one to select all elements along that dimension.

        Arguments:
            key (int, slice, tuple): indices of each dimension's slice.
            value (numeric array, Tensor): values to be assigned to the
                                          extracted element subset.  If an
                                          array it should be the same shape
                                          as what key indexes (or be
                                          broadcastable as such).

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def asnumpyarray(self):
        """
        Convert the tensor to an in host memory `numpy.ndarray`.  A copy of the
        data may be made depending on where the Tensor normally resides.

        Returns:
            numpy.ndarray view or copy of the Tensor data.
        """
        raise NotImplementedError()

    def reshape(self, shape):
        """
        Adjusts the dimensions of the data to the specified shape.  The number
        of elements represented by the new shape must be the same as before.

        Arguments:
            shape (int, list): new length of each dimension

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array relative to the specified axis.

        Arguments:
            repeats (int, list): The number of repetitions of each element.  It
                                 will be broadcast to fit the shape of the
                                 given axis
            axis (int, optional): The axis along which to repeat values.  If
                                  set to None, we flatten input to 1-D.

        Returns:
            Tensor: new variant with the same dimensions as self except along
                    the specified axis, where it will contain the repeated
                    elements.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def transpose(self):
        """
        Returns a view of the data in this Tensor whereby row and column
        elements are swapped.

        Return:
            Tensor: transposed data view.

        Raises:
            NotImplementedError: Can't be instantiated directly.
        """
        raise NotImplementedError()

    def take(self, indices, axis=None, out=None):
        """
        Extract a subset of elements along a given axis.

        Arguments:
            indices (list): 0 based element offsets to extract.
            axis (int, optional): The axis over which to select values.  If
                                  None, we index over a 1-D flattened version
                                  of the Tensor
            out (Tensor, optional): Pre-allocated Tensor of the correct size
                                    in which we will write the results.

        Returns:
            Tensor: subset of original elements

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Tensor.__getitem__`,
        """
        raise NotImplementedError()

    def fill(self, value):
        """
        Assign specified value to each element of this Tensor.

        Arguments:
            value (numeric): The value to be assigned to each element.

        Return:
            Tensor: updated view of the data.

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.backend.Tensor.__setitem__`,
        """
        raise NotImplementedError()

    def log(self):
        """
        Computes the elementwise natural logarithmic transform on this tensor.

        Returns:
            Tensor: log transformed values

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()

    def exp(self):
        """
        Exponentiates each element of this tensor.

        Returns:
            Tensor: e raised to the power of each value

        Raises:
            NotImplmentedError: Must override in a child Tensor class
        """
        raise NotImplementedError()
