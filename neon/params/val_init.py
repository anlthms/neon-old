# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Classes used to construct and initialize the values of parameter Tensors.
"""

import logging
import math
import numpy as np

from neon.util.param import opt_param
from neon.util.persist import YAMLable

logger = logging.getLogger(__name__)


class ValGen(YAMLable):
    """
    Base class used to generate new Tensors initialized in a specific manner.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        opt_param(self, ['backend'])

    def __str__(self):
        return ("{cl_nm} utilizing {be_nm} backend".format(
                cl_nm=self.__class__.__name__,
                be_nm=self.backend.__class__.__name__))

    def initialize(self, backend):
        """
        Perform any additional setup (like attaching the backend), required
        prior to generating values.
        """
        self.backend = backend

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        raise NotImplementedError('This class should not be instantiated.')


class UniformValGen(ValGen):
    """
    Uniform random value initialization scheme.  All values are chosen to lie
    in the range [low, high) with equal probability.

    Arguments:
        low (float, optional): Minimal sample value.  Defaults to 0.0
        high (float, optional): Maximal sample value (open-ended range).
                                Defaults to 1.0
    """
    def __init__(self, **kwargs):
        super(UniformValGen, self).__init__(**kwargs)
        opt_param(self, ['low'], 0.0)
        opt_param(self, ['high'], 1.0)

    def __str__(self):
        return (super(UniformValGen, self).__str__() +
                "\n\tlow: {self.low}, high: {self.high}".format(self=self))

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        logger.info("Generating {cl_nm} values of shape {shape}".format(
                    cl_nm=self.__class__.__name__, shape=shape))
        return self.backend.uniform(self.low, self.high, shape, dtype)


class AutoUniformValGen(UniformValGen):
    """
    Uniform random value initialization scheme with low and high automatically
    inferred from the dimensions of the shape being passed.  Typically this
    will be uniform between +/- 1/sqrt(fan-in).  If relu is passed in and true
    we further scale these values by sqrt(2).

    Arguments:
        relu (bool, optional): Producing values for ReLU activated weights,
                               further scale values by sqrt(2).  Defaults to
                               False
    """
    def __init__(self, **kwargs):
        super(AutoUniformValGen, self).__init__(**kwargs)
        opt_param(self, ['relu'], False)
        self.low = float('nan')
        self.high = float('nan')

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        logger.info("Generating {cl_nm} values of shape {shape}".format(
                    cl_nm=self.__class__.__name__, shape=shape))
        self.low = - 1.0 / math.sqrt(shape[-1])
        if self.relu:
            self.low *= math.sqrt(2)
        self.high = - self.low
        return super(AutoUniformValGen, self).generate(shape, dtype)


class GaussianValGen(ValGen):
    """
    Gaussian (aka Normal) distributed random value initialization scheme.

    Arguments:
        loc (float, optional): Central value location.  Defaults to 0.0
        scale (float, optional): Standard deviation for samples.  Defaults to
                                 1.0
    """
    def __init__(self, **kwargs):
        super(GaussianValGen, self).__init__(**kwargs)
        opt_param(self, ['loc'], 0.0)
        opt_param(self, ['scale'], 1.0)

    def __str__(self):
        return (super(GaussianValGen, self).__str__() +
                "\n\tloc: {self.loc}, scale: {self.scale}".format(self=self))

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        logger.info("Generating {cl_nm} values of shape {shape}".format(
                    cl_nm=self.__class__.__name__, shape=shape))
        return self.backend.normal(self.loc, self.scale, shape, dtype)


# alias NormalValGen as GaussianValGen
NormalValGen = GaussianValGen


class SparseEigenValGen(ValGen):
    """
    Sparse Eigenvalue based initialization scheme suitable for recurrent neural
    networks, as described in Sutskever2013.

    Arguments:
        sparseness (int, optional): controls number of non-zero entries.
                                    Should set to a value between 1 (extremely
                                    sparse) and fan-in count (dense).  Defaults
                                    to 15.
        eigenvalue (float, optional): For square matrices, we scale by this
                                      value after dividing by the maximum
                                      eigenvalue.  Defaults to 1.2
    """
    def __init__(self, **kwargs):
        super(SparseEigenValGen, self).__init__(**kwargs)
        opt_param(self, ['sparseness'], 15)
        opt_param(self, ['eigenvalue'], 1.2)

    def __str__(self):
        return (super(SparseEigenValGen, self).__str__() +
                "\n\tsparseness: {self.sparseness}, eigenvalue: "
                "{self.eigenvalue}".format(self=self))

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        logger.info("Generating {cl_nm} values of shape {shape}".format(
                    cl_nm=self.__class__.__name__, shape=shape))
        if len(shape) < 2:
            raise ValueError("Can only generate Tensors with at least 2"
                             " dimensions, you gave: {}".format(len(shape)))
        elements = shape[-2] * shape[-1]
        nonzeros = shape[-2] * self.sparseness
        weights = np.zeros(shape).flatten()
        nonzeroindex = np.random.permutation(elements)[0:nonzeros]
        weights[nonzeroindex] = 0.3 * np.random.randn(nonzeros)
        weights = weights.reshape(shape)
        if shape[-2] == shape[-1]:
            temp = np.linalg.eig(weights)
            max_eig = np.max(np.absolute(temp[0]))
            logger.info("dividing by max eigenvalue %2.2f", max_eig)
            weights = self.eigenvalue * weights / max_eig
        else:
            logger.info("maxtrix is non-square, no eigenvalue scaling.")
        return self.backend.array(weights)


class NodeNormalizedValGen(ValGen):
    """
    Normalized initialization as described in Glorot2010.  Values are uniform
    distributed to lie in the range:
    scale * [ - sqrt(6) / sqrt(sum(shape)), sqrt(6) / sqrt(sum(shape)))

    Arguments:
        scale (float, optional): Additional scalar to multiply by to extend the
                                 range.  Defaults to 1.0.
    """
    def __init__(self, **kwargs):
        super(NodeNormalizedValGen, self).__init__(**kwargs)
        opt_param(self, ['scale'], 1.0)

    def __str__(self):
        return (super(NodeNormalizedValGen, self).__str__() +
                "\n\tscale: {self.scale}".format(self=self))

    def generate(self, shape, dtype=None):
        """
        Construct and initialize a new Tensor object of the specified shape.

        Arguments:
            shape (list of ints): The size of each dimension of the Tensor.
            dtype (dtype, optional): Element data type.  If not specifed we use
                                     the default dtype associated with that
                                     backend.

        Returns:
            neon.backneds.Tensor: newly initialized data structure.
        """
        logger.info("Generating {cl_nm} values of shape {shape}".format(
                    cl_nm=self.__class__.__name__, shape=shape))
        node_norm = self.scale * math.sqrt(6.0 / sum(shape))
        return self.backend.uniform(-node_norm, node_norm, shape, dtype)