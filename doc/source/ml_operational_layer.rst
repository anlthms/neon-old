.. currentmodule:: neon

.. |Tensor| replace:: :py:class:`~neon.backends.backend.Tensor`
.. |Backend| replace:: :py:class:`~neon.backends.backend.Backend`

******************************
ML OPerational Layer (MOP) API 
******************************

In order to interact with our T101 compiler and driver, we expose the following
API which we refer to as our ML operational layer (aka MOP layer). It currently
consists of the functions defined in the following two interface classes, which
we detail further on the rest of this page:

.. autosummary::
   :toctree: generated/

   neon.backends.backend.Tensor
   neon.backends.backend.Backend

Basic Data Structure
====================

The |Tensor| class is used to represent an arbitrary dimensional array in which
each element is stored using a consistent underlying type.

We have the ability to instantiate and copy instances of this data
structure, as well as initialize its elements, reshape its dimensions, and 
access metadata.

|Tensor| Creation
-----------------

.. autosummary::

   neon.backends.backend.Backend.array
   neon.backends.backend.Backend.zeros
   neon.backends.backend.Backend.ones
   neon.backends.backend.Backend.copy
   neon.backends.backend.Backend.uniform
   neon.backends.backend.Backend.normal

|Tensor| Manipulation
---------------------

.. autosummary::

   neon.backends.backend.Tensor.take
   neon.backends.backend.Tensor.__getitem__
   neon.backends.backend.Tensor.__setitem__
   neon.backends.backend.Tensor.transpose
   neon.backends.backend.Tensor.reshape
   neon.backends.backend.Tensor.repeat

|Tensor| Attributes
-------------------

.. autosummary::

   neon.backends.backend.Tensor.shape
   neon.backends.backend.Tensor.dtype
   neon.backends.backend.Tensor.raw

Arithmetic Operation Support
============================

Unary and binary arithmetic operations can be performed on |Tensor| objects via
appropriate |Backend| calls.  In all cases it is up to the user to pre-allocate
correctly sized output to house the result.

Element-wise Binary Operations
------------------------------
.. autosummary::

   neon.backends.backend.Backend.add
   neon.backends.backend.Backend.subtract
   neon.backends.backend.Backend.multiply
   neon.backends.backend.Backend.divide

Element-wise Unary Transcendental Functions
-------------------------------------------
.. autosummary::

   neon.backends.backend.Backend.log
   neon.backends.backend.Backend.exp
   neon.backends.backend.Backend.power

Matrix Algebra Operations
-------------------------
.. autosummary::

   neon.backends.backend.Backend.dot

Logical Operation Support
=========================

.. autosummary::

   neon.backends.backend.Backend.equal
   neon.backends.backend.Backend.not_equal
   neon.backends.backend.Backend.greater
   neon.backends.backend.Backend.greater_equal
   neon.backends.backend.Backend.less
   neon.backends.backend.Backend.less_equal

Summarization Operation Support
===============================
.. autosummary::

   neon.backends.backend.Backend.sum
   neon.backends.backend.Backend.mean
   neon.backends.backend.Backend.min
   neon.backends.backend.Backend.max
   neon.backends.backend.Backend.argmin
   neon.backends.backend.Backend.argmax
   neon.backends.backend.Backend.norm

Initialization and Setup
========================
.. autosummary::

   neon.backends.backend.Backend.rng_init
   neon.backends.backend.Backend.err_init

Higher Level Operation Support
==============================

TODO: determine what to include here
