.. currentmodule:: neon

.. |Tensor| replace:: :py:class:`~neon.backends.backend.Tensor`
.. |Backend| replace:: :py:class:`~neon.backends.backend.Backend`

************************
ML Operational Layer API 
************************

In order to interact with our T101 compiler and driver, we expose the following
API which we refer to as our ML operational layer. It currently consists of the
functions defined in the following two classes, which we detail further on the
rest of this page:

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

|Tensor| Manipulation
---------------------

.. autosummary::

   neon.backends.backend.Tensor.reshape
   neon.backends.backend.Tensor.transpose
   neon.backends.backend.Tensor.take

|Tensor| Attributes
-------------------

.. autosummary::

   neon.backends.backend.Tensor.shape
   neon.backends.backend.Tensor.dtype

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

Matrix Algebra Operations
-------------------------
.. autosummary::

   neon.backends.backend.Backend.dot

Logical Operation Support
=========================

TODO: format and add gt, le, ne, etc.

Summarization Operation Support
===============================

TODO: format and add sum, mean, min, max, argmax, etc.
