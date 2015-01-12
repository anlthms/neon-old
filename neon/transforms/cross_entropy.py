# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Cross entropy transform functions and classes.
"""

from neon.transforms.cost import Cost
from neon.transforms.logistic import Logistic
from neon.transforms.softmax import Softmax
from neon.util.param import opt_param


def cross_entropy(backend, outputs, targets, temp):
    """
    Evaluates cross entropy on pairwise elements from outputs and targets.

    Given that this is undefined for predicted outputs equal to exactly 0 or
    1.0, we first clip these outputs to epsilon (backend machine precision) and
    1.0 - epsilon respectively.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """
    # Compute (t-1)*log(1-y).
    backend.add(targets, -1.0, out=temp[0])
    backend.subtract(1.0, outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(temp[0], temp[1], out=temp[0])

    # Compute t*log(y).
    backend.clip(outputs, backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.log(temp[1], out=temp[1])
    backend.multiply(targets, temp[1], out=temp[1])

    # Compute t*log(y) - (t-1)*log(1-y)
    backend.subtract(temp[0], temp[1], out=temp[0])
    return backend.sum(temp[0])


def cross_entropy_multi(backend, outputs, targets, temp):
    """
    Evaluates cross entropy on elements from outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and type as outputs.
    """

    # Compute (t*log(y)).
    backend.clip(outputs, backend.epsilon, 1, out=temp[1]) # THIS IS NEW!
    backend.log(temp[1], out=temp[1]) # WAS log(outputs)
    backend.multiply(targets, temp[1], out=temp[1])
    backend.multiply(temp[1], -1.0, out=temp[0])
    print "FUCK", temp[0].shape # never arrive here. That's good, not using multi.
    return backend.sum(temp[0]) # WAS MEAN


def cross_entropy_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Note that this is undefined for predicted outputs equal to exactly 0 or
    1.0, so we clip these to epsilon (backend machine precision) and 1.0 -
    epsilon respectively.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.subtract(1.0, outputs, out=temp[1])
    backend.multiply(temp[1], outputs, out=temp[1])
    backend.clip(temp[1], backend.epsilon, 1 - backend.epsilon, out=temp[1])
    backend.divide(temp[0], temp[1], out=temp[0])
    return temp[0]


def cross_entropy_multi_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    Applies derivative of the cross entropy to the pairwise elements from
    outputs and targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.

    Returns:
        array_like: Calculated cross entropy values for each element.  Will
                    have the same shape and backend as outputs.
    """
    backend.divide(targets, outputs, out=temp[0])
    backend.multiply(temp[0], -scale, out=temp[0])
    return temp[0]


def shortcut_derivative(backend, outputs, targets, temp, scale=1.0):
    """
    For use when combining cost with matched activation
    i.e. cross_entropy_binary with logistic or
         cross_entropy_multi  with softmax
    Derivative has simpler form and removes numerical errors
    """
    backend.subtract(outputs, targets, out=temp[0])
    backend.multiply(temp[0], scale, out=temp[0])
    return temp[0]

class OldCrossEntropy(Cost):

    """
    Embodiment of a cross entropy cost function.
    """
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        # The default cross entropy to use is where each output node
        # can take on value with binary prob
        if not hasattr(self, 'use_binary'):
            self.use_binary = True

        if not hasattr(self, 'shortcut_deriv'):
            self.shortcut_deriv = False
            if (isinstance(self.olayer.activation, Logistic)
                    and self.use_binary):
                self.shortcut_deriv = True

            if (isinstance(self.olayer.activation, Softmax)
                    and not self.use_binary):
                self.shortcut_deriv = True

        # Set the appropriate functions
        self.ce_function = cross_entropy
        self.cd_function = cross_entropy_derivative

        if not self.use_binary:
            self.ce_function = cross_entropy_multi
            self.cd_function = cross_entropy_multi_derivative

        # if self.shortcut_deriv:
            # self.cd_function = shortcut_derivative

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf, tempbuf.copy()]
        self.outputbuf = databuf

    def apply_function(self, targets):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return self.ce_function(self.backend, self.outputbuf,
                                targets, self.temp)

    def apply_derivative(self, targets):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return self.cd_function(self.backend, self.outputbuf,
                                targets, self.temp)


class CrossEntropy(Cost):

    """
    Embodiment of a cross entropy cost function.
    """
    def __init__(self, **kwargs):
        print "RUNNING NEW INIT"
        super(CrossEntropy, self).__init__(**kwargs)

    # new self.cost.initialize(kwargs) call in rnn init.
    # running this function breaks shit. previous initialize() call went to super.
    def initialize(self, kwargs):
        print "RUNNING NEW INITIALIZE"
        opt_param(self, ['shortcut_deriv'], True)
        super(CrossEntropy, self).initialize(kwargs) # not the problmen
        if isinstance(self.olayer.activation, Softmax):
            self.ce_function = cross_entropy_multi
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
                print "ce:init:Softmax skip" # no
            else:
                self.cd_function = cross_entropy_multi_derivative
        elif isinstance(self.olayer.activation, Logistic):
            self.ce_function = cross_entropy
            print "self.ce_function1 = cross_entropy" # yes
            if self.shortcut_deriv:
                self.cd_function = shortcut_derivative
                self.olayer.skip_act = True
                print "using shortcut_derivative" # yes
            else:
                self.cd_function = cross_entropy_derivative
                print "self.cd_function = cross_entropy_derivative" # no
        else:
            self.ce_function = cross_entropy
            self.cd_function = cross_entropy_derivative
            print "self.ce_function2 = cross_entropy" # no
        #import ipdb; ipdb.set_trace()
        self.cd_function = cross_entropy_derivative # FUCKING HELL!

# -- old functions --
    def o_initialize(self, kwargs):
        print "2. RUNNING OLD INITIALIZE"
        self.__dict__.update(kwargs)
        if not hasattr(self, 'backend'):
            self.backend = self.olayer.backend

        if not hasattr(self, 'batch_size'):
            self.batch_size = self.olayer.batch_size
        print "super batchsize", self.batch_size

        if not hasattr(self, 'olayer_data'):
            self.olayer_data = 'output'

        if not hasattr(self.olayer, self.olayer_data):
            raise ValueError("Layer %s does not have buffer %s" %
                             (self.olayer.name, self.olayer_data))
        else:
            self.set_outputbuf(getattr(self.olayer, self.olayer_data))

        import ipdb; ipdb.set_trace()


    def o__init__(self, **kwargs):
        print "1. RUNNING OLD INIT"
        super(CrossEntropy, self).__init__(**kwargs)
        # The default cross entropy to use is where each output node
        # can take on value with binary prob
        if not hasattr(self, 'use_binary'):
            self.use_binary = True
            print "self.use_binary = True" # yes

        if not hasattr(self, 'shortcut_deriv'):
            self.shortcut_deriv = False
            if (isinstance(self.olayer.activation, Logistic)
                    and self.use_binary):
                self.shortcut_deriv = True
                print "self.shortcut_deriv = True" # yes

            if (isinstance(self.olayer.activation, Softmax)
                    and not self.use_binary):
                self.shortcut_deriv = True
                print "self.shortcut_deriv = True"

        # Set the appropriate functions
        self.ce_function = cross_entropy
        self.cd_function = cross_entropy_derivative
        print "self.ce_function = cross_entropy" # yes

        if not self.use_binary:
            self.ce_function = cross_entropy_multi
            self.cd_function = cross_entropy_multi_derivative
            print "self.ce_function = cross_entropy_multi" # no


# -- this stuff is ok --
    def __str__(self):
        return ("Cost Function: {bnry} {shrtct}\n".format(
                bnry=self.use_binary, shrtct=self.shortcut_deriv))

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf, tempbuf.copy()]
        self.outputbuf = databuf

    def set_berrbuf(self):
        # THIS IS NEW, used by layer2 only.
        return self.temp[0]

    def apply_function(self, targets):
        """
        Apply the cross entropy cost function to the datasets passed.
        """
        return self.ce_function(self.backend, self.outputbuf,
                                targets, self.temp) * self.scale

    def apply_derivative(self, targets):
        """
        Apply the derivative of the cross entropy cost function to the datasets
        passed.
        """
        return self.cd_function(self.backend, self.outputbuf,
                                targets, self.temp, self.scale)

