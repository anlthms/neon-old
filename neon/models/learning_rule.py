"""
Generic updating class
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class LearningRule(object):

    """
    LearningRule object for applying learning rule on params to be updated

    Attributes:
        name (str): Used to identify this LearningRule when logging.
        backend (neon.backends.backend.Backend): underlying type for stored
                                                    data parameters like
                                                    weights.
        batch_size (int): Number of examples presented at this iteration
    """

    def __init__(self, name, lr_params, param_dtype=None,     gradient_dtype=None):
        self.name = name
        self.backend = lr_params['backend']
        self.param_dtype = param_dtype
        self.gradient_dtype = gradient_dtype


    def __str__(self):
        return ("LearningRule {upd_nm}: {upd_tp} upd_rl, "
                "utilizing {be_nm} backend\n\t".format
                (upd_nm=self.name, upd_tp=self.__class__.__name__,
                 be_nm=self.backend.__class__.__name__))

    def allocate_state(self, params):
        pass

    def apply_rule(self, params, updates, epoch):
        raise NotImplementedError()

class GradientDescent(LearningRule):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescent, self).__init__(name, lr_params)
        if 'learning_rate' in lr_params:
            self.learning_rate = lr_params['learning_rate']
        else:
            raise AttributeError("Missing required learning rate")

        if 'weight_decay' in lr_params:
            self.weight_decay = lr_params['weight_decay']
        else:
            self.weight_decay = 0.0

    def apply_rule(self, params, updates, epoch):
        self.backend.multiply(updates, self.backend.wrap(self.learning_rate),
                              out=updates)
        self.backend.subtract(params, updates, out=params)


class GradientDescentPretrain(GradientDescent):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentPretrain, self).__init__(name, lr_params)
        if 'pretrain_learning_rate' in lr_params:
            self.pretrain_learning_rate = lr_params['pretrain_learning_rate']
        else:
            raise AttributeError("Missing required pretrain learning rate")

        self.pretrain_mode = False

    def set_pretrain_mode(self, pretrain_mode):
        self.pretrain_mode = pretrain_mode

    def apply_rule(self, params, updates, epoch):
        if (self.pretrain_mode):
            lr = self.learning_rate
        else:
            lr = self.pretrain_learning_rate

        self.backend.multiply(updates, self.backend.wrap(lr), out=updates)
        self.backend.subtract(params, updates, out=params)

class GradientDescentMomentum(GradientDescent):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentMomentum, self).__init__(name, lr_params)
        if 'momentum_params' in lr_params:
            self.momentum_params = lr_params['momentum_params']
        else:
            raise AttributeError("Missing required momentum parameters")
        self.velocity = None
        self.velocity_dtype = param_dtype

    def allocate_state(self, params):
        if (self.velocity == None):
            self.velocity = self.backend.zeros(params.shape, self.velocity_dtype)

    def apply_rule(self, params, updates, epoch):
        momentum_coef = self.get_momentum_coef(epoch)
        self.backend.multiply(self.velocity, self.backend.wrap(momentum_coef),
                              out=self.velocity)
        self.backend.multiply(updates,
                              self.backend.wrap(self.learning_rate),
                              out=updates)
        self.backend.subtract(self.velocity, updates, out=self.velocity)
        self.backend.add(params, self.velocity, out=params)

    def get_momentum_coef(self, epoch):
        coef = 0.0
        if 'coef' in self.momentum_params:
            coef = self.momentum_params['coef']
        if 'initial_coef' in self.momentum_params:
            init_coef = self.momentum_params['initial_coef']
        else:
            init_coef = coef
        if 'saturated_coef' in self.momentum_params:
            saturated_coef = self.momentum_params['saturated_coef']
        else:
            saturated_coef = coef
        if 'start_epoch' in self.momentum_params:
            start_epoch = self.momentum_params['start_epoch']
        else:
            start_epoch = None
        if 'saturate_epoch' in self.momentum_params:
            saturate_epoch = self.momentum_params['saturate_epoch']
        else:
            saturate_epoch = None

        if self.momentum_params['type'] == 'constant':
            pass
        elif self.momentum_params['type'] == 'linear_monotone':
            coef = init_coef
            if start_epoch is not None and epoch >= start_epoch:
                if saturate_epoch is not None and epoch <= saturate_epoch:
                    if start_epoch == saturate_epoch:
                        coef = saturated_coef
                    else:
                        init_proportion = ((epoch - start_epoch + 0.0) /
                                           (saturate_epoch - start_epoch))
                        coef = (init_proportion * init_coef +
                                (1.0 - init_proportion) * saturated_coef)
                elif saturate_epoch is not None and epoch > saturate_epoch:
                    coef = saturated_coef
            else:
                coef = saturated_coef
        elif self.momentum_params['type'] == 'nesterov':
            raise NotImplementedError("TODO!")
        else:
            raise AttributeError("invalid momentum_params specified")
        return coef


# TODO:  Use the built-in ada-delta update funcs in the backends to make this cleaner/faster
class AdaDelta(LearningRule):

    """
    Single NNet layer with no bias node
    """

    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(AdaDelta, self).__init__(name, lr_params)
        self.rho = rho
        self.epsilon = epsilon
        self.exp_gradsq_dtype = param_dtype
        self.exp_deltsq_dtype = param_dtype
        self.lrates_dtype = param_dtype
        self.lrates_dtype = param_dtype
        self.exp_gradsq = None
        self.exp_deltsq = None
        self.lrates = None
        self.scratch_space = None

    def allocate_state(self, params):

        if (self.exp_gradsq == None):
            self.exp_gradsq = self.backend.zeros(params.shape, self.exp_gradsq_dtype)
        if (self.exp_deltsq == None):
            self.exp_deltsq = self.backend.zeros(params.shape, self.exp_deltsq_dtype)
        if (self.lrates == None):
            self.lrates = self.backend.zeros(params.shape, self.lrates_dtype)
        if (self.scratch_space == None):
            self.scratch_space = self.backend.zeros(params.shape, self.scratch_space_dtype)

    def apply_rule(self, params, updates, epoch):
        momentum_coef = self.get_momentum_coef(epoch)

        # Accumulate E[Grad^2]
        self.backend.multiply(self.exp_gradsq, self.backend.wrap(self.rho),
                              out=self.exp_gradsq)
        self.backend.multiply(updates, updates,
                              out=self.scratch_space)
        self.backend.multiply(self.scratch_space, self.backend.wrap(1.0-self.rho),
                              out=self.scratch_space)
        self.backend.add(self.exp_gradsq, self.scratch_space, out=self.exp_gradsq)

        # Calculate Updates
        self.backend.add(self.exp_gradsq, self.backend.wrap(self.epsilon),
                         out=self.scratch_space)
        self.backend.add(self.exp_deltsq, self.backend.wrap(self.epsilon),
                         out=self.lrates)
        self.backend.divide(self.lrates, self.scratch_space, out=self.lrates)
        self.backend.sqrt(self.lrates, out=self.lrates)
        self.backend.multiply(self.lrates, self.backend.wrap(-1.0), out=self.lrates)
        self.backend.multiply(self.lrates, updates, out=self.lrates)

        # Accumulate E[Delt^2]
        self.backend.multiply(self.exp_deltsq, self.backend.wrap(self.rho),
                              out=self.exp_deltsq)
        self.backend.multiply(self.lrates, self.lrates,
                              out=self.scratch_space)
        self.backend.multiply(self.scratch_space, self.backend.wrap(1.0-self.rho),
                              out=self.scratch_space)
        self.backend.add(self.exp_deltsq, self.scratch_space, out=self.exp_deltsq)

        # Final update to the params
        self.backend.add(params, self.lrates, out=params)
