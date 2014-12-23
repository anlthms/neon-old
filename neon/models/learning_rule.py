# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Classes used to control how updates are applied to coefficients
i.e. how the learning should proceed.
"""

import logging

logger = logging.getLogger(__name__)


class LearningRule(object):
    """
    Base object for applying learning rule on the parameters to be updated

    Attributes:
        name (str): Used to identify this LearningRule when logging.
        backend (neon.backends.backend.Backend): underlying type for stored
                                                 data parameters like weights.
        batch_size (int): Number of examples presented at this iteration
    """
    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
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

    def set_pretrain_mode(self, pretrain_mode):
        pass

    def apply_rule(self, params, updates, epoch):
        raise NotImplementedError()


class GradientDescent(LearningRule):
    """
    Vanilla gradient descent based update rule that can optionally support use
    of weight decay.
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
        for ps_item, us_item in zip(params, updates):
            self.backend.multiply(us_item,
                                  self.backend.wrap(self.learning_rate),
                                  out=us_item)
            self.backend.subtract(ps_item, us_item, out=ps_item)


class GradientDescentPretrain(GradientDescent):
    """
    Gradient descent based variant that also supports a separate learning
    rate during pre-training.
    """
    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentPretrain, self).__init__(name, lr_params)
        if 'pretrain_learning_rate' in lr_params:
            self.pretrain_learning_rate = lr_params['pretrain_learning_rate']
        else:
            raise AttributeError("Missing required pretrain learning rate")

        self.train_learning_rate = self.learning_rate
        self.pretrain_mode = False

    def set_pretrain_mode(self, pretrain_mode):
        if (pretrain_mode):
            self.learning_rate = self.pretrain_learning_rate
        else:
            self.learning_rate = self.train_learning_rate

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item in zip(params, updates):
            self.backend.multiply(us_item,
                                  self.backend.wrap(self.learning_rate),
                                  out=us_item)
            self.backend.subtract(ps_item, us_item, out=ps_item)


class GradientDescentMomentum(GradientDescent):
    """
    Gradient descent learning rate variant that supports different types of
    momentum based updates
    """
    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(GradientDescentMomentum, self).__init__(name, lr_params)
        if 'momentum_params' in lr_params:
            self.momentum_params = lr_params['momentum_params']
        else:
            raise AttributeError("Missing required momentum parameters")
        self.velocity = []
        self.velocity_rec = None
        self.velocity_dtype = param_dtype

    def allocate_state(self, params):
        self.velocity = []
        for item in params:
            self.velocity.append(self.backend.zeros(item.shape,
                                                    self.velocity_dtype))

    def allocate_state_rec(self, params):
        """For recurrent layer, need an extra velocity """
        if (self.velocity_rec is None) \
                or (self.velocity_rec.shape != params.shape):
                    self.velocity_rec = self.backend.zeros(params.shape,
                                                           self.velocity_dtype)

    def apply_rule_rec(self, params, updates, epoch):
        """ For recurrent layer, need an extra velocity """
        momentum_coef = self.get_momentum_coef(epoch)
        self.backend.multiply(self.velocity_rec,
                              self.backend.wrap(momentum_coef),
                              out=self.velocity_rec)
        self.backend.multiply(updates,
                              self.backend.wrap(self.learning_rate),
                              out=updates)
        self.backend.subtract(self.velocity_rec,
                              updates,
                              out=self.velocity_rec)
        self.backend.add(params, self.velocity_rec, out=params)

    def apply_rule(self, params, updates, epoch):
        """
        Steps for momentum:
        1. velo = mu * velo    scale down old velocity
        2. upda = eps * upda   scale down new updates
        3. velo = velo - upda  combine old and new part
        4. update the actual weights.
        """
        momentum_coef = self.get_momentum_coef(epoch)
        for ps_item, us_item, vs_item in zip(params, updates, self.velocity):
            self.backend.multiply(vs_item,
                                  self.backend.wrap(momentum_coef),
                                  out=vs_item)
            self.backend.multiply(us_item,
                                  self.backend.wrap(self.learning_rate),
                                  out=us_item)
            self.backend.subtract(vs_item, us_item, out=vs_item)
            self.backend.add(ps_item, vs_item, out=ps_item)

    def get_momentum_coef(self, epoch):
        """
        Explanation here what the different momentum parameters mean.
        initial_coef:   momentum coefficient used from first epoch on
        saturated_coef: momentum after saturate_epoch is reached
        start_epoch:    start increasing momentum at this epoch
        saturate_epoch: saturated_coef is reached and held
        ...
        """
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
            if 'coef' not in self.momentum_params:
                coef = init_coef
        elif self.momentum_params['type'] == 'linear_monotone':
            coef = init_coef
            if start_epoch is not None and epoch >= start_epoch:
                if saturate_epoch is not None and epoch <= saturate_epoch:
                    if start_epoch == saturate_epoch:
                        coef = saturated_coef
                    else:
                        init_proportion = 1 - ((epoch - start_epoch + 0.0) /
                                               (saturate_epoch - start_epoch))
                        coef = (init_proportion * init_coef +
                                (1.0 - init_proportion) * saturated_coef)
                elif saturate_epoch is not None and epoch > saturate_epoch:
                    coef = saturated_coef
            else:
                pass
        elif self.momentum_params['type'] == 'nesterov':
            raise NotImplementedError("TODO!")
        else:
            raise AttributeError("invalid momentum_params specified")
        return coef


# TODO:  Use the built-in ada-delta update funcs in the backends to make this
# cleaner/faster
class AdaDelta(LearningRule):
    """
    Adadelta based learning rule updates.  See Zeiler2012 for instance.
    """
    def __init__(self, name, lr_params, param_dtype=None, gradient_dtype=None):
        super(AdaDelta, self).__init__(name, lr_params)
        if 'rho' in lr_params:
            self.rho = lr_params['rho']
        else:
            raise AttributeError("Missing required parameter rho")
        if 'epsilon' in lr_params:
            self.epsilon = lr_params['epsilon']
        else:
            raise AttributeError("Missing required parameter epsilon")
        self.exp_gradsq_dtype = param_dtype
        self.exp_deltsq_dtype = param_dtype
        self.scratch_space_dtype = param_dtype
        self.lrates_dtype = param_dtype
        self.lrates_dtype = param_dtype
        self.exp_gradsq = []
        self.exp_deltsq = []
        self.lrates = []
        self.scratch_space = []

    def allocate_state(self, params):
        assert len(self.exp_gradsq) == 0
        for item in params:
            self.exp_gradsq.append(self.backend.zeros(item.shape,
                                                      self.exp_gradsq_dtype))
            self.exp_deltsq.append(self.backend.zeros(item.shape,
                                                      self.exp_deltsq_dtype))
            self.lrates.append(self.backend.zeros(item.shape,
                                                  self.lrates_dtype))
            self.scratch_space.append(self.backend.zeros(
                item.shape, self.scratch_space_dtype))

    def apply_rule(self, params, updates, epoch):
        for ps_item, us_item, gs_item, ds_item, ls_item, ss_item in zip(
                params, updates, self.exp_gradsq,
                self.exp_deltsq, self.lrates, self.scratch_space):
            self.backend.ada_update(ps_item, us_item, gs_item, ds_item,
                                    ls_item, ss_item, self.rho, self.epsilon)
