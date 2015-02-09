# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Classes used to control how updates are applied to coefficients
i.e. how the learning should proceed.
"""

import logging
from neon.optimizers.learning_rule import LearningRule

logger = logging.getLogger(__name__)


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