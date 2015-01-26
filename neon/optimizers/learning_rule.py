# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic parent class used to control how updates are applied to coefficients
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
