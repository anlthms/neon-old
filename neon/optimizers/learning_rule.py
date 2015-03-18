# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Generic parent class used to control how updates are applied to coefficients
i.e. how the learning should proceed.
"""

from neon.util.param import opt_param, req_param
import logging

logger = logging.getLogger(__name__)


class LearningRule(object):

    """
    Base object for applying learning rule on the parameters to be updated

    Attributes:
        name (str): Used to identify this LearningRule when logging.
        batch_size (int): Number of examples presented at this iteration
    """
    def __init__(self, name, lr_params):
        self.name = name

        opt_param(self, ['half_precision'], False)
        if self.half_precision:
            import numpy as np
            setattr(self, 'velocity_dtype', np.float16)
            setattr(self, 'param_dtype', np.float16)
            setattr(self, 'gradient_dtype', np.float16)

    def initialize(self, backend):
        self.backend = backend

    def __str__(self):
        be_nm = ''
        if hasattr(self, 'backend'):
            be_nm = ", utilizing {} backend".format(
                    self.backend.__class__.__name__)
        return ("LearningRule {upd_nm}: {upd_tp} upd_rl{be_nm}\n\t".format(
                upd_nm=self.name, upd_tp=self.__class__.__name__, be_nm=be_nm))

    def allocate_state(self, params):
        pass

    def set_pretrain_mode(self, pretrain_mode):
        pass

    def apply_rule(self, params, updates, epoch):
        raise NotImplementedError()
