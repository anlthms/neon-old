# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Experiment in which a model is trained (parameters learned)
"""

import logging

from neon.experiments.experiment import Experiment
from neon.util.param import req_param, opt_param
from neon.util.compat import StringIO
import numpy as np

logger = logging.getLogger(__name__)


def recv_msg(socket, flags=0, copy=True, track=False):
    msg = socket.recv(flags=flags, copy=copy, track=track)
    return msg


def msg_to_img(msgstring, input_shape):
    from PIL import Image
    im = Image.open(StringIO(msgstring))
    if (im.size[0], im.size[1]) != input_shape:
        im = im.resize(input_shape, Image.ANTIALIAS)
    return im


class InferenceServer(Experiment):

    """
    In this `Experiment`, a model is trained on a training dataset to
    learn a set of parameters

    Note that a pre-fit model may be loaded depending on serialization
    parameters (rather than learning from scratch).  The same may also apply to
    the datasets specified.

    Kwargs:
        backend (neon.backends.Backend): The backend to associate with the
                                            datasets to use in this experiment
    TODO:
        add other params
    """

    def __init__(self, **kwargs):
        self.dist_flag = False
        self.datapar = False
        self.modelpar = False
        self.initialized = False
        self.__dict__.update(kwargs)
        req_param(self, ['dataset', 'model'])
        opt_param(self, ['backend'])
        opt_param(self, ['port'], 55542)

    def initialize(self, backend):
        if self.initialized:
            return
        self.backend = backend
        self.model.link()
        self.backend.par.init_model(self.model, self.backend)
        self.model.initialize(backend)
        self.initialized = True

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        import zmq

        for ll in self.model.layers[1:]:
            ll.reallocate_output_bufs(1)

        if self.model.data_layer.is_local:
            img_shape = self.model.data_layer.ofmshape
        else:
            imdim = int(np.sqrt(self.model.data_layer.nout))
            img_shape = (imdim, imdim)

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:" + str(self.port))

        input_shape = (self.model.data_layer.nout, 1)
        input_data = self.backend.empty(input_shape, dtype='float32')

        # Kick off a loop to wait for data
        while True:
            imgstring = recv_msg(socket)
            img = msg_to_img(imgstring, img_shape)
            input_data.copy_from(
                np.array(img, dtype='float32').reshape(input_shape) / 255.)
            logger.info("Received and decoded image")
            self.model.fprop_skip_data_layer(input_data)

            results = self.model.class_layer.output.asnumpyarray()
            hyp = np.argmax(results, axis=0)
            logger.info("Most Likely Label is %d", hyp)
            socket.send(results)
