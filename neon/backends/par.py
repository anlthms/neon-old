import logging
import numpy as np
from neon.util.compat import MPI_INSTALLED, mpi_size, mpi_rank

if MPI_INSTALLED:
    from mpi4py import MPI
    from mpi4py.MPI import COMM_WORLD as comm

logger = logging.getLogger(__name__)


class NoPar(object):

    def __init__(self, backend, model):
        self.backend = backend

    def distributable(self, layer):
        if hasattr(layer, 'distributable'):
            return layer.distributable
        return False

    def distribute(self, batchdata):
        return self.backend.array(batchdata)

    def rank(self):
        return mpi_rank

    def reduce_cost(self, tensor):
        return tensor.asnumpyarray()


class ModelPar(NoPar):

    class Config(object):
        pass

    def __init__(self, backend, model):
        super(ModelPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Model-parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc
        self.init_model(model, backend)

    def init_model(self, model, backend):
        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = ModelPar.Config()
            nout = layer.nout
            realnin = layer.nin
            nin = realnin / mpi_size
            conf.start = mpi_rank * nin
            if mpi_rank == (mpi_size - 1):
                # If the weights cannot be evenly partitioned, let the last
                # MPI node handle the extra weights.
                conf.end = realnin
            else:
                conf.end = conf.start + nin
            bs = model.batch_size
            bufshape = (layer.nout, bs)
            conf.fpropbuf = np.empty(bufshape, dtype=np.float32)
            bufshape = (layer.nin, bs)
            conf.bpropbuf = np.empty(bufshape, dtype=np.float32)
            conf.rcount = np.empty(mpi_size, dtype=np.int32)
            conf.rcount.fill(nin)
            conf.scount = conf.end - conf.start
            conf.rcount[-1] = realnin - nin * (mpi_size - 1)
            conf.displ = np.arange(0, realnin - nin + 1, nin)
            conf.scount *= bs
            conf.rcount *= bs
            conf.displ *= bs
            layer.weight_shape = (nout, conf.end - conf.start)
            layer.parconf = conf

    def fprop_fc(self, out, inputs, weights, layer):
        conf = layer.parconf
        self.orig_fprop_fc(out, inputs[conf.start:conf.end], weights)
        sendbuf = [out.asnumpyarray(), MPI.FLOAT]
        recvbuf = [conf.fpropbuf, MPI.FLOAT]
        comm.Reduce(sendbuf, recvbuf, op=MPI.SUM)
        comm.Bcast(buf=[conf.fpropbuf, MPI.FLOAT])
        out[:] = self.backend.array(conf.fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        conf = layer.parconf
        self.orig_bprop_fc(out[conf.start:conf.end], weights, deltas)
        outbuf = out.asnumpyarray()[conf.start:conf.end]
        sendbuf = [outbuf, conf.scount, MPI.FLOAT]
        recvbuf = [conf.bpropbuf, conf.rcount,
                   conf.displ, MPI.FLOAT]
        comm.Allgatherv(sendbuf, recvbuf)
        out[:] = self.backend.array(conf.bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        conf = layer.parconf
        self.orig_update_fc(out, inputs[conf.start:conf.end], deltas)


class DataPar(NoPar):

    class Config(object):
        pass

    def __init__(self, backend, model):
        super(DataPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Data-parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_update_fc = backend.update_fc
        self.orig_update_conv = backend.update_conv
        self.init_model(model, backend)
        self.reducebuf = np.empty((1, 1), dtype=np.float32)

    def init_model(self, model, backend):
        self.batch_size = backend.actual_batch_size / mpi_size
        self.start = mpi_rank * self.batch_size
        if mpi_rank == (mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size
        model.batch_size = self.batch_size

        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = DataPar.Config()
            conf.updatebuf = np.empty(layer.weight_shape, dtype=np.float32)
            layer.parconf = conf

    def distribute(self, batchdata):
        return self.backend.array(batchdata[:, self.start:self.end])

    def reduce_cost(self, tensor):
        comm.Reduce([tensor.asnumpyarray(), MPI.FLOAT],
                    [self.reducebuf, MPI.FLOAT], op=MPI.SUM)
        if mpi_rank == 0:
            return self.reducebuf / mpi_size
        return 0

    def update(self, out, conf):
        # NOTE: To make this faster, compute the weight updates
        # asynchronously. There is no need to wait for completion
        # until the updates are to be applied to the weights (the
        # weights are updated after the gradients are propagated
        # all the way back).
        sendbuf = [out.asnumpyarray(), MPI.FLOAT]
        recvbuf = [conf.updatebuf, MPI.FLOAT]
        comm.Reduce(sendbuf, recvbuf, op=MPI.SUM)
        comm.Bcast(buf=[conf.updatebuf, MPI.FLOAT])
        out[:] = self.backend.array(conf.updatebuf)

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs, deltas)
        self.update(out, layer.parconf)

    def update_conv(self, out, inputs, weights, deltas, ofmshape, ofmsize,
                    ofmlocs, ifmshape, links, nifm, padding, stride,
                    ngroups, fwidth, updatebuf, local=False, layer=None):
        self.orig_update_conv(out, inputs, weights, deltas, ofmshape, ofmsize,
                              ofmlocs, ifmshape, links, nifm, padding, stride,
                              ngroups, fwidth, updatebuf, local)
        self.update(out, layer.parconf)
