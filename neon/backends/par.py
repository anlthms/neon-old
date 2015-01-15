import logging
import numpy as np
from neon.util.compat import MPI_INSTALLED, mpi_size, mpi_rank, range

if MPI_INSTALLED:
    from mpi4py import MPI
    from mpi4py.MPI import COMM_WORLD as comm

logger = logging.getLogger(__name__)


class NoPar(object):

    def __init__(self, backend, model):
        self.backend = backend

    def distribute(self, batchdata):
        return self.backend.array(batchdata)


class ModelPar(NoPar):

    class Config(object):
        pass

    def __init__(self, backend, model):
        super(ModelPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Model-parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_gen_weights = backend.gen_weights
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc

    def configure(self, layer):
        assert hasattr(layer, 'nin')
        assert not hasattr(layer, 'par')
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
        bufshape = (layer.nout, layer.batch_size)
        conf.fpropbuf = np.empty(bufshape, dtype=np.float32)
        bufshape = (layer.nin, layer.batch_size)
        conf.bpropbuf = np.empty(bufshape, dtype=np.float32)
        conf.rcount = np.empty(mpi_size, dtype=np.int32)
        conf.rcount.fill(nin)
        conf.scount = conf.end - conf.start
        conf.rcount[-1] = realnin - nin * (mpi_size - 1)
        conf.displ = np.arange(0, realnin - nin + 1, nin)
        conf.scount *= layer.batch_size
        conf.rcount *= layer.batch_size
        conf.displ *= layer.batch_size
        layer.par = conf

    def gen_weights(self, size, weight_params, dtype=None, layer=None):
        assert layer is not None
        self.configure(layer)
        weights = self.orig_gen_weights(size, weight_params, dtype)
        weights = weights[:, layer.par.start:layer.par.end]
        return weights

    def fprop_fc(self, out, inputs, weights, layer):
        self.orig_fprop_fc(out, inputs[layer.par.start:layer.par.end], weights)
        sendbuf = [out.asnumpyarray(), MPI.FLOAT]
        recvbuf = [layer.par.fpropbuf, MPI.FLOAT]
        comm.Reduce(sendbuf, recvbuf, op=MPI.SUM)
        comm.Bcast(buf=[layer.par.fpropbuf, MPI.FLOAT])
        out[:] = self.backend.array(layer.par.fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        self.orig_bprop_fc(out[layer.par.start:layer.par.end], weights, deltas)
        outbuf = out.asnumpyarray()[layer.par.start:layer.par.end]
        sendbuf = [outbuf, layer.par.scount, MPI.FLOAT]
        recvbuf = [layer.par.bpropbuf, layer.par.rcount,
                   layer.par.displ, MPI.FLOAT]
        comm.Allgatherv(sendbuf, recvbuf)
        out[:] = self.backend.array(layer.par.bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs[layer.par.start:layer.par.end], deltas)


class DataPar(NoPar):

    class Config(object):
        pass

    def __init__(self, backend, model):
        super(DataPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Data-parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_gen_weights = backend.gen_weights
        self.orig_update_fc = backend.update_fc
        self.batch_size = backend.actual_batch_size / mpi_size
        self.start = mpi_rank * self.batch_size
        if mpi_rank == (mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size
        model.batch_size = self.batch_size

    def configure(self, layer):
        assert hasattr(layer, 'nin')
        assert not hasattr(layer, 'par')
        conf = DataPar.Config()
        bufshape = (layer.nout, layer.nin)
        conf.updatebuf = np.empty(bufshape, dtype=np.float32)
        layer.par = conf

    def distribute(self, batchdata):
        return self.backend.array(batchdata[:, self.start:self.end])

    def gen_weights(self, size, weight_params, dtype=None, layer=None):
        self.configure(layer)
        return self.orig_gen_weights(size, weight_params, dtype)

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs, deltas)
        # NOTE: To make this faster, compute the weight updates
        # asynchronously. There is no need to wait for completion
        # until the updates are to be applied to the weights (the
        # weights are updated after the gradients are propagated
        # all the way back).
        sendbuf = [out.asnumpyarray(), MPI.FLOAT]
        recvbuf = [layer.par.updatebuf, MPI.FLOAT]
        comm.Reduce(sendbuf, recvbuf, op=MPI.SUM)
        comm.Bcast(buf=[layer.par.updatebuf, MPI.FLOAT])
        out[:] = self.backend.array(layer.par.updatebuf)
