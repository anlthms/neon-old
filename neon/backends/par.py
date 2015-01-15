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
    def __init__(self, backend, model):
        super(ModelPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Model parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_gen_weights = backend.gen_weights
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc

    def configure(self, layer):
        assert hasattr(layer, 'nin')
        nout = layer.nout
        realnin = layer.nin
        nin = realnin / mpi_size
        layer.par_start = mpi_rank * nin
        if mpi_rank == (mpi_size - 1):
            # If the weights cannot be evenly partitioned, let the last
            # MPI node handle the extra weights.
            layer.par_end = realnin
        else:
            layer.par_end = layer.par_start + nin
        layer.par_fpropbuf = np.empty((layer.nout, layer.batch_size),
                                      dtype=np.float32)
        layer.par_bpropbuf = np.empty((layer.nin, layer.batch_size),
                                      dtype=np.float32)
        layer.par_rcount = np.empty(mpi_size, dtype='int32')
        layer.par_rcount.fill(nin)
        layer.par_scount = layer.par_end - layer.par_start
        layer.par_rcount[-1] = realnin - nin * (mpi_size - 1)
        layer.par_displ = np.arange(0, realnin - nin + 1, nin)
        layer.par_scount *= layer.batch_size
        layer.par_rcount *= layer.batch_size
        layer.par_displ *= layer.batch_size

    def gen_weights(self, size, weight_params, dtype=None, layer=None):
        assert layer is not None
        self.configure(layer)
        weights = self.orig_gen_weights(size, weight_params, dtype)
        weights = weights[:, layer.par_start:layer.par_end]
        return weights

    def fprop_fc(self, out, inputs, weights, layer):
        self.orig_fprop_fc(out, inputs[layer.par_start:layer.par_end], weights)
        comm.Reduce(sendbuf=[out.asnumpyarray(), MPI.FLOAT],
                    recvbuf=[layer.par_fpropbuf, MPI.FLOAT],
                    op=MPI.SUM)
        comm.Bcast(buf=[layer.par_fpropbuf, MPI.FLOAT])
        out[:] = self.backend.array(layer.par_fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        self.orig_bprop_fc(out[layer.par_start:layer.par_end],
                           weights, deltas)
        comm.Allgatherv(
            sendbuf=[out.asnumpyarray()[layer.par_start:layer.par_end],
                     layer.par_scount, MPI.FLOAT],
            recvbuf=[layer.par_bpropbuf, layer.par_rcount,
                     layer.par_displ, MPI.FLOAT])
        out[:] = self.backend.array(layer.par_bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        self.orig_update_fc(out, inputs[layer.par_start:layer.par_end],
                            deltas)


class DataPar(NoPar):
    def __init__(self, backend, model):
        super(DataPar, self).__init__(backend, model)
        if mpi_rank == 0:
            logger.info('Data parallel mode. Number of nodes = %d.', mpi_size)
        self.orig_gen_weights = backend.gen_weights
        self.orig_update_fc = backend.update_fc
        self.batch_size = backend.actual_batch_size / mpi_size
        self.start = mpi_rank * self.batch_size
        if mpi_rank == (mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size

        model.batch_size = self.batch_size
        logger.debug('Node: %d: Changed batch size from %d to %d',
                     mpi_rank, backend.actual_batch_size, model.batch_size)

    def configure(self, layer):
        assert hasattr(layer, 'nin')
        layer.par_updatebuf = np.empty((layer.nout, layer.nin),
                                       dtype=np.float32)

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
        comm.Reduce(sendbuf=[out.asnumpyarray(), MPI.FLOAT],
                    recvbuf=[layer.par_updatebuf, MPI.FLOAT],
                    op=MPI.SUM)
        comm.Bcast(buf=[layer.par_updatebuf, MPI.FLOAT])
        out[:] = self.backend.array(layer.par_updatebuf)
