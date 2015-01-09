import logging
import numpy as np
from neon.util.compat import MPI_INSTALLED, mpi_size, mpi_rank, range

if MPI_INSTALLED:
    from mpi4py import MPI

logger = logging.getLogger(__name__)


class VecPar:
    def __init__(self, backend):
        if mpi_rank == 0:
            logger.info('MPI mode. Number of nodes = %d.', mpi_size)
        self.backend = backend

    def gen_weights(self, size, weight_params, dtype=None):
        nout, realnin = size
        nin = realnin / mpi_size
        start = mpi_rank * nin
        if mpi_rank == (mpi_size - 1):
            # If the weights cannot be evenly partitioned, let the last
            # MPI node handle the extra weights.
            end = realnin
        else:
            end = start + nin
        weights = self.backend.gen_weights_impl(size, weight_params, dtype)
        weights = weights[:, start:end]
        return weights

    def fprop_fc(self, out, inputs, weights):
        realnin = inputs.shape[0]
        nin = realnin / mpi_size
        start = mpi_rank * nin
        end = start + weights.shape[1]
        self.backend.fprop_fc_impl(out, inputs[start:end], weights)
        #TODO: avoid this allocation.
        recvbuf = np.zeros(out.shape, dtype=np.float32)
        MPI.COMM_WORLD.Reduce(sendbuf=[out.raw(), MPI.FLOAT],
                              recvbuf=[recvbuf, MPI.FLOAT], op=MPI.SUM)
        MPI.COMM_WORLD.Bcast(buf=[recvbuf, MPI.FLOAT])
        out[:] = self.backend.array(recvbuf)

    def bprop_fc(self, out, weights, deltas):
        realnin = out.shape[0]
        nin = realnin / mpi_size
        start = mpi_rank * nin
        end = start + weights.shape[1]
        self.backend.bprop_fc_impl(out[start:end], weights, deltas)

        #TODO: avoid this allocation.
        recvbuf = np.zeros(out.shape, dtype=np.float32)
        rcount = np.ones(mpi_size) * nin
        bs = out.shape[1]
        scount = end - start
        rcount[-1] = realnin - nin * (mpi_size - 1)
        displ = np.arange(0, realnin - nin + 1, nin)
        scount *= bs
        rcount *= bs
        displ *= bs
        MPI.COMM_WORLD.Allgatherv(sendbuf=[out.raw()[start:end], scount, MPI.FLOAT],
                                  recvbuf=[recvbuf, rcount, displ, MPI.FLOAT])
        out[:] = self.backend.array(recvbuf)

    def update_fc(self, out, inputs, deltas):
        realnin = inputs.shape[0]
        nin = realnin / mpi_size
        start = mpi_rank * nin
        end = start + out.shape[1]
        self.backend.update_fc_impl(out, inputs[start:end], deltas)
