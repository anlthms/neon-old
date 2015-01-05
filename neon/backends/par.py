import numpy as np
from neon.util.compat import MPI_INSTALLED, mpi_size, mpi_rank, range

if MPI_INSTALLED:
    from mpi4py import MPI


class VecPar:
    def __init__(self, backend):
        self.backend = backend

    def gen_weights(self, size, weight_params, dtype=None):
        nout, realnin = size
        nin = realnin / mpi_size
        if mpi_rank == (mpi_size - 1):
            # If there are any extra weights, let the last MPI node handle it.
            nin += realnin % mpi_size
        start = mpi_rank * nin
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
        recvobj = MPI.COMM_WORLD.reduce(sendobj=out.raw(), op=MPI.SUM)
        recvobj = MPI.COMM_WORLD.bcast(obj=recvobj)
        out._tensor = self.backend.array(recvobj)._tensor

    def bprop_fc(self, out, weights, deltas):
        realnin = out.shape[0]
        nin = realnin / mpi_size
        start = mpi_rank * nin
        end = start + weights.shape[1]
        self.backend.bprop_fc_impl(out[start:end], weights, deltas)
        recvlst = MPI.COMM_WORLD.allgather(sendobj=out.raw()[start:end])
        recvobj = np.vstack(recvlst)
        out._tensor = self.backend.array(recvobj)._tensor

    def update_fc(self, out, inputs, deltas):
        realnin = inputs.shape[0]
        nin = realnin / mpi_size
        start = mpi_rank * nin
        end = start + out.shape[1]
        self.backend.update_fc_impl(out, inputs[start:end], deltas)
