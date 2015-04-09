import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class NoPar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None

    def init_model(self, model, backend):
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        self.backend = backend

    def distribute(self, src, dest=None):
        if dest is None:
            return self.backend.array(src)
        else:
            dest.copy_from(src)
            return dest

    def scatter(self, src, dest):
        dest.copy_from(src.T.astype(dest.dtype, order='C'))

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def rank(self):
        return 0

    def size(self):
        return 1

    def allocate_fragment(self, buf_shape, dtype=None):
        return self.backend.empty(buf_shape, dtype=dtype)


class BasePar(object):

    def __init__(self):
        self.backend = None
        self.device_id = None
        try:
            from mpi4py import MPI
            self.mpi = MPI
            self.comm = self.mpi.COMM_WORLD
            self.mpi_size = self.comm.size
            self.mpi_rank = self.comm.rank
        except ImportError:
            raise RuntimeError(
                "mpi4py not found, can't run in datapar or modelpar")

        try:
            # Determine local rank (assumes OpenMPI).
            self.mpi_local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.mpi_local_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        except:
            raise RuntimeError(
                "OpenMPI variable OMPI_COMM_WORLD_LOCAL_RANK or "
                "OMPI_COMM_WORLD_LOCAL_SIZE not found.\n"
                "Are you using: mpirun -n <#procs> neon <example.yaml>?")
        self.device_id = self.mpi_local_rank

    def init_model(self, model, backend):
        # save the original batch_size value that is specified in
        # the configuration file
        backend.actual_batch_size = model.batch_size

    def associate(self, backend):
        backend.par = self
        self.backend = backend

    def distribute(self, src, dest=None):
        raise NotImplementedError()

    def scatter(self, src, dest):
        raise NotImplementedError()

    def reduce_tensor(self, tensor):
        raise NotImplementedError()

    def allocate_fragment(self, buf_shape, dtype=None):
        raise NotImplementedError()

    def distributable(self, layer):
        if hasattr(layer, 'distributable'):
            return layer.distributable
        return False

    def rank(self):
        return self.mpi_rank

    def size(self):
        return self.mpi_size


class ModelPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(ModelPar, self).__init__()
        if self.mpi_rank == 0:
            logger.info('Model-parallel mode. Number of nodes = %d.',
                        self.mpi_size)

    def init_model(self, model, backend):
        super(ModelPar, self).init_model(model, backend)
        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = ModelPar.Config()
            nout = layer.nout
            realnin = layer.nin
            nin = realnin / self.mpi_size
            conf.start = self.mpi_rank * nin
            if self.mpi_rank == (self.mpi_size - 1):
                # If the weights cannot be evenly partitioned, let the last
                # MPI node handle the extra weights.
                conf.end = realnin
            else:
                conf.end = conf.start + nin
            bs = model.batch_size
            bufshape = (layer.nout, bs)
            conf.fpropbuf = backend.empty(bufshape, dtype=np.float32)
            bufshape = (layer.nin, bs)
            conf.bpropbuf = backend.empty(bufshape, dtype=np.float32)
            conf.rcount = np.empty(self.mpi_size, dtype=np.int32)
            conf.rcount.fill(nin)
            conf.scount = conf.end - conf.start
            conf.rcount[-1] = realnin - nin * (self.mpi_size - 1)
            conf.displ = np.arange(0, realnin - nin + 1, nin)
            conf.scount *= bs
            conf.rcount *= bs
            conf.displ *= bs
            layer.weight_shape = (nout, conf.end - conf.start)
            layer.parconf = conf

    def associate(self, backend):
        super(ModelPar, self).associate(backend)
        self.orig_fprop_fc = backend.fprop_fc
        self.orig_bprop_fc = backend.bprop_fc
        self.orig_update_fc = backend.update_fc
        backend.fprop_fc = self.fprop_fc
        backend.bprop_fc = self.bprop_fc
        backend.update_fc = self.update_fc

    def distribute(self, src, dest=None):
        if dest is None:
            return self.backend.array(src)
        else:
            dest.copy_from(src)
            return dest

    def scatter(self, src, dest):
        dest.copy_from(src.transpose().astype(dest.dtype))

    def allocate_fragment(self, buf_shape, dtype=None):
        return self.backend.empty(buf_shape, dtype=dtype)

    def reduce_tensor(self, tensor):
        return tensor.asnumpyarray()

    def fprop_fc(self, out, inputs, weights, layer):
        conf = layer.parconf
        self.orig_fprop_fc(out, inputs[conf.start:conf.end], weights)
        sendbuf = [out.asmpibuffer(), self.mpi.FLOAT]
        recvbuf = [conf.fpropbuf.asmpibuffer(), self.mpi.FLOAT]
        self.comm.Reduce(sendbuf, recvbuf, op=self.mpi.SUM)
        self.comm.Bcast(buf=recvbuf)
        out.copy_from(conf.fpropbuf)

    def bprop_fc(self, out, weights, deltas, layer):
        conf = layer.parconf
        outbuf = out[conf.start:conf.end]
        self.orig_bprop_fc(outbuf, weights, deltas)
        sendbuf = [outbuf.asmpibuffer(), conf.scount, self.mpi.FLOAT]
        recvbuf = [conf.bpropbuf.asmpibuffer(), conf.rcount,
                   conf.displ, self.mpi.FLOAT]
        self.comm.Allgatherv(sendbuf, recvbuf)
        out.copy_from(conf.bpropbuf)

    def update_fc(self, out, inputs, deltas, layer):
        conf = layer.parconf
        self.orig_update_fc(out, inputs[conf.start:conf.end], deltas)


class DataPar(BasePar):

    class Config(object):
        pass

    def __init__(self):
        super(DataPar, self).__init__()
        self.bdtype = self.mpi.FLOAT
        if self.mpi_rank == 0:
            logger.info('Data-parallel mode. Number of nodes = %d.',
                        self.mpi_size)

    def init_model(self, model, backend):
        super(DataPar, self).init_model(model, backend)
        self.batch_size = backend.actual_batch_size / self.mpi_size
        self.start = self.mpi_rank * self.batch_size
        if self.mpi_rank == (self.mpi_size - 1):
            self.batch_size = backend.actual_batch_size - self.start
        self.end = self.start + self.batch_size
        model.batch_size = self.batch_size

        for layer in model.layers:
            if not self.distributable(layer):
                continue
            assert hasattr(layer, 'nin')
            assert not hasattr(layer, 'parconf')
            conf = DataPar.Config()
            # conf.updatebuf = backend.empty(layer.weight_shape,
            #                                dtype=np.float32)
            conf.updatesz = layer.weight_shape[0] * layer.weight_shape[1]
            if self.mpi_rank == 0:
                conf.updatebuf = backend.empty((self.mpi_size, conf.updatesz),
                                               dtype=np.float32)
            layer.parconf = conf

    def associate(self, backend):
        super(DataPar, self).associate(backend)
        self.orig_update_fc = backend.update_fc
        self.orig_update_conv = backend.update_conv
        backend.update_fc = self.update_fc
        backend.update_conv = self.update_conv
        self.npreducebuf = np.empty((self.mpi_size, 1), dtype=np.float32)

    def distribute(self, src, dest=None):
        if dest is None:
            return self.backend.array(src[:, self.start:self.end])
        else:
            dest.copy_from(src[:, self.start:self.end])
            return dest

    def scatter(self, src, dest):
        if src is not None:
            # Assumption is that src is (batch_size * mpi_size, num_dims)
            dims = src.shape[1]
            src = src.reshape(
                self.mpi_size, self.batch_size, dims).transpose(
                0, 2, 1).reshape(
                self.mpi_size, dims * self.batch_size).astype(
                dest.dtype, order='C')
        # NOTE: Change dest buffer dtype if we end up using uint 8
        self.comm.Scatter([src, self.mpi.FLOAT],
                          [dest.asmpibuffer(), self.mpi.FLOAT], root=0)

    def allocate_fragment(self, buf_shape, dtype=None):
        fragment_buf_shape = (buf_shape[0], self.batch_size)
        return self.backend.empty(fragment_buf_shape, dtype=dtype)

    def reduce_tensor(self, tensor):
        # This is the case where we have a 1x1 tensor
        self.comm.Gather([tensor.asmpibuffer(), self.bdtype],
                         [self.npreducebuf, self.bdtype])
        if self.mpi_rank == 0:
            return self.npreducebuf.sum() / self.mpi_size
        return 0

    def update(self, out, conf):
        # NOTE: To make this faster, compute the weight updates
        # asynchronously. There is no need to wait for completion
        # until the updates are to be applied to the weights (the
        # weights are updated after the gradients are propagated
        # all the way back).

        # NOTE: We should be able to shard the updates and do summation in
        # parts across the different devices, but it seems to block in MPI

        gbuf = conf.updatebuf.asmpibuffer() if self.mpi_rank == 0 else None
        self.comm.Gather([out.asmpibuffer(), self.bdtype], [gbuf, self.bdtype])
        if self.mpi_rank == 0:
            orig_shape = out.shape
            out = out.reshape((1, conf.updatebuf.shape[1]))
            self.backend.sum(conf.updatebuf, axes=0, out=out)
            out = out.reshape(orig_shape)
        self.comm.Bcast([out.asmpibuffer(), self.bdtype])

        # logger.info('\tNode %d about to Bcast', self.mpi_rank)
        # print "about to update"
        # wsz, i, k = self.conf.updatesz, self.mpi_rank, self.mpi_size
        # shardsz = wsz / k
        # ubuf = conf.updatebuf

        # orig_shape = out.shape
        # out = out.reshape((wsz, 1))
        # self.comm.Gather([out[i * shardsz : (i + 1) * shardsz].asmpibuffer(),
        #                   shardsz, self.mpi.FLOAT],
        #                  [ubuf.asmpibuffer(), wsz, self.mpi.FLOAT], root=i)
        # ubuf = ubuf.reshape((k, shardsz))
        # out = out.reshape((k, shardsz))
        # self.backend.sum(ubuf, axis=0, out=out[i])
        # self.comm.Bcast([out[i].asmpibuffer(), shardsz, self.mpi.FLOAT],
        #                 root=i)
        # print "finished update reduction"
        # out = out.reshape(orig_shape)
        # conf.updatebuf = conf.updatebuf.reshape(orig_shape)
        # print "finished update"

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
