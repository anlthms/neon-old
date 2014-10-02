'''
Local View of the Data

'''


import numpy as np
from mpi4py import MPI
import gdist_consts as gc


def pprint(string, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        print(string)


class RecvHalo(object):

    def __init__(self, neighbor_array_index, halo_size, mb_size=1,
                 backend=None):

        self.neighbor_array_index = neighbor_array_index
        self.halo_size = halo_size
        self.halo_insert_indices = []
        # initialize a temporary buffer space if we need it
        if backend is None:
            self.halo_data = np.empty((mb_size, halo_size), dtype='float32')
        else:
            self.halo_data = backend.zeros(
                (mb_size, halo_size), dtype='float32')


class SendHalo(object):

    def __init__(self, target_array_index, halo_indices, mb_size=1,
                 backend=None):

        self.target_array_index = target_array_index
        self.halo_indices = halo_indices
        # space to store received defiltered data
        if backend is None:
            self.halo_data_defiltering = np.empty(
                (mb_size, len(halo_indices)), dtype='float32')
        else:
            self.halo_data_defiltering = backend.zeros(
                (mb_size, len(halo_indices)), dtype='float32')


class LocalArray(object):

    def __init__(self, batch_size=None, global_row_index=-1,
                 global_col_index=-1, height=0, width=0, act_channels=0,
                 top_left_row=-1, top_left_col=-1, border_id=-1,
                 halo_size_row=-1, halo_size_col=-1, comm_per_dim=1,
                 backend=None, top_left_row_output=-1, top_left_col_output=-1):

        self.global_row_index = global_row_index  # in comm_size (#CPUS/#GPUs)
        self.global_col_index = global_col_index
        self.comm_per_dim = comm_per_dim
        self.height = height
        self.width = width
        self.act_channels = act_channels

        self.top_left_col = top_left_col  # in px relative to global matrix(2d)
        self.top_left_row = top_left_row  # in px relative to global matrix(2d)
        self.top_left_col_output = top_left_col_output
        self.top_left_row_output = top_left_row_output

        self.local2d_size = height * width
        self.local_array_size = self.local2d_size * act_channels
        self.local_image_indices = []  # local_image relative to chunk

        self.halo_size_col = halo_size_col
        self.halo_size_row = halo_size_row

        self.border_id = border_id
        self.halo_ids = np.sort(gc.halo_dict[border_id])

        self.send_halos = dict()
        self.recv_halos = dict()
        self.neighbor_dims = dict()

        # print type(batch_size)
        if backend is None:
            self.local_image = np.empty(
                (batch_size, self.local_array_size), dtype='float32')
            # todo: can skip this if no defiltering layer (i.e., if not
            # pretraining)
            self.defiltering_local_image = np.empty_like(self.local_image)
        else:
            self.backend = backend
            self.local_image = backend.zeros(
                (batch_size, self.local_array_size), dtype='float32')
            # todo: can skip this if no defiltering layer (i.e., if not
            # pretraining)
            self.defiltering_local_image = backend.zeros(
                self.local_image.shape,  dtype='float32')

        self.batch_size = batch_size  # mini-batch size

        if halo_size_row != -1 and halo_size_col != -1:
            if border_id == -1:
                self.width_with_halos = self.width
                self.height_with_halos = self.height
            elif border_id == 0 or border_id == 2:
                self.width_with_halos = self.width + self.halo_size_col * 2
                self.height_with_halos = self.height + self.halo_size_row
            elif border_id == 1 or border_id == 3:
                self.width_with_halos = self.width + self.halo_size_col
                self.height_with_halos = self.height + self.halo_size_row * 2
            elif border_id in [4, 5, 6, 7]:
                self.width_with_halos = self.width + self.halo_size_col
                self.height_with_halos = self.height + self.halo_size_row

            self.local_array_size_with_halo = self.width_with_halos * \
                self.height_with_halos * self.act_channels
            if backend is None:
                self.chunk = np.empty(
                    (batch_size, self.local_array_size_with_halo),
                    dtype='float32')
                self.defiltering_chunk = np.empty_like(self.chunk)
            else:
                self.chunk = backend.zeros(
                    (batch_size, self.local_array_size_with_halo),
                    dtype='float32')
                self.defiltering_chunk = backend.zeros(
                    self.chunk.shape,
                    dtype='float32')

    def send_recv_halos(self, dbg=False):
        comm = MPI.COMM_WORLD
        comm.barrier()

        req = []
        # exchange and store neighbor dims
        for k in self.halo_ids:
            req.extend([None])
            neighbor_array_index = [
                gc.pos_offsets[k][0] + self.global_row_index,
                gc.pos_offsets[k][1] + self.global_col_index]
            neighbor_comm_index = neighbor_array_index[
                0] * self.comm_per_dim + neighbor_array_index[1]

            # if dbg:
            # print comm.rank, k, self.local_image.raw().shape,
            # self.send_halos[k].halo_indices
            comm.Sendrecv(sendbuf=self.local_image.take(
                self.send_halos[k].halo_indices, axis=1).raw(),
                dest=neighbor_comm_index, sendtag=0,
                recvbuf=self.recv_halos[k].halo_data.raw(),
                source=neighbor_comm_index,
                recvtag=0)
        comm.barrier()

    def send_recv_defiltering_layer_halos(self):
        comm = MPI.COMM_WORLD
        comm.barrier()
        # exchange and store neighbor dims
        for k in self.halo_ids:
            # k_reverse_dir = gc.send_dict[k]
            neighbor_array_index = [
                gc.pos_offsets[k][0] + self.global_row_index,
                gc.pos_offsets[k][1] + self.global_col_index]
            neighbor_comm_index = neighbor_array_index[
                0] * self.comm_per_dim + neighbor_array_index[1]
            # todo: make sure size is not zero before sending
            # /receiving (e.g. 2x2 filters)
            comm.Sendrecv(sendbuf=self.defiltering_chunk.take(
                self.recv_halos[k].halo_insert_indices,
                axis=1).raw().astype('float32'),
                dest=neighbor_comm_index,
                sendtag=0,
                recvbuf=self.send_halos[k].halo_data_defiltering.raw(),
                source=neighbor_comm_index,
                recvtag=0)

        comm.barrier()
        # print comm.rank, ' done with sendrecv_halos.'

    def compute_halo_insert_indices(self):
        # for defiltering layers, store the indices of halos in chunk

        neighbor_traverse_order1 = [gc.NORTHWEST, gc.NORTH, gc.NORTHEAST]
        neighbor_traverse_order2 = [gc.WEST, gc.CENTER, gc.EAST]
        neighbor_traverse_order3 = [gc.SOUTHWEST, gc.SOUTH, gc.SOUTHEAST]
        c_ptr = 0
        d_ptrs = dict()

        for n_id in range(9):
            d_ptrs[n_id] = 0

        for c in range(self.act_channels):
            # top halos
            for r in range(self.halo_size_row):
                for halo in neighbor_traverse_order1:
                    if halo not in self.halo_ids:
                        continue

                    if halo == gc.NORTH:
                        self.recv_halos[halo].halo_insert_indices.extend(
                            range(c_ptr, c_ptr + self.width))
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    else:
                        self.recv_halos[halo].halo_insert_indices.extend(
                            range(c_ptr, c_ptr + self.halo_size_col))
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

            # W/E halos and local image
            for r in range(self.height):
                for halo in neighbor_traverse_order2:
                    if halo == gc.CENTER:
                        self.local_image_indices.extend(
                            range(c_ptr, c_ptr + self.width))
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    elif halo in self.halo_ids:
                        self.recv_halos[halo].halo_insert_indices.extend(
                            range(c_ptr, c_ptr + self.halo_size_col))
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

            # bottom halos
            for r in range(self.halo_size_row):
                for halo in neighbor_traverse_order3:
                    if halo not in self.halo_ids:
                        continue

                    if halo == gc.SOUTH:
                        self.recv_halos[halo].halo_insert_indices.extend(
                            range(c_ptr, c_ptr + self.width))
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    else:
                        self.recv_halos[halo].halo_insert_indices.extend(
                            range(c_ptr, c_ptr + self.halo_size_col))
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

    def make_local_chunk_consistent(self):
        # for convergent layers

        neighbor_traverse_order1 = [gc.NORTHWEST, gc.NORTH, gc.NORTHEAST]
        neighbor_traverse_order2 = [gc.WEST, gc.CENTER, gc.EAST]
        neighbor_traverse_order3 = [gc.SOUTHWEST, gc.SOUTH, gc.SOUTHEAST]
        c_ptr = 0
        d_ptrs = dict()

        for n_id in range(9):
            d_ptrs[n_id] = 0

        MPI.COMM_WORLD.barrier()
        for c in range(self.act_channels):
            # top halos
            for r in range(self.halo_size_row):
                for halo in neighbor_traverse_order1:
                    if halo not in self.halo_ids:
                        continue

                    if halo == gc.NORTH:
                        self.chunk[:, c_ptr:c_ptr + self.width] = (
                            self.recv_halos[halo].halo_data[
                                :, d_ptrs[halo]:d_ptrs[halo] + self.width])
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    else:
                        self.chunk[:, c_ptr:c_ptr + self.halo_size_col] = (
                            self.recv_halos[halo].halo_data[
                                :, d_ptrs[halo]:d_ptrs[
                                    halo] + self.halo_size_col])
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

            # W/E halos and local image
            for r in range(self.height):
                for halo in neighbor_traverse_order2:
                    if halo == gc.CENTER:
                        # print 'cptr', c_ptr, self.width
                        self.chunk[:, c_ptr:c_ptr + self.width] = (
                            self.local_image[
                                :, d_ptrs[halo]:d_ptrs[halo] + self.width])
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    elif halo in self.halo_ids:
                        self.chunk[:, c_ptr:c_ptr + self.halo_size_col] = (
                            self.recv_halos[halo].halo_data[
                                :, d_ptrs[halo]:d_ptrs[
                                    halo] + self.halo_size_col])
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

            # bottom halos
            for r in range(self.halo_size_row):
                for halo in neighbor_traverse_order3:
                    if halo not in self.halo_ids:
                        continue

                    if halo == gc.SOUTH:
                        self.chunk[:, c_ptr:c_ptr + self.width] = (
                            self.recv_halos[halo].halo_data[
                                :, d_ptrs[halo]:d_ptrs[halo] + self.width])
                        c_ptr += self.width
                        d_ptrs[halo] += self.width
                    else:
                        self.chunk[:, c_ptr:c_ptr + self.halo_size_col] = (
                            self.recv_halos[halo].halo_data[
                                :, d_ptrs[halo]:d_ptrs[
                                    halo] + self.halo_size_col])
                        c_ptr += self.halo_size_col
                        d_ptrs[halo] += self.halo_size_col

        MPI.COMM_WORLD.barrier()

    # this could be useful if we need to sum products in the output feature map
    def make_defiltering_layer_consistent(self):
        d_ptrs = dict()

        for n_id in range(9):
            d_ptrs[n_id] = 0

        self.defiltering_local_image = self.defiltering_chunk.take(
            self.local_image_indices, axis=1)

        for halo in self.halo_ids:
            self.defiltering_local_image[
                :, self.send_halos[halo].halo_indices] += (
                self.send_halos[halo].halo_data_defiltering)
        MPI.COMM_WORLD.barrier()

    def set_halo_indices(self, neighbor_direction, neighbor_array_index):
        # todo: the indices extracted here are non-contiguous, will likely need
        # to speedup somehow

        halo_size_row = self.halo_size_row
        halo_size_col = self.halo_size_col

        recvhalo_indices = []
        sendhalo_indices = []

        nw, nh, nc, n_hsr, n_hsc = self.neighbor_dims[neighbor_direction]
        nlocal2d_size = nh * nw

        sw, sh, sc = self.width, self.height, self.act_channels
        slocal2d_size = sh * sw

        if nc != sc:
            print "warning nc != sc"
            print nc, ' ', sc

        for c in range(nc):
            if neighbor_direction == gc.NORTH:
                # neighbor sends their southern border rows
                recvhalo_indices.extend(
                    range((c + 1) * nlocal2d_size - halo_size_row * nw,
                          (c + 1) * nlocal2d_size))

                # src sends their northern border rows
                sendhalo_indices.extend(
                    range(c * slocal2d_size, c * slocal2d_size + n_hsr * sw))

            elif neighbor_direction == gc.EAST:
                # neighbor sends their western edge
                offset_in_channel = c * nlocal2d_size
                for r in range(nh):
                    col_offset = offset_in_channel + r * nw
                    recvhalo_indices.extend(
                        range(col_offset, col_offset + halo_size_col))

                # src sends their eastern edge
                offset_in_channel = c * slocal2d_size
                for r in range(sh):
                    col_offset = offset_in_channel + (r + 1) * sw
                    sendhalo_indices.extend(
                        range(col_offset - n_hsc, col_offset))

            elif neighbor_direction == gc.SOUTH:
                # neighbor sends their northern edge
                recvhalo_indices.extend(
                    range(c * nlocal2d_size,
                          c * nlocal2d_size + halo_size_row * nw))
                # src sends their southern edge
                sendhalo_indices.extend(
                    range((c + 1) * slocal2d_size - n_hsr * sw,
                          (c + 1) * slocal2d_size))

            elif neighbor_direction == gc.WEST:
                # neighbor sends their eastern edge
                offset_in_channel = c * nlocal2d_size
                for r in range(nh):
                    col_offset = offset_in_channel + (r + 1) * nw
                    recvhalo_indices.extend(
                        range(col_offset - halo_size_col, col_offset))

                # src sends their western edge
                offset_in_channel = c * slocal2d_size
                for r in range(sh):
                    col_offset = offset_in_channel + r * sw
                    sendhalo_indices.extend(
                        range(col_offset, col_offset + n_hsc))

            elif neighbor_direction == gc.NORTHEAST:
                # neighbor sends their SW edge
                offset_in_channel = c * nlocal2d_size
                for r in range(nh - halo_size_row, nh):
                    col_offset = offset_in_channel + r * nw
                    recvhalo_indices.extend(range(col_offset,
                                                  col_offset + halo_size_col))

                # src sends their NE edge
                offset_in_channel = c * slocal2d_size
                for r in range(n_hsr):
                    col_offset = offset_in_channel + (r + 1) * sw
                    sendhalo_indices.extend(range(col_offset - n_hsc,
                                                  col_offset))

            elif neighbor_direction == gc.SOUTHEAST:
                # neighbor sends their NW edge
                offset_in_channel = c * nlocal2d_size
                for r in range(halo_size_row):
                    col_offset = offset_in_channel + r * nw
                    recvhalo_indices.extend(range(col_offset,
                                                  col_offset + halo_size_col))

                # src sends their SE edge
                offset_in_channel = c * slocal2d_size
                for r in range(sh - n_hsr, sh):
                    col_offset = offset_in_channel + (r + 1) * sw
                    sendhalo_indices.extend(range(col_offset - n_hsc,
                                                  col_offset))

            elif neighbor_direction == gc.SOUTHWEST:
                # neighbor sends their NE edge
                offset_in_channel = c * nlocal2d_size
                for r in range(halo_size_row):
                    col_offset = offset_in_channel + (r + 1) * nw
                    recvhalo_indices.extend(
                        range(col_offset - halo_size_col,
                              col_offset))

                # src sends their SW edge
                offset_in_channel = c * slocal2d_size
                for r in range(sh - n_hsr, sh):
                    col_offset = offset_in_channel + r * sw
                    sendhalo_indices.extend(range(col_offset,
                                                  col_offset + n_hsc))

            elif neighbor_direction == gc.NORTHWEST:
                # neighbor sends their SE edge
                offset_in_channel = c * nlocal2d_size
                for r in range(nh - halo_size_row, nh):
                    col_offset = offset_in_channel + (r + 1) * nw
                    recvhalo_indices.extend(
                        range(col_offset - halo_size_col,
                              col_offset))

                # src sends their NW edge
                offset_in_channel = c * slocal2d_size
                for r in range(n_hsr):
                    col_offset = offset_in_channel + r * sw
                    sendhalo_indices.extend(range(col_offset,
                                                  col_offset + n_hsc))
                    # print MPI.COMM_WORLD.rank, 'NW halos:', slocal2d_size, r,
                    # sw, n_hsc, col_offset,sendhalo_indices

        # neighbor_direction is the direction of the target w.r.t src
        self.recv_halos[neighbor_direction] = RecvHalo(neighbor_array_index,
                                                       len(recvhalo_indices),
                                                       self.batch_size,
                                                       self.backend)
        self.send_halos[neighbor_direction] = SendHalo(neighbor_array_index,
                                                       sendhalo_indices,
                                                       self.batch_size,
                                                       self.backend)
        return len(recvhalo_indices)