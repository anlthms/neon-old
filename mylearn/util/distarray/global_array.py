'''
Global View of the Data

'''

import numpy as np
from mpi4py import MPI
import local_array as laa

import gdist_consts as gc


def pprint(string, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        print(string)

# this defines the input layer for the DNN


class GlobalArray(object):

    def compute_border(self, i, j, comm_per_dim):
        if comm_per_dim == 1:
            return -1

        if i == 0:
            if j == 0:
                return gc.NORTHWEST
            elif j == comm_per_dim - 1:
                return gc.NORTHEAST
            else:
                return gc.NORTH
        elif i == comm_per_dim - 1:
            if j == 0:
                return gc.SOUTHWEST
            elif j == comm_per_dim - 1:
                return gc.SOUTHEAST
            else:
                return gc.SOUTH
        elif j == 0:
            return gc.WEST
        elif j == comm_per_dim - 1:
            return gc.EAST
        else:
            return gc.CENTER

    def initialize_halos(self):
        i, j = self.ccomm.Get_coords(self.ccomm.rank)
        for k in self.local_array.halo_ids:
            if k == -1:
                return

            neighbor_act_array_index = [
                gc.pos_offsets[k][0] + i, gc.pos_offsets[k][1] + j]
            # figure out send halo indices and recv halo size
            self.local_array.set_halo_indices(k, neighbor_act_array_index)
            # border_id order

            if self.print_debug:
                print 'Setting up halos to send for block: ', i, ',', j, ' ', \
                    gc.halo_dir_names[k], 'halo_indices: ', \
                    self.local_array.send_halos[k].halo_indices

        self.local_array.compute_halo_insert_indices()

    def __init__(self, batch_size=1, act_size_height=8, act_size_width=8,
                 act_channels=1, filter_size=2, backend=None,
                 create_comm=False, ccomm=None, h=-1, w=-1,
                 lcn_layer_flag=False):
        comm_size = MPI.COMM_WORLD.size
        self.act_size_width = act_size_width
        self.act_size_height = act_size_height
        self.act_channels = act_channels
        self.comm_size = comm_size
        self.filter_size = filter_size
        self.print_debug = False

        comm_per_dim = np.int(np.sqrt(comm_size))
        self.comm_per_dim = comm_per_dim

        top_left_row = 0  # not used right now

        comm = MPI.COMM_WORLD

        # comm cart needs to only be created the first time (first layer)
        if create_comm:
            pprint("Creating a %d x %d processor grid..." %
                   (comm_per_dim, comm_per_dim))
            self.ccomm = comm.Create_cart((comm_per_dim, comm_per_dim))
            i, j = self.ccomm.Get_coords(self.ccomm.rank)
            h = act_size_height // comm_per_dim + \
                (act_size_height % comm_per_dim > i)
            w = act_size_width // comm_per_dim + \
                (act_size_width % comm_per_dim > j)
        else:
            self.ccomm = ccomm
        i, j = self.ccomm.Get_coords(self.ccomm.rank)
        # i = comm.rank // comm_per_dim
        # j = comm.rank % comm_per_dim

        # initialize halo dimensions
        halo_size_row = (filter_size - 1) // 2 + (
            (filter_size - 1) % 2 > i)  # dimensions along up/down axis
        halo_size_col = (filter_size - 1) // 2 + (
            (filter_size - 1) % 2 > j)  # dimensions along left/right axis
        self.border_id = self.compute_border(i, j, comm_per_dim)

        top_left_row = 0
        top_left_col = 0
        top_left_row_output = 0
        top_left_col_output = 0

        for i_iter in range(i):
            h_iter = act_size_height // comm_per_dim + \
                (act_size_height % comm_per_dim > i_iter)
            hsr_iter = (filter_size - 1) // 2 + (
                (filter_size - 1) % 2 > i_iter)  # dims along up/down axis
            top_left_row += h_iter
            if lcn_layer_flag:
                top_left_row_output += h_iter
            else:
                top_left_row_output += h_iter + hsr_iter - filter_size + 1

        for j_iter in range(j):
            w_iter = act_size_width // comm_per_dim + \
                (act_size_width % comm_per_dim > j_iter)
            hsc_iter = (filter_size - 1) // 2 + (
                (filter_size - 1) % 2 > j_iter)  # dims along left/right axis
            top_left_col += w_iter
            if lcn_layer_flag:
                top_left_col_output += w_iter
            else:
                top_left_col_output += w_iter + hsc_iter - filter_size + 1

        self.local_array = laa.LocalArray(
            batch_size, i, j, h, w, act_channels, top_left_row, top_left_col,
            self.border_id, halo_size_row, halo_size_col, comm_per_dim,
            backend, top_left_row_output, top_left_col_output)

        # synchronize everyone here
        comm.barrier()

        # exchange and store neighbor dims
        for k in self.local_array.halo_ids:
            neighbor_act_array_index = [
                gc.pos_offsets[k][0] + i, gc.pos_offsets[k][1] + j]
            neighbor_comm_index = neighbor_act_array_index[
                0] * self.comm_per_dim + neighbor_act_array_index[1]
            send_dims = [
                self.local_array.width,
                self.local_array.height,
                self.local_array.act_channels,
                self.local_array.halo_size_row,
                self.local_array.halo_size_col]
            comm.Isend(
                np.asarray(send_dims, dtype='int32'), neighbor_comm_index)
            self.local_array.neighbor_dims[k] = np.empty((5), dtype='int32')
            comm.Irecv(self.local_array.neighbor_dims[k], neighbor_comm_index)

        comm.barrier()

        self.initialize_halos()
