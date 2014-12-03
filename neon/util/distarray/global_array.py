# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
'''
Global View of the Data

'''

import logging
import numpy as np

import gdist_consts as gc
import local_array as laa
from neon.util.compat import MPI_INSTALLED

logger = logging.getLogger(__name__)

if MPI_INSTALLED:
    from mpi4py import MPI
else:
    logger.error("mpi4py not installed")


class GlobalArray():

    create_comm = True
    ccomm = None

    def make_fprop_view(self, input_data):
        self.local_array.make_fprop_view(input_data)

    def get_fprop_view(self, input_data):
        return self.local_array.get_fprop_view(input_data)

    def make_bprop_view(self, input_data):
        self.local_array.make_bprop_view(input_data)

    def get_bprop_view(self, input_data):
        return self.local_array.get_bprop_view(input_data)

    def get_local_acts(self):
        return self.local_array.get_local_acts()

    def compute_border(self, i, j, comm_per_dim):
        if comm_per_dim == 1:
            return gc.SINGLE

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
        i, j = self.__class__.ccomm.Get_coords(self.__class__.ccomm.rank)
        for k in self.local_array.halo_ids:
            if k == -1:
                return

            neighbor_act_array_index = [
                gc.pos_offsets[k][0] + i, gc.pos_offsets[k][1] + j]
            # figure out send halo indices and recv halo size
            self.local_array.set_halo_indices(k, neighbor_act_array_index)
            # border_id order

            if self.print_debug:
                logger.debug('Setting up halos to send for block: %d,%d %s'
                             'halo_indices: %s', i, j, gc.halo_dir_names[k],
                             str(self.local_array.send_halos[k].halo_indices))

        self.local_array.compute_halo_insert_indices()

    def __init__(self, cur_layer, h=-1, w=-1,
                 lcn_layer_flag=False, tensor_name='output'):
        self.batch_size = cur_layer.batch_size
        self.act_size_height = cur_layer.ifmshape[0]
        self.act_size_width = cur_layer.ifmshape[1]
        self.act_channels = cur_layer.nifm
        self.filter_size = cur_layer.fwidth
        self.backend = cur_layer.backend

        comm = MPI.COMM_WORLD
        comm_size = comm.size
        self.comm_size = comm_size
        self.print_debug = False

        comm_per_dim = np.int(np.sqrt(comm_size))
        self.comm_per_dim = comm_per_dim

        act_size_height = self.act_size_height
        act_size_width = self.act_size_width
        filter_size = self.filter_size
        # comm cart needs to only be created the first time (first layer)
        if self.__class__.create_comm:
            logger.info("MPI proc: %d: Creating a %d x %d processor grid..." %
                        (comm.rank, comm_per_dim, comm_per_dim))
            self.__class__.ccomm = comm.Create_cart(
                (comm_per_dim, comm_per_dim))
            i, j = self.__class__.ccomm.Get_coords(self.__class__.ccomm.rank)
            h = (act_size_height // comm_per_dim +
                 (act_size_height % comm_per_dim > i))
            w = (act_size_width // comm_per_dim +
                 (act_size_width % comm_per_dim > j))
            self.__class__.create_comm = False
        i, j = self.__class__.ccomm.Get_coords(self.__class__.ccomm.rank)
        self.border_id = self.compute_border(i, j, comm_per_dim)

        # initialize halo dimensions
        west_halo = 0
        east_halo = 0
        carry = 0
        north_halo = 0
        south_halo = 0
        top_left_row = 0
        top_left_col = 0
        top_left_row_output = 0
        top_left_col_output = 0

        if lcn_layer_flag:  # if padded, nout = nin
            if cur_layer.stride > 1:
                raise ValueError('LCN layer stride > 1 not supported for dist')
            # West/East halos
            pad_width_right = (self.filter_size - 1) // 2
            pad_width_left = (self.filter_size - 1) - pad_width_right
            for j_iter in range(j + 1):
                # local halo size width
                lhsw = self.filter_size - 1
                # if west border then halo - left_padding is added to other
                # side
                if j_iter == 0:
                    east_halo = lhsw - pad_width_left
                elif j_iter == comm_per_dim:
                    west_halo = carry
                    east_halo = 0
                # otherwise split across east and west border
                else:
                    west_halo = carry
                    east_halo = lhsw - west_halo
                # compute relative location of local array
                if j_iter < j:
                    w_iter = act_size_width // comm_per_dim + (
                        act_size_width % comm_per_dim > j_iter)
                    top_left_col += w_iter
                    top_left_col_output += w_iter
                carry = (self.filter_size - 1) - east_halo
            hsc_west = west_halo
            hsc_east = east_halo

            pad_height_bottom = (self.filter_size - 1) // 2
            pad_height_top = (self.filter_size - 1) - pad_height_bottom
            # north/south halos
            carry = 0
            for i_iter in range(i + 1):
                # local halo size height
                lhsh = self.filter_size - 1
                # if north border then halo - top pad is added to other side
                if i_iter == 0:
                    south_halo = lhsh - pad_height_top
                elif i_iter == comm_per_dim:
                    north_halo = carry
                    south_halo = 0
                # otherwise split across north and south border
                else:
                    north_halo = carry
                    south_halo = lhsh - north_halo
                # compute relative location of local array
                if i_iter < i:
                    h_iter = act_size_height // comm_per_dim + (
                        act_size_height % comm_per_dim > i_iter)
                    top_left_row += h_iter
                    top_left_row_output += h_iter
                carry = (self.filter_size - 1) - south_halo
            hsr_north = north_halo
            hsr_south = south_halo

        else:  # no padding
            # West/East halos
            # total next layer input width
            tnlw = ((self.act_size_width - self.filter_size) /
                    cur_layer.stride) + 1
            for j_iter in range(j + 1):
                w_iter = act_size_width // comm_per_dim + (
                    act_size_width % comm_per_dim > j_iter)
                # local next layer width
                lnlw = tnlw // comm_per_dim + (tnlw % comm_per_dim > j_iter)
                # local halo size width
                lhsw = ((lnlw - 1) * cur_layer.stride + self.filter_size -
                        w_iter)
                # if west border then entire halo is added to other side
                if j_iter == 0:
                    east_halo = lhsw
                # otherwise split across east and west border
                else:
                    west_halo = carry
                    east_halo = lhsw - west_halo
                # compute relative location of local array
                if j_iter < j:
                    top_left_col += w_iter
                    top_left_col_output += (w_iter + west_halo +
                                            east_halo - filter_size) / (
                        cur_layer.stride) + 1
                if cur_layer.stride > 1:
                    # assuming that stride is divisible by w_iter for now
                    if w_iter % cur_layer.stride != 0:
                        raise ValueError('w_iter has to divide stride')
                    carry = 0
                else:
                    carry = (self.filter_size - 1) - east_halo
            hsc_west = west_halo
            hsc_east = east_halo

            # north/south halos
            carry = 0
            # total next layer input height
            tnlh = ((self.act_size_height - self.filter_size) /
                    cur_layer.stride) + 1
            for i_iter in range(i + 1):
                h_iter = act_size_height // comm_per_dim + (
                    act_size_height % comm_per_dim > i_iter)
                # local next layer height
                lnlh = tnlh // comm_per_dim + (tnlh % comm_per_dim > i_iter)
                # local halo size height
                lhsh = ((lnlh - 1) * cur_layer.stride + self.filter_size -
                        h_iter)
                # if north border then entire halo is added to other side
                if i_iter == 0:
                    south_halo = lhsh
                # otherwise split across north and south border
                else:
                    north_halo = carry
                    south_halo = lhsh - north_halo
                # compute relative location of local array
                if i_iter < i:
                    top_left_row += h_iter
                    top_left_row_output += (h_iter + north_halo +
                                            south_halo - filter_size) / (
                        cur_layer.stride) + 1
                if cur_layer.stride > 1:
                    if h_iter % cur_layer.stride != 0:
                        raise ValueError('h_iter has to divide stride')
                    # assuming that stride is divisible by h_iter for now
                    carry = 0
                else:
                    carry = (self.filter_size - 1) - south_halo
            hsr_north = north_halo
            hsr_south = south_halo

        self.local_array = laa.LocalArray(
            self.batch_size, i, j, h, w, self.act_channels, top_left_row,
            top_left_col, self.border_id, hsr_north, hsr_south, hsc_west,
            hsc_east, comm_per_dim, self.backend, top_left_row_output,
            top_left_col_output)

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
                self.local_array.hsr_north,
                self.local_array.hsr_south,
                self.local_array.hsc_west,
                self.local_array.hsc_east]
            comm.Isend(
                np.asarray(send_dims, dtype='int32'), neighbor_comm_index)
            self.local_array.neighbor_dims[k] = np.empty(
                (len(send_dims)), dtype='int32')
            comm.Irecv(self.local_array.neighbor_dims[k], neighbor_comm_index)

        comm.barrier()

        self.initialize_halos()
