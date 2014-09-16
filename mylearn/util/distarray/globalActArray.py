'''
Global View of the Data

'''

import numpy as np
from mpi4py import MPI
import localActArray as laa

import globalConsts as gc


def pprint(string, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        print(string)

# this defines the input layer for the DNN


class GlobalActArray(object):

    def computeBorder(self, i, j, commPerDim):
        if commPerDim == 1:
            return -1

        if i == 0:
            if j == 0:
                return gc.NORTHWEST
            elif j == commPerDim - 1:
                return gc.NORTHEAST
            else:
                return gc.NORTH
        elif i == commPerDim - 1:
            if j == 0:
                return gc.SOUTHWEST
            elif j == commPerDim - 1:
                return gc.SOUTHEAST
            else:
                return gc.SOUTH
        elif j == 0:
            return gc.WEST
        elif j == commPerDim - 1:
            return gc.EAST
        else:
            return gc.CENTER

    def initializeHalos(self):
        i, j = self.ccomm.Get_coords(self.ccomm.rank)
        for k in self.localActArray.haloIds:
            if k == -1:
                return

            neighborActArrayIndex = [
                gc.posOffsets[k][0] + i, gc.posOffsets[k][1] + j]
            # figure out send halo indices and recv halo size
            self.localActArray.setHaloIndices(k, neighborActArrayIndex)
            # borderId order

            if self.printDebug:
                print 'Setting up halos to send for block: ', i, ',', j, ' ', \
                    gc.haloDirNames[k], 'haloIndices: ', \
                    self.localActArray.sendHalos[k].haloIndices

        self.localActArray.computeHaloInsertIndices()

    def __init__(self, batchSize=1, actSize=8, actChannels=1,
                 filterSize=2, backend=None):
        commSize = MPI.COMM_WORLD.size
        self.actSize = actSize
        self.actChannels = actChannels
        self.commSize = commSize
        self.filterSize = filterSize
        self.printDebug = False

        commPerDim = np.int(np.sqrt(commSize))
        self.commPerDim = commPerDim

        topLeftRow = 0  # not used right now

        comm = MPI.COMM_WORLD

        # todo comm cart needs to only be created the first time (first layer)
        pprint("Creating a %d x %d processor grid..." %
               (commPerDim, commPerDim))
        # if layerId ==0:
        self.ccomm = comm.Create_cart((commPerDim, commPerDim))

        i, j = self.ccomm.Get_coords(self.ccomm.rank)
        # i = comm.rank // commPerDim
        # j = comm.rank % commPerDim

        # initialize halo dimensions
        h = actSize // commPerDim + (actSize % commPerDim > i)
        haloSizeRow = (filterSize - 1) // 2 + (
            (filterSize - 1) % 2 > i)  # dimensions along up/down axis
        w = actSize // commPerDim + (actSize % commPerDim > j)
        haloSizeCol = (filterSize - 1) // 2 + (
            (filterSize - 1) % 2 > j)  # dimensions along left/right axis
        borderId = self.computeBorder(i, j, commPerDim)
        topLeftRow = h * i
        topLeftCol = w * j
        self.localActArray = laa.LocalActArray(
            batchSize, i, j, h, w, actChannels, topLeftRow, topLeftCol,
            borderId, haloSizeRow, haloSizeCol, commPerDim, backend)
        # self.localImage = self.localActArray.localImage

        # synchronize everyone here
        comm.barrier()

        # exchange and store neighbor dims
        for k in self.localActArray.haloIds:
            neighborActArrayIndex = [
                gc.posOffsets[k][0] + i, gc.posOffsets[k][1] + j]
            neighborCommIndex = neighborActArrayIndex[
                0] * self.commPerDim + neighborActArrayIndex[1]
            # sendTag = i*100 + j*10 + k
            sendDims = [
                self.localActArray.width,
                self.localActArray.height,
                self.localActArray.actChannels,
                self.localActArray.haloSizeRow,
                self.localActArray.haloSizeCol]
            comm.Isend(np.asarray(sendDims, dtype='int32'), neighborCommIndex)
            # recvTag = neighborActArrayIndex[0]*100 +
            # neighborActArrayIndex[1]*10 + k
            self.localActArray.neighborDims[k] = np.empty((5), dtype='int32')
            comm.Irecv(self.localActArray.neighborDims[k], neighborCommIndex)

        comm.barrier()

        self.initializeHalos()
