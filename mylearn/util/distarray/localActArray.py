'''
Local View of the Data

'''


import numpy as np
from mpi4py import MPI
import globalConsts as gc
import pickle 

def pprint(string, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        print(string)

class RecvHalo(object):
    def __init__(self, neighborActArrayIndex, haloSize, mbSize=1, backend=None):
        self.neighborActArrayIndex = neighborActArrayIndex
        self.haloSize = haloSize
        self.haloInsertIndices = []
        #initialize a temporary buffer space if we need it
        if backend == None:
            self.haloData = np.empty((mbSize,haloSize), dtype='float32')
        else:
            self.haloData = backend.zeros((mbSize,haloSize), dtype='float32')

class SendHalo(object):
    def __init__(self, targetActArrayIndex, haloIndices, mbSize=1, backend=None):
        self.targetActArrayIndex = targetActArrayIndex
        self.haloIndices = haloIndices
        #space to store received defiltered data
        if backend == None:
            self.haloDataDefiltering = np.empty((mbSize, len(haloIndices)), dtype='float32')
        else:
            self.haloDataDefiltering = backend.zeros((mbSize, len(haloIndices)), dtype='float32')

class LocalActArray(object):        
    
    def __init__(self, batchSize=None, globalRowIndex=-1, globalColIndex=-1, height=0, width=0, actChannels=0, topLeftRow=-1, topLeftCol=-1, 
                    borderId=-1, haloSizeRow=-1, haloSizeCol=-1, commPerDim=1, backend=None):
        
        self.globalRowIndex = globalRowIndex #in commSize (#CPUS or #GPUs)
        self.globalColIndex = globalColIndex
        self.commPerDim = commPerDim
        self.height = height
        self.width = width
        self.actChannels = actChannels        
        self.topLeftCol = topLeftCol #in px relative to global matrix in 2d 
        self.topLeftRow = topLeftRow #in px relative to global matrix in 2d
        self.local2dSize = height*width
        self.localArraySize = self.local2dSize*actChannels
        
        self.haloSizeCol = haloSizeCol
        self.haloSizeRow = haloSizeRow
        
        self.borderId = borderId
        self.haloIds = np.sort(gc.haloDict[borderId])            
        
        self.sendHalos = dict()
        self.recvHalos = dict()
        self.neighborDims = dict()
        
        #print type(batchSize)
        if backend == None:
            self.localImage =  np.empty((batchSize, self.localArraySize), dtype='float32')
            #todo: can skip this if no defiltering layer (i.e., if not pretraining)
            self.localImageDefiltering = np.empty_like(self.localImage)
        else:
            self.backend = backend
            self.localImage =  backend.zeros((batchSize, self.localArraySize), dtype='float32')
            #todo: can skip this if no defiltering layer (i.e., if not pretraining)
            self.localImageDefiltering = backend.zeros(self.localImage.shape,  dtype='float32')
        
        self.batchSize = batchSize #mini-batch size
        
        if haloSizeRow!=-1 and haloSizeCol!=-1:
            if borderId == -1:
                self.widthWithHalos = self.width
                self.heightWithHalos = self.height
            elif borderId == 0 or borderId == 2:
                self.widthWithHalos = self.width + self.haloSizeCol*2
                self.heightWithHalos = self.height + self.haloSizeRow
            elif borderId ==1 or borderId == 3:
                self.widthWithHalos = self.width + self.haloSizeCol
                self.heightWithHalos = self.height + self.haloSizeRow*2
            elif borderId in [4,5,6,7]:
                self.widthWithHalos = self.width + self.haloSizeCol
                self.heightWithHalos = self.height + self.haloSizeRow
            
            self.localArraySizeWithHalo = self.widthWithHalos * self.heightWithHalos * self.actChannels 
            if backend == None:
                self.chunk = np.empty((batchSize, self.localArraySizeWithHalo), dtype='float32')
            else:
                self.chunk = backend.zeros((batchSize, self.localArraySizeWithHalo), dtype='float32')

    # def allocateLocalMemory(self): #call after initializeHalos has been called for layer 1
    #     #for images replace this with loading a chunk of the image from local disk
    #     if self.testIndex==gc.SMALL_TEST:
    #         self.localImage = (self.globalRowIndex*self.commPerDim + self.globalColIndex)*np.ones((self.localArraySize)).astype(np.float32)
    #     elif self.testIndex==gc.MNIST:
    #         #load local MNIST quadrant
    #         mnist_img = pickle.load(open('/home/users/arjun/localgit/distmatrix/MNIST/trainData_' + str(MPI.COMM_WORLD.rank)))
    #         self.localImage = mnist_img[2,:] #no mini-batch for now
        
    def sendRecvHalos(self):
        comm = MPI.COMM_WORLD
        comm.barrier()
        req = [None, None]
        #exchange and store neighbor dims 
        for k in self.haloIds:
            neighborActArrayIndex = [gc.posOffsets[k][0] + self.globalRowIndex, gc.posOffsets[k][1]+ self.globalColIndex]     
            neighborCommIndex = neighborActArrayIndex[0]*self.commPerDim + neighborActArrayIndex[1]
            
            req[0] = comm.Isend(self.localImage.take(self.sendHalos[k].haloIndices, axis=1).raw(), neighborCommIndex)
            req[1] = comm.Irecv(self.recvHalos[k].haloData.raw(), neighborCommIndex)
            #print comm.rank, ' dir:', k, 's: ', self.localImage[self.sendHalos[k].haloIndices],  self.sendHalos[k].haloIndices
            
            #wait for requests to complete: todo: is this correct? don't need it to be synchronized across all nodes, just send/recv pairs 
            #req[0].Waitall(req)
              
        comm.barrier()
        #print comm.rank, ' done with sendRecvHalos.'  
        
        # for k in self.haloIds:
        #     print comm.rank, ' dir:', k, 'r: ', self.recvHalos[k].haloData
            
    def sendRecvDefilteringLayerHalos(self):
        comm = MPI.COMM_WORLD
        comm.barrier()
        req = [None, None] #,None, None,None, None,None, None,None, None,None, None,None, None,None, None]
        #reqId = 0
        #exchange and store neighbor dims 
        for k in self.haloIds:
            #k_reverse_dir = gc.sendDict[k]
            neighborActArrayIndex = [gc.posOffsets[k][0] + self.globalRowIndex, gc.posOffsets[k][1]+ self.globalColIndex]     
            neighborCommIndex = neighborActArrayIndex[0]*self.commPerDim + neighborActArrayIndex[1]
            #print comm.rank, ' dir:', k, k_reverse_dir, neighborActArrayIndex, neighborCommIndex
            #print neighborCommIndex, comm.rank, k,  'sendsize: ', self.localImageDefiltering.take(self.recvHalos[k].haloInsertIndices, axis=1).raw().shape
            #print neighborCommIndex, comm.rank, k, 'recvsize: ', self.sendHalos[k].haloDataDefiltering.raw().shape
            #print neighborCommIndex, comm.rank, k, self.recvHalos[k].haloInsertIndices
            #print comm.rank, neighborCommIndex, k, 'send', self.localImageDefiltering.take(self.recvHalos[k].haloInsertIndices, axis=1).raw().shape
            #print neighborCommIndex, comm.rank, k, 'recv', self.sendHalos[k].haloDataDefiltering.raw().shape
            #sndDebugBuffer = np.ones(self.localImageDefiltering.take(self.recvHalos[k].haloInsertIndices, axis=1).raw().shape, dtype='float32')
            #rcvDebugBuffer = np.ones(self.sendHalos[k].haloDataDefiltering.raw().shape, dtype='float32')
            #todo: make sure size is not zero before sending /receiving (e.g. 2x2 filters)
            #todo: not sure why this isn't working
            req[0] = comm.Isend(self.localImageDefiltering.take(self.recvHalos[k].haloInsertIndices, axis=1).raw().astype('float32'), neighborCommIndex)
            #req[reqId] = comm.Isend(sndDebugBuffer, neighborCommIndex)
            #reqId += 1
            req[1] = comm.Irecv(self.sendHalos[k].haloDataDefiltering.raw(), neighborCommIndex)
            #req[reqId] = comm.Irecv(rcvDebugBuffer, neighborCommIndex)
            #reqId += 1
        
        comm.barrier()
        #print comm.rank, ' done with sendRecvHalos.'  
        

    
    def computeHaloInsertIndices(self):
        #for defiltering layers, store the indices of halos in chunk
        
        neighborTraverseOrder1 = [7, 0, 4]
        neighborTraverseOrder2 = [3, 8, 1]
        neighborTraverseOrder3 = [6, 2, 5]
        cPtr = 0 
        dPtrs = dict()
        
        for nId in range(9):
            dPtrs[nId] = 0
        
        for c in range(self.actChannels):
            #top halos    
            for r in range(self.haloSizeRow):
                for halo in neighborTraverseOrder1:
                    if halo not in self.haloIds:
                        continue
                    
                    if halo == 0:
                        #self.chunk[:,cPtr:cPtr+self.width] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        self.recvHalos[halo].haloInsertIndices.extend(range(cPtr, cPtr+self.width))
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    else:
                        #self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        self.recvHalos[halo].haloInsertIndices.extend(range(cPtr, cPtr+self.haloSizeCol))
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol
                    
            #W/E halos and local image
            for r in range(self.height):
                for halo in neighborTraverseOrder2:
                    if halo == 8:
                        #self.chunk[:,cPtr:cPtr+self.width] = self.localImage[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    elif halo in self.haloIds:
                        #self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        self.recvHalos[halo].haloInsertIndices.extend(range(cPtr, cPtr+self.haloSizeCol))
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol
                    
            #bottom halos    
            for r in range(self.haloSizeRow):
                for halo in neighborTraverseOrder3:
                    if halo not in self.haloIds:
                        continue
                    
                    if halo == 2:
                        #self.chunk[:,cPtr:cPtr+self.width] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        self.recvHalos[halo].haloInsertIndices.extend(range(cPtr, cPtr+self.width))
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    else:
                        #self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        self.recvHalos[halo].haloInsertIndices.extend(range(cPtr, cPtr+self.haloSizeCol))
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol        

    def makeLocalChunkConsistent(self):
        #for convergent layers
        
        neighborTraverseOrder1 = [7, 0, 4]
        neighborTraverseOrder2 = [3, 8, 1]
        neighborTraverseOrder3 = [6, 2, 5]
        cPtr = 0 
        dPtrs = dict()
        
        for nId in range(9):
            dPtrs[nId] = 0
        
        for c in range(self.actChannels):
            #top halos    
            for r in range(self.haloSizeRow):
                for halo in neighborTraverseOrder1:
                    if halo not in self.haloIds:
                        continue
                    
                    if halo == 0:
                        self.chunk[:,cPtr:cPtr+self.width] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    else:
                        #print self.borderId, self.globalRowIndex, self.globalColIndex, r, halo, cPtr, self.haloSizeCol, dPtrs[halo], self.chunk #, self.recvHalos[halo].haloData
                        self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol
                    
            #W/E halos and local image
            for r in range(self.height):
                for halo in neighborTraverseOrder2:
                    if halo == 8:
                        #print 'cptr', cPtr, self.width
                        self.chunk[:,cPtr:cPtr+self.width] = self.localImage[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    elif halo in self.haloIds:
                        self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol
                    
            #bottom halos    
            for r in range(self.haloSizeRow):
                for halo in neighborTraverseOrder3:
                    if halo not in self.haloIds:
                        continue
                    
                    if halo == 2:
                        self.chunk[:,cPtr:cPtr+self.width] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.width]
                        cPtr += self.width
                        dPtrs[halo] += self.width
                    else:
                        self.chunk[:,cPtr:cPtr+self.haloSizeCol] = self.recvHalos[halo].haloData[:,dPtrs[halo]:dPtrs[halo]+self.haloSizeCol]
                        cPtr += self.haloSizeCol
                        dPtrs[halo] += self.haloSizeCol        
        
        MPI.COMM_WORLD.barrier()
        # if self.heightWithHalos>10:
        #     print np.reshape(self.chunk, [self.actChannels, self.heightWithHalos, self.widthWithHalos])[:,10], ' :local array: ', MPI.COMM_WORLD.rank
        # else:            
        #     print np.reshape(self.chunk, [self.actChannels, self.heightWithHalos, self.widthWithHalos]), ' :local array: ', MPI.COMM_WORLD.rank
        
    #this could be useful if we need to sum products in the output feature map
    def makeDefilteringLayerConsistent(self):
        #for divergent layers
        #MPI.COMM_WORLD.barrier()
        neighborTraverseOrder1 = [7, 0, 4]
        neighborTraverseOrder2 = [3, 8, 1]
        neighborTraverseOrder3 = [6, 2, 5]
        cPtr = 0 
        dPtrs = dict()
    
        for nId in range(9):
            dPtrs[nId] = 0
        
        #self.localImageDefiltering = localImageDefiltering
        for halo in self.haloIds:
            #print 'makeDefilteringLayerConsistent', MPI.COMM_WORLD.rank, halo
            self.localImageDefiltering[:,self.sendHalos[halo].haloIndices] += self.sendHalos[halo].haloDataDefiltering
        MPI.COMM_WORLD.barrier()

                
    def setHaloIndices(self, neighborDirection, neighborActArrayIndex ):
        #todo: the indices extracted here are non-contiguous, will likely need to speedup somehow

        haloSizeRow = self.haloSizeRow
        haloSizeCol = self.haloSizeCol

        recvHaloIndices = []
        sendHaloIndices = []

        nw, nh, nc, nHSR, nHSC = self.neighborDims[neighborDirection]
        nlocal2dSize = nh*nw

        sw, sh, sc = self.width, self.height, self.actChannels
        slocal2dSize = sh*sw

        if nc != sc:
            print "warning nc != sc"
            print nc , ' ', sc

        for c in range(nc):        
            if neighborDirection==0: 
                #neighbor sends their southern border rows
                recvHaloIndices.extend(range( (c+1)*nlocal2dSize - haloSizeRow*nw, (c+1)*nlocal2dSize ))
                
                # if layerType == gc.DIVERGENT:
                #     self.topLeftRow -= haloSizeRow
                
                #src sends their northern border rows
                sendHaloIndices.extend(range( c*slocal2dSize, c*slocal2dSize + nHSR*sw))

            elif neighborDirection ==1: 
                #neighbor sends their western edge
                offsetInChannel = c*nlocal2dSize 
                for r in range(nh):
                    offsetInRow = offsetInChannel + r*nw
                    recvHaloIndices.extend(range( offsetInRow , offsetInRow + haloSizeCol ))
                
                #src sends their eastern edge
                offsetInChannel = c*slocal2dSize 
                for r in range(sh):
                    offsetInRow = offsetInChannel + (r+1)*sw 
                    sendHaloIndices.extend(range( offsetInRow - nHSC, offsetInRow  ))

            elif neighborDirection == 2: 
                #neighbor sends their northern edge
                recvHaloIndices.extend(range( c*nlocal2dSize, c*nlocal2dSize + haloSizeRow*nw))
                #src sends their southern edge
                sendHaloIndices.extend(range( (c+1)*slocal2dSize - nHSR*sw, (c+1)*slocal2dSize ))

            elif neighborDirection == 3: 
                #neighbor sends their eastern edge
                offsetInChannel = c*nlocal2dSize 
                for r in range(nh):
                    offsetInRow = offsetInChannel + (r+1)*nw 
                    recvHaloIndices.extend(range( offsetInRow - haloSizeCol, offsetInRow  ))

                # if layerType == gc.DIVERGENT:                
                #     self.topLeftCol -= haloSizeCol 
                
                #src sends their western edge
                offsetInChannel = c*slocal2dSize 
                for r in range(sh):
                    offsetInRow = offsetInChannel + r*sw
                    sendHaloIndices.extend(range( offsetInRow , offsetInRow + nHSC ))

            elif neighborDirection == 4: 
                #neighbor sends their SW edge
                offsetInChannel = c*nlocal2dSize                 
                for r in range(nh - haloSizeRow, nh):
                    offsetInRow = offsetInChannel + r*nw
                    recvHaloIndices.extend(range( offsetInRow , offsetInRow + haloSizeCol ))
                
                #src sends their NE edge
                offsetInChannel = c*slocal2dSize                 
                for r in range(nHSR):
                    offsetInRow = offsetInChannel + (r+1)*sw
                    sendHaloIndices.extend(range( offsetInRow - nHSC , offsetInRow ))

            elif neighborDirection == 5: 
                #neighbor sends their NW edge
                offsetInChannel = c*nlocal2dSize                 
                for r in range(haloSizeRow):
                    offsetInRow = offsetInChannel + r*nw
                    recvHaloIndices.extend(range( offsetInRow , offsetInRow + haloSizeCol ))

                #src sends their SE edge
                offsetInChannel = c*slocal2dSize                 
                for r in range(sh - nHSR, sh):
                    offsetInRow = offsetInChannel + (r+1)*sw
                    sendHaloIndices.extend(range( offsetInRow - nHSC , offsetInRow ))

            elif neighborDirection == 6: 
                #neighbor sends their NE edge
                offsetInChannel = c*nlocal2dSize                 
                for r in range(haloSizeRow):
                    offsetInRow = offsetInChannel + (r+1)*nw
                    recvHaloIndices.extend(range( offsetInRow - haloSizeCol , offsetInRow ))

                #src sends their SW edge
                offsetInChannel = c*slocal2dSize                 
                for r in range(sh - nHSR, sh):
                    offsetInRow = offsetInChannel + r*sw
                    sendHaloIndices.extend(range( offsetInRow , offsetInRow + nHSC ))

            elif neighborDirection == 7: 
                #neighbor sends their SE edge
                offsetInChannel = c*nlocal2dSize                 
                for r in range(nh - haloSizeRow, nh):
                    offsetInRow = offsetInChannel + (r+1)*nw
                    recvHaloIndices.extend(range( offsetInRow - haloSizeCol , offsetInRow ))

                #src sends their NW edge    
                offsetInChannel = c*nlocal2dSize                 
                for r in range(nHSR):
                    offsetInRow = offsetInChannel + r*sw
                    sendHaloIndices.extend(range( offsetInRow , offsetInRow + nHSC ))

        #neighborDirection is the direction of the target w.r.t src
        self.recvHalos[neighborDirection] = RecvHalo(neighborActArrayIndex, len(recvHaloIndices), self.batchSize, self.backend)
        #neighborLocalActArray.sendHalos[neighborDirection] = SendHalo([self.globalRowIndex, self.globalColIndex], haloIndices)
        self.sendHalos[neighborDirection] = SendHalo(neighborActArrayIndex, sendHaloIndices, self.batchSize, self.backend)
        return len(recvHaloIndices)


