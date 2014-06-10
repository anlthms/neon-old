"""
RBM implementation on top of Theano, adapted from example code on deeplearning.net.

The RBM class in this file is meant to be a drop-in replacement for other RBM classes
implemented with cudamat/gnumpy/numpy. To make this possible, it implements the applyCD()
and rbmup() calls with the same signature as the other implementations.

CAVEAT: This code was initially written assuming that the input images are arranged in rows.
The code in dbn_class.py assumes that the images are arranged in columns. As a temporary
hack, transpose operations were put in. These are tagged with "FIXME" in the code.

"""

import pickle
import rbm_math as rm
import numpy as np
import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class rbm_class:
    """ A stackable RBM that can be used to build a DBN.
    """
    def __init__(self, opts):
        self.nvis = opts['numVisible']
        self.nhid = opts['numHidden']
        numpyrng = np.random.RandomState(0) 
        self.theanorng = RandomStreams(numpyrng.randint(2 ** 30))
        
        if opts['FILE_LOAD_FLAG']:
            self.load_rbm_weights()
        else:
            w = 0.01 * numpyrng.randn(self.nvis, self.nhid)

            self.weights = theano.shared(value=w, name='weights', borrow=True)
            self.hbias = theano.shared(value=np.zeros(self.nhid,
                                   dtype=theano.config.floatX),
                                   name='hbias', borrow=True)
            self.vbias = theano.shared(value=np.zeros(self.nvis,
                                   dtype=theano.config.floatX),
                                   name='vbias', borrow=True)
                    
        self.penalty = .001                 # FIXME: add regularization.
        self.eta = opts['eta']
        self.momentum = opts['momentum']    # FIXME: use momentum.
        self.maxepoch = opts['maxEpoch']
        self.batchsize = opts['batchsize']
        
        self.params = [self.weights, self.hbias, self.vbias]

    def save_rbm_weights(self, labelsFlag=False):
        with open('rbm_weights.pkl', 'w') as f:
            pickle.dump([self.weights, self.hbias, self.vbias, self.nhid,
                         self.nvis],f)
    
    def load_rbm_weights(self, labelsFlag=False):
        with open('rbm_weights.pkl') as f:
            [self.weights, self.hbias, self.vbias, self.nhid,
             self.nvis] = pickle.load(f)

    def rbmup(self, v):
        # FIXME: unnecessary transposing.
        z = np.dot(v.T, self.weights.get_value(borrow=True)) + \
                self.hbias.get_value(borrow=True)
        return rm.logistic(z).T

    def uprop(self, v):
        z = T.dot(v, self.weights) + self.hbias
        return [z, T.nnet.sigmoid(z)]
            
    def hsample(self, v):
        z, h = self.uprop(v)
        sample = self.theanorng.binomial(size=h.shape, n=1, p=h,
                                          dtype=theano.config.floatX)
        return [z, h, sample]

    def dprop(self, h):
        z = T.dot(h, self.weights.T) + self.vbias
        return [z, T.nnet.sigmoid(z)]

    def vsample(self, h):
        z, v = self.dprop(h)
        sample = self.theanorng.binomial(size=v.shape, n=1, p=v,
                                          dtype=theano.config.floatX)
        return [z, v, sample]

    def hgibbs(self, h):
        zv, v, vsample = self.vsample(h)
        zh, h, hsample = self.hsample(vsample)
        return [zv, v, vsample,
                zh, h, hsample]

    def vgibbs(self, v):
        zh, h, hsample = self.hsample(v)
        zv, v, vsample = self.vsample(hsample)
        return [zh, h, hsample,
                zv, v, vsample]

    def get_updates(self, k=1):
        # Positive phase.
        zh, h, hsample = self.hsample(self.input)
        chain = hsample

        # Negative phase. Scan over the gibbs sampling function k times.
        [zvs, vs, vsamples,
         zhs, hs, hsamples], updates = \
            theano.scan(self.hgibbs,
                        outputs_info=[None, None, None, None, None, chain],
                        n_steps=k)

        # Get the end of the chain.
        end = vsamples[-1]

        cost = T.mean(self.get_free_energy(self.input)) - T.mean(
                self.get_free_energy(end)) 
        gparams = T.grad(cost, self.params, consider_constant=[end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(self.eta,
                                                     dtype=theano.config.floatX)
        ce = self.get_cross_entropy(updates, zvs[-1])

        return ce, updates

    def get_free_energy(self, vsample):
        wxb = T.dot(vsample, self.weights) + self.hbias
        vbiasterm = T.dot(vsample, self.vbias)
        hiddenterm = T.sum(T.log(1 + T.exp(wxb)), axis=1)
        return -hiddenterm - vbiasterm

    def get_cross_entropy(self, updates, zv):
        y = T.nnet.sigmoid(zv)
        return T.mean(T.sum(self.input * T.log(y) +
                      (1 - self.input) * T.log(1 - y), axis=1))

    def applyCD(self, xtrain, k=1):
        """ Perform CD-k. This function uses a different naming convention to
            match with other rbm_class implementations.
        """
        sharedx = theano.shared(np.asarray(xtrain,
                                           dtype=theano.config.floatX),
                                           borrow=True) 
        assert xtrain.shape[1] % self.batchsize == 0  # FIXME
        nbatches = xtrain.shape[1] / self.batchsize 

        index = T.lscalar()
        x = T.matrix('x')
        self.input = x

        cost, updates = self.get_updates(k)

        train_rbm = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                # FIXME: unnecessary transposing.
                givens={x: sharedx[:, (index * self.batchsize) :
                                   (index + 1) * self.batchsize].T},
                name='train_rbm')

        starttime = time.clock()

        for epoch in xrange(self.maxepoch):
            meancost = []
            for batchindex in xrange(nbatches):
                meancost += [train_rbm(batchindex)]

            print 'Training epoch %d, cost ' % epoch, np.mean(meancost)

        print ('Training took %f minutes' % ((time.clock() - starttime) / 60.))
        self.W = self.weights.get_value(borrow=True)
        self.b = self.hbias.get_value(borrow=True).reshape((self.nhid, 1))
        self.c = self.vbias.get_value(borrow=True).reshape((self.nvis, 1))
        return self
