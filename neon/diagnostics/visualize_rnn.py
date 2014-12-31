# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Visualization for recurrent neural networks
"""

import matplotlib.pyplot as plt
plt.interactive(1)
import numpy as np
from neon.util.compat import range


class VisualizeRNN(object):
    """
    Visualzing weight matrices during training
    """
    def __init__(self):
        pass

    def plot_weights(self, weights_in, weights_rec, weights_out):
        """
        Visizualize the three weight matrices after every epoch. Serves to
        check that weights are structured, not exploding, and get upated
        """
        plt.figure(2)
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(weights_in.T, vmin=-1, vmax=1, interpolation='nearest')
        plt.title('input.T')
        plt.subplot(1, 3, 2)
        plt.imshow(weights_rec, vmin=-1, vmax=1, interpolation='nearest')
        plt.title('recurrent')
        plt.subplot(1, 3, 3)
        plt.imshow(weights_out, vmin=-1, vmax=1, interpolation='nearest')
        plt.title('output')
        plt.colorbar()
        plt.draw()
        plt.show()

    def plot_error(self, suberror_list, error_list):
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(len(suberror_list)) / np.float(len(suberror_list))
                 * len(error_list), suberror_list)
        plt.plot(error_list, linewidth=2)
        plt.ylim((.010, .035))
        plt.draw()
        plt.show()

    def plot_activations(self, pre1, out1, pre2, out2, targets):
        """
        Loop over tau unrolling steps, at each time step show the pre-acts
        and outputs of the recurrent layer and output layer. Note that the
        pre-acts are actually the g', so if the activation is linear it will
        be one.
        """

        plt.figure(3)
        plt.clf()
        for i in range(4):
            plt.subplot(4, 5, 5*i+1)
            plt.imshow(pre1[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('pre1 or g\'1')
            plt.subplot(4, 5, 5*i+2)
            plt.imshow(out1[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('out1')
            plt.subplot(4, 5, 5*i+3)
            plt.imshow(pre2[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('pre2 or g\'2')
            plt.subplot(4, 5, 5*i+4)
            plt.imshow(out2[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('out2')
            plt.subplot(4, 5, 5*i+5)
            plt.imshow(targets[i*128:(i+1)*128, :].asnumpyarray(),
                       vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                plt.title('target')
        plt.draw()
        plt.show()

    def print_text(self, inputs, outputs):
        """
        Moved this here so it's legal to use numpy.
        """
        print("Prediction inputs")
        print(np.argmax(inputs, 0).asnumpyarray().astype(np.int8).view('c'))
        print("Prediction outputs")
        print(np.argmax(outputs, 0).asnumpyarray().astype(np.int8).view('c'))
