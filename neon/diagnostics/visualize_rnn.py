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
# from ipdb import set_trace as trace


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

    def plot_lstm(self, W_ix, W_fx, W_ox, W_cx, W_ih, W_fh, W_oh, W_ch,
                  scale=1, fig=4):
        """
        Visizualize the three weight matrices after every epoch. Serves to
        check that weights are structured, not exploding, and get upated
        """
        plt.figure(fig)
        plt.clf()
        plt.subplot(2, 4, 1)
        plt.imshow(W_ix.T, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('input.T')
        plt.subplot(2, 4, 2)
        plt.imshow(W_fx.T, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('forget.T')
        plt.subplot(2, 4, 3)
        plt.imshow(W_ox.T, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('output.T')
        plt.subplot(2, 4, 4)
        plt.imshow(W_cx.T, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('cell.T')

        plt.subplot(2, 4, 5)
        plt.imshow(W_ih, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('input')
        plt.subplot(2, 4, 6)
        plt.imshow(W_fh, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('forget')
        plt.subplot(2, 4, 7)
        plt.imshow(W_oh, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('output')
        plt.subplot(2, 4, 8)
        plt.imshow(W_ch, vmin=-scale, vmax=scale, interpolation='nearest')
        plt.title('cell')

        # plt.colorbar()
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
        for i in range(len(pre1)):  # loop over unrolling
            plt.subplot(len(pre1), 5, 5*i+1)
            plt.imshow(pre1[i].raw(), vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                plt.title('pre1 or g\'1')
            plt.subplot(len(pre1), 5, 5*i+2)
            plt.imshow(out1[i].raw(), vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                plt.title('out1')
            plt.subplot(len(pre1), 5, 5*i+3)
            plt.imshow(pre2[i].raw(), vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                plt.title('pre2 or g\'2')
            plt.subplot(len(pre1), 5, 5*i+4)
            plt.imshow(out2[i].raw(), vmin=-1, vmax=1, interpolation='nearest')
            if i == 0:
                plt.title('out2')
            plt.subplot(len(pre1), 5, 5*i+5)
            plt.imshow(targets[i*128:(i+1)*128, :].raw(),
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
        print(np.argmax(inputs, 0).raw().astype(np.int8).view('c'))
        print("Prediction outputs")
        print(np.argmax(outputs, 0).raw().astype(np.int8).view('c'))
