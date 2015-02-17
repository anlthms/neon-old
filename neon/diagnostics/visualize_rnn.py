# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Visualization for recurrent neural networks
"""

import numpy as np
from neon.util.compat import range
import matplotlib.pyplot as plt
plt.interactive(1)


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

    def plot_lstm_wts(self, lstm_layer, scale=1, fig=4):

        """
        Visizualize the three weight matrices after every epoch. Serves to
        check that weights are structured, not exploding, and get upated
        """
        plt.figure(fig)
        plt.clf()
        pltidx = 1
        for lbl, wts in zip(lstm_layer.param_names, lstm_layer.params[:4]):
            plt.subplot(2, 4, pltidx)
            plt.imshow(wts.asnumpyarray().T, vmin=-scale, vmax=scale,
                       interpolation='nearest')
            plt.title(lbl + ' Wx.T')
            pltidx += 1

        for lbl, wts, bs in zip(lstm_layer.param_names,
                                lstm_layer.params[4:8],
                                lstm_layer.params[8:12]):
            plt.subplot(2, 4, pltidx)
            plt.imshow(np.hstack((wts.asnumpyarray(),
                                  bs.asnumpyarray(),
                                  bs.asnumpyarray())).T,
                       vmin=-scale, vmax=scale, interpolation='nearest')
            plt.title(lbl + ' Wh.T')
            pltidx += 1

        plt.draw()
        plt.show()

    def plot_lstm_acts(self, lstm_layer, scale=1, fig=4):
        acts_lbl = ['i_t', 'f_t', 'o_t', 'g_t', 'net_i', 'c_t', 'c_t', 'c_phi']
        acts_stp = [0, 0, 0, 1, 0, 0, 1, 1]
        plt.figure(fig)
        plt.clf()
        for idx, lbl in enumerate(acts_lbl):
            act_tsr = getattr(lstm_layer, lbl)[acts_stp[idx]]
            plt.subplot(2, 4, idx+1)
            plt.imshow(act_tsr.asnumpyarray().T,
                       vmin=-scale, vmax=scale, interpolation='nearest')
            plt.title(lbl + '[' + str(acts_stp[idx]) + '].T')

        plt.draw()
        plt.show()

    def plot_error(self, suberror_list, error_list):
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(len(suberror_list)) / np.float(len(suberror_list))
                 * len(error_list), suberror_list)
        plt.plot(error_list, linewidth=2)
        plt.ylim((min(suberror_list), max(error_list)))
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
            plt.subplot(len(pre1), 5, 5 * i + 1)
            plt.imshow(pre1[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('pre1 or g\'1')
            plt.subplot(len(pre1), 5, 5 * i + 2)
            plt.imshow(out1[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('out1')
            plt.subplot(len(pre1), 5, 5 * i + 3)
            plt.imshow(pre2[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('pre2 or g\'2')
            plt.subplot(len(pre1), 5, 5 * i + 4)
            plt.imshow(out2[i].asnumpyarray(), vmin=-1, vmax=1,
                       interpolation='nearest')
            if i == 0:
                plt.title('out2')
            plt.subplot(len(pre1), 5, 5 * i + 5)
            plt.imshow(targets[i].asnumpyarray(),
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
