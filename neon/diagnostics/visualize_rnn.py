import matplotlib.pyplot as plt
import numpy as np
import os


class VisualizeRNN(object):
    """
    Visualzing weight matrices during training
    """
    def __init__():
        pass

    def visualize_training(self, ind, layer, output):
        """
        W_in, W_rec, W_out
        """

        plt.figure(1); plt.clf()
        plt.plot(self.cost_list); plt.legend(('recon', 'sparse', 'both'))
        plt.draw()
        plt.figure(2); plt.clf()
        self.visual_filters(os.path.join('recon', 'filters'), ind, layer.nifm)
        plt.draw()
        plt.figure(3); plt.clf()
        self.save_figs_all(layer.nifm, layer.ifmshape,
                        [output, layer.defilter.output],
                        [os.path.join('recon', 'input'),
                        os.path.join('recon', 'output')], ind)

