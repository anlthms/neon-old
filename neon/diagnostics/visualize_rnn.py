import matplotlib.pyplot as plt
plt.interactive(1)
import numpy as np
import os
from ipdb import set_trace as trace

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
        plt.subplot(1,3,1); 
        plt.imshow(weights_in.T, vmin=-1, vmax=1, interpolation='nearest');
        plt.title('input.T')
        plt.subplot(1,3,2); 
        plt.imshow(weights_rec, vmin=-1, vmax=1, interpolation='nearest');
        plt.title('recurrent')
        plt.subplot(1,3,3); 
        plt.imshow(weights_out, vmin=-1, vmax=1, interpolation='nearest');
        plt.title('output')
        plt.colorbar()
        plt.draw(); plt.show()

    def plot_error(self, suberror_list, error_list):
        plt.figure(1)
        plt.clf()
        #trace()
        plt.plot(np.arange(len(suberror_list))/np.float(len(suberror_list))*len(error_list), suberror_list)
        plt.plot(error_list, linewidth=2)
        plt.ylim((0,.05))
        plt.draw(); plt.show()