import matplotlib.pyplot as plt
plt.interactive(1)
import numpy as np
import os
from ipdb import set_trace as trace

# bokeh is a library that plots to the browser.
from bokeh import plotting as blt

"""
Using Bokeh:
    first, need to start a server with 
        bokeh-server --redis-port 7002 &
    from shell, make sure old sessions are killed

    then create a document
        blt.output_server('lala')
    and add plots to it. 
        blt.line(np.arange(3), np.random.rand(3))
        blt.scatter(flowers["petal_length"], flowers["petal_width"])
    which all appear one after the other. 

Some more example stuff: 
    holding on:
        blt.line(np.arange(3), np.random.rand(3))
        blt.hold(True)
        blt.line(np.arange(3), np.random.rand(3))
        blt.hold(False)
    making subplots:



"""

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
        plt.plot(np.arange(len(suberror_list))/np.float(len(suberror_list))
                                              *len(error_list), suberror_list)
        plt.plot(error_list, linewidth=2)
        plt.ylim((.010,.035))
        plt.draw(); plt.show()

    def plot_activations(self, pre1, out1, pre2, out2, targets, batch_inx):
        plt.figure(3)
        plt.clf()
        for i in range(4):
            plt.subplot(4,5,5*i+1)
            plt.imshow(pre1[i].raw(), vmin=-1, vmax=1, interpolation='nearest');
            if i==0: plt.title('pre1 or g\'1')
            plt.subplot(4,5,5*i+2)
            plt.imshow(out1[i].raw(), vmin=-1, vmax=1, interpolation='nearest');
            if i==0: plt.title('out1')
            plt.subplot(4,5,5*i+3)
            plt.imshow(pre2[i].raw(), vmin=-1, vmax=1, interpolation='nearest');
            if i==0: plt.title('pre2 or g\'2')
            plt.subplot(4,5,5*i+4)
            plt.imshow(out2[i].raw(), vmin=-1, vmax=1, interpolation='nearest');
            if i==0: plt.title('out2')
            plt.subplot(4,5,5*i+5)
            plt.imshow(targets[batch_inx[:,i]].raw(), vmin=-1, vmax=1,
                                                      interpolation='nearest');
            if i==0: plt.title('target')
        plt.draw(); plt.show()


