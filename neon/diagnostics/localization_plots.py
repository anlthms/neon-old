# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Logging and visalization for the data collected from backend timing decorators
"""

import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')  # not for plotting but write to file.
from matplotlib import pyplot as plt  # with a middlefinger to pep8: # noqa
matplotlib.rcParams['pdf.fonttype'] = 42  # TTF to be editable

logger = logging.getLogger(__name__)


# def predict_and_localize(dataset=None):

#     # setting up data
#     if dataset is not None:
#         self.data_layer.init_dataset(dataset)
#     dataset.set_batch_size(self.batch_size)
#     self.data_layer.use_set('validation', predict=True)

#     # seting up layers
#     self.layers[0].ofmshape = [32, 32]  # TODO: Move this to yaml

#     for l in range(1, len(self.layers)-1):
#         delattr(self.layers[l], 'delta_shape')
#         delattr(self.layers[l], 'out_shape')

#     self.link()
#     self.initialize(self.backend)
#     self.print_layers()
#     self.fprop()
#     self.visualize()

def visualize_location_maps(model):
    """
    Rudimentary visualization code for localization experiments:
    """
    # look at the data
    mapp = model.layers[5].output.asnumpyarray()  # [200 is 8 x (5x5)] x 128
    mapp0 = mapp[0:25].reshape(5, 5, -1)  # take the first feature map (zeros)
    mapp1 = mapp[25:50].reshape(5, 5, -1)  # and second feature map (ones)
    databatch = model.layers[0].output.asnumpyarray()  # grab input batch
    labels = model.layers[0].targets.asnumpyarray()[0,:]
    print "labels", labels
    new_order = np.argsort(labels)
    data0 = databatch[0*1024:1*1024].reshape(32, 32, -1)  # feature 1/8 (pressure?)
    data1 = databatch[1*1024:2*1024].reshape(32, 32, -1)  # feature 2/8 (temperature?)

    myplot(plt, mapp0, title='positive class label strength',
           span=(0,1), fig=0)
    # myplot(plt, mapp1, title='negative class label strength',
    #        span=(mapp1.min(), mapp1.max()), fig=1)
    myplot(plt, data0, title='data variable 0',
           span=(-1, 1.5), fig=2)
    # myplot(plt, data1, title='data variable 1',
    #        span=(-2, 2), fig=3)

    print("setting trace to keep plots open...")
    import pdb; pdb.set_trace()

def myplot(plt, data, title, span, fig):
    """
    wrapper for imshow that goes through 100 examples and makes subplots.
    TODO: Move this and visualize() to diagnostics.
    """
    plt.figure(fig, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
    plt.clf()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(data[..., i], interpolation='none',
                   vmin=span[0], vmax=span[1])
    plt.subplot(10, 10, 5)
    plt.title(title)
    #plt.tight_layout()
    print("saving figure: " + 'localization_' + title.replace(' ', '_'))
    plt.savefig('localization_' + title.replace(' ', '_'), dpi=120)

