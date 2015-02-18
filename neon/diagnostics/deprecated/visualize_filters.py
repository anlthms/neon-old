import numpy as np
import os

from neon.util.compat import range


class Visual(object):
    """
    Visualizations from the Google Brain model
    """
    def __init__(self):
        import matplotlib.pyplot
        self.plt = matplotlib.pyplot

    def visualize_pretraining(self, ind, layer, output):
        """
        A collection of visualization routines useful for tracking the
        pretraining and showing the filters as well as the reconstucted digits.
        """

        self.plt.figure(1)
        self.plt.clf()
        self.plt.plot(self.cost_list)
        self.plt.legend(('recon', 'sparse', 'both'))
        self.plt.draw()

        self.plt.figure(2)
        self.plt.clf()
        self.visual_filters(os.path.join('recon', 'filters'), ind, layer.nifm)
        self.plt.draw()

        self.plt.figure(3)
        self.plt.clf()
        self.save_figs_all(layer.nifm, layer.ifmshape,
                           [output, layer.defilter.output],
                           [os.path.join('recon', 'input'),
                            os.path.join('recon', 'output')], ind)

    def save_figs_all(self, nfm, fmshape, imgs, names, ind):
        """
        show (color) images of the (input) data.
        Useful to see what images look like after preprocessing,
        or what denoised reconstructions look like.

        Inputs:
            nfm: number of feature maps
            fmshape: feature map shape
            imgs: dict of separate images to plot
            names: dict of names for the images
            ind: running index that goes into file names of plots.

        Outputs:
            saves png to recon/ folder
        """
        assert len(names) == len(imgs)
        hght, wdth = fmshape
        # loop over input / output
        for i in range(len(names)):
            # samples in the mini-batch
            grd = np.sqrt(imgs[i].shape[0]).astype(np.int)  # 10x10 plot grid
            win = np.sqrt(imgs[i].shape[1]/nfm).astype(np.int)  # 32 win size
            img = np.zeros((nfm, (win+1)*grd, (win+1)*grd))  # fm size

            for j in range(grd):
                for k in range(grd):
                    patch = imgs[i].asnumpyarray()[grd*j+k].reshape((nfm, hght,
                                                                     wdth))
                    img[:,
                        j*(win+1):j*(win+1)+win,
                        k*(win+1):k*(win+1)+win] = patch

            if nfm == 3:
                # Plot in color.
                self.plt.imshow(np.transpose(img/5.+.5, [2, 1, 0]))
            else:
                # Save the first feature map.
                self.plt.imshow(img[0].reshape(((win+1)*grd, (win+1)*grd)),
                                interpolation='nearest', cmap='gray',
                                vmin=-1.1, vmax=1.1)
            self.plt.show()
            self.plt.savefig(ensure_dirs_exist(names[i] + str(ind)))

    def visual_filters(self, names, ind, nifm):
        """
        reshape sparse autoencoder weights into pixel space and plot them

        Inputs:
            names: file name prefix to save figures to
            ind: index for bookkeeping purpose in the file name
            nifm: number of input feature maps, if this is 3 treat them as
                  color channels.

        Outputs:
            Plots a figure, imsaves to png.
        """

        self.plt.clf()
        w = self.layers[0].weights.asnumpyarray()  # (576, 25)
        k = np.sqrt(self.layers[0].nofm).astype(np.int)  # output feature maps
        n = np.sqrt(w.shape[0]).astype(np.int) / k
        m = np.sqrt(w.shape[1]/nifm).astype(np.int)

        def showme_sparse(w, n, m, k, nifm):
            "this places the individual filters on a canvas and plots it"
            if nifm == 1:
                canvas = np.zeros((n*(n+m), n*(n+m)))+1
                for i in range(n):
                    for j in range(n):
                        block = w[n*i+j, :].reshape(m, m)
                        canvas[(n+m)*i+0:(n+m)*i+n+m-1,
                               (n+m)*j+0:(n+m)*j+n+m-1] = 0
                        canvas[(n+m)*i+i:(n+m)*i+i+m,
                               (n+m)*j+j:(n+m)*j+j+m] = block
                self.plt.imshow(canvas, interpolation='nearest', vmin=-0.5,
                                vmax=0.5, cmap=self.plt.cm.gray)
                self.plt.show()
                self.plt.imsave(ensure_dirs_exist(names + 'grayscale' +
                                                  str(ind) + str(k)), canvas,
                                vmin=-0.5, vmax=0.5, cmap=self.plt.cm.gray)
            "this plots the filters as an RGB color image"
            if nifm == 3:
                canvas = np.zeros((n*(n+m), n*(n+m), 3))+1
                for i in range(n):
                    for j in range(n):
                        block = w[n*i+j, :].reshape(3, m, m).transpose(1, 2, 0)
                        canvas[(n+m)*i+0:(n+m)*i+n+m-1,
                               (n+m)*j+0:(n+m)*j+n+m-1, :] = 0
                        canvas[(n+m)*i+i:(n+m)*i+i+m,
                               (n+m)*j+j:(n+m)*j+j+m, :] = block
                self.plt.imshow(canvas*2.+.5, interpolation='nearest')
                self.plt.show()
                self.plt.imsave(ensure_dirs_exist(names + 'color' + str(ind)
                                                  + str(k)), canvas*2.+.5)

        def showme_dense(w, n, m, k, nifm):
            "this places the individual filters on a canvas and plots it"
            if nifm == 1:
                canvas = np.zeros((n*(m+1), n*(m+1)))
                for i in range(n):
                    for j in range(n):
                        canvas[(m+1)*i:(m+1)*i+m,
                               (m+1)*j:(m+1)*j+m] = w[n*i+j, :].reshape(m, m)
                self.plt.imshow(canvas, interpolation='nearest',
                                vmin=-0.5, vmax=0.5, cmap=self.plt.cm.gray)
                self.plt.show()
            if nifm == 3:
                canvas = np.zeros((n*(m+1), n*(m+1), 3))
                for i in range(n):
                    for j in range(n):
                        block = w[n*i+j, :].reshape(3, m, m).transpose(1, 2, 0)
                        canvas[(m+1)*i:(m+1)*i+m, (m+1)*j:(m+1)*j+m, :] = block
                self.plt.imshow(canvas*2.+.5)
                self.plt.show()

        for i in range(k**2):
            self.plt.subplot(k, k, i)
            showme_sparse(w[i*n**2:(i+1)*n**2], n, m, i, nifm)
            showme_dense(w[i*n**2:(i+1)*n**2], n, m, i, nifm)


def ensure_dirs_exist():
    raise NotImplementedError

# I don't think this will be used but I want to keep it around for now.
    # def save_state(self):
    #     "pickle current weight matrix, error curve, for offline analysis"

    #     import pickle
    #     w = self.weight_list
    #     e = self.cost_list
    #     writeout = {'w':w, 'e':e}
    #     pickle.dump( writeout, open( "dump.pkl", "wb" ) )
