# Urs Koster


from ipdb import set_trace as trace
from matplotlib import pyplot as plt
# plt.interactive(1)

def plot_trace():
    """
    
    """
    #stuff= pickle.load( open('/home/local/code/cuda-convnet2/spearmint/chooser.GPEIOptChooser.pkl', 'rb'))
    #stuff['hyper_samples'] # 10 samples, not sure where the number comes form. 

    stuff= pickle.load( open('/home/local/code/cuda-convnet2/spearmint/expt-grid.pkl', 'rb') ) # dies no more
    inx = nonzero(~isnan(stuff['values']))
    order=argsort(stuff['values'][inx])
    stuff['grid'][inx][order]*4.5+.5 # values normalized to 0-1. 
    stuff['values'][inx][order]

    # scatter plots of the parameters?
    plt.plot(stuff['grid'][inx][order]*4.5+.5, 'o')
    plt.title('step parameters from best to worst')
    plt.legend(['stepW', 'stepB'])

    # scatter plots of the parameters?
    plt.plot(stuff['values'][inx][order], stuff['grid'][inx][order][:,0]*4.5+.5, 'o')
    plt.plot(stuff['values'][inx][order], stuff['grid'][inx][order][:,1]*4.5+.5, 'o')

    # search path over time:
    plt.plot(stuff['values'][inx])



    # mean and variance of the parameters over the 10% best experiments:
    plt.clf()
    extra=0
    for i in range(7):
        if i==3: extra=1
        plt.subplot(5,2,i+1+extra)
        #plt.hist(stuff['grid'][inx][order][:,i],10,(0,1))
        for k in range(10):
            plt.hist(stuff['grid'][inx][order[0:np.int(order.shape[0]/(k+1.))]][:,i],10,(0,1), color=str(1./(k+1)))
            plt.title(str(stuff['grid'][inx][order[0:np.int(order.shape[0]/10.)]][:,i].mean()))






    
if __name__ == "__main__":

    """
    
    """

    plot_trace()

    