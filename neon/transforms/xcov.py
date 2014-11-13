"""
XCov cost functions and classes for balance networks
"""

from neon.transforms.cost import Cost

def xcov_cost(backend, outputs, targets, temp, blkidx=None):
    N = outputs.shape[outputs.major_axis()]
    if (blkidx!=None):
        blk1 = outputs.get_major_slice(0, blkidx)
        blk2 = outputs.get_major_slice(blkidx, N)
    else:
        blk1 = outputs
        blk2 = outputs

    backend.xcov(blk1, blk2, out=temp[0])
    return 0.5*temp[0].sumsq()

def xcov_cost_derivative(backend, outputs, targets, temp, blkidx=None):
    #temp[0] is k1 x k2
    #temp[1] is N x K1
    #temp[2] is N x K2
    #temp[3] is N x (K1+K2)

    # TODO: make sure that the dots are consistent across backends for this
    # arrangement
    N = outputs.shape[outputs.major_axis()]
    K1 = blkidx
    K2 = outputs.shape[outputs.minor_axis()] - blkidx
    if (blkidx!=None):
        blk1 = outputs.get_major_slice(0, blkidx)
        blk2 = outputs.get_major_slice(blkidx, N)

        backend.mean_norm(blk1, axis=outputs.major_axis(), out=temp[1])
        backend.xcov(blk1, blk2, out=temp[0])
        backend.dot(temp[1], temp[0], out=temp[2])
        temp[3].set_major_slice(blkidx, N, temp[2])

        backend.mean_norm(blk2, axis=outputs.major_axis(), out=temp[2])
        temp[0].reshape((K2,K1))
        backend.xcov(blk2, blk1, out=temp[0])
        backend.dot(temp[2], temp[0], out=temp[1])
        temp[3].set_major_slice(0, blkidx, temp[1])

        backend.multiply(temp[3], backend.wrap(1./N), out=temp[3])
        return temp[3]
    else:
        raise NotImplementedError('Need blkidx defined for xcov_cost_derivative')

class XCovariance(Cost):

    """
    Embodiment of a X covariance cost function.
    """

    @staticmethod
    def apply_function(backend, outputs, targets, temp):
        """
        Apply the xcov cost function to the datasets passed.
        """
        return xcov_cost(backend, outputs, targets, temp, blkidx=self.blkidx)

    @staticmethod
    def apply_derivative(backend, outputs, targets, temp):
        """
        Apply the derivative of the xcov cost function to the datasets
        passed.
        """
        return xcov_cost_derivative(backend, outputs, targets, temp, blkidx=self.blkidx)
