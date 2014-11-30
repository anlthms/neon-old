"""
XCov cost functions and classes for balance networks
"""

from neon.transforms.cost import Cost


def xcov_cost(backend, outputs, targets, temp, blkidx=None):
    n = outputs.shape[0]
    if not blkidx:
        blk1 = outputs
        blk2 = outputs
    else:
        blk1 = outputs[0:blkidx]
        blk2 = outputs[blkidx:n]

    backend.xcov(blk1, blk2, out=temp[0])
    return 0.5*temp[0].sumsq()


def xcov_cost_derivative(backend, outputs, targets, temp, blkidx=None):
    #temp[0] is k1 x k2
    #temp[1] is n x k1
    #temp[2] is n x k2
    #temp[3] is n x (k1+k2)

    # TODO: make sure that the dots are consistent across backends for this
    # arrangement
    n = outputs.shape[0]

    if not blkidx:
        raise NotImplementedError("Need blkidx defined for"
                                  "xcov_cost_derivative")
    else:
        k1 = blkidx
        k2 = outputs.shape[1] - blkidx
        blk1 = outputs[0:blkidx]
        blk2 = outputs[blkidx:n]

        backend.mean_norm(blk1, axis=0, out=temp[1])
        backend.xcov(blk1, blk2, out=temp[0])
        backend.dot(temp[1], temp[0], out=temp[2])
        temp[3][blkidx:n] = temp[2]

        backend.mean_norm(blk2, axis=0, out=temp[2])
        temp[0].reshape((k2, k1))
        backend.xcov(blk2, blk1, out=temp[0])
        backend.dot(temp[2], temp[0], out=temp[1])
        temp[3][0:blkidx] = temp[1]

        backend.multiply(temp[3], backend.wrap(1./n), out=temp[3])
        return temp[3]


class XCovariance(Cost):

    """
    Embodiment of a X covariance cost function.
    """
    def __init__(self, blkidx=None):
        self.blkidx = blkidx

    def apply_function(self, backend, outputs, targets, temp):
        """
        Apply the xcov cost function to the datasets passed.
        """
        return xcov_cost(backend, outputs, targets, temp, blkidx=self.blkidx)

    def apply_derivative(self, backend, outputs, targets, temp):
        """
        Apply the derivative of the xcov cost function to the datasets
        passed.
        """
        return xcov_cost_derivative(backend, outputs, targets, temp,
                                    blkidx=self.blkidx)
