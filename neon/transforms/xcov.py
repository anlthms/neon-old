"""
XCov cost functions and classes for balance networks
"""

from neon.transforms.cost import Cost


def xcov_cost(backend, outputs, targets, temp, blkidx):
    blk1 = outputs[0:blkidx]
    blk2 = outputs[blkidx:]
    backend.xcov(blk1, blk2, out=temp[2])
    return 0.5*temp[2].sumsq()


def xcov_cost_derivative(backend, outputs, targets, temp, blkidx,
                         scale=1.0):
    # temp[0] is k1 x n
    # temp[1] is k2 x n
    # temp[2] is k1 x k2
    # temp[3] is (k1+k2) x n

    n = outputs.shape[1]
    blk1 = outputs[0:blkidx]
    blk2 = outputs[blkidx:]

    backend.mean_norm(blk1, axis=1, out=temp[0])
    backend.xcov(blk1, blk2, out=temp[2])
    backend.dot(temp[2].transpose(), temp[0], out=temp[1])
    temp[3][blkidx:] = temp[1]

    backend.mean_norm(blk2, axis=1, out=temp[1])
    backend.dot(temp[2], temp[1], out=temp[0])
    temp[3][:blkidx] = temp[0]

    backend.multiply(temp[3], scale / n, out=temp[3])
    return temp[3]


class XCovariance(Cost):

    """
    Embodiment of a X covariance cost function.
    """
    def __init__(self, **kwargs):
        super(XCovariance, self).__init__(**kwargs)

        for req_param in ['blkidx']:
            if not hasattr(self, req_param):
                raise ValueError("required parameter: %s not specified" %
                                 req_param)

    def initialize(self, kwargs):
        super(XCovariance, self).initialize(kwargs)
        if self.blkidx > self.outputbuf.shape[0]:
            raise ValueError("blkidx %d too large" % self.blkidx)

    def set_outputbuf(self, databuf):
        if not self.outputbuf:
            self.outputbuf = databuf

        if self.outputbuf.shape != databuf.shape or not self.temp:
            n = self.outputbuf.shape[1]
            k1 = self.blkidx
            k2 = self.outputbuf.shape[0]-k1
            tempbuf1 = self.backend.empty((k1, n), self.temp_dtype)
            tempbuf2 = self.backend.empty((k2, n), self.temp_dtype)
            tempbuf3 = self.backend.empty((k1, k2), self.temp_dtype)
            tempbuf4 = self.backend.empty(self.outputbuf.shape,
                                          self.temp_dtype)
            self.temp = [tempbuf1, tempbuf2, tempbuf3, tempbuf4]
        self.outputbuf = databuf

    def get_berrbuf(self):
        return self.temp[3]

    def apply_function(self, targets):
        """
        Apply the xcov cost function to the datasets passed.
        """
        return xcov_cost(self.backend, self.outputbuf, targets, self.temp,
                         self.blkidx) * self.scale

    def apply_derivative(self, targets):
        """
        Apply the derivative of the xcov cost function to the datasets
        passed.
        """
        return xcov_cost_derivative(self.backend, self.outputbuf, targets,
                                    self.temp, self.blkidx, self.scale)
