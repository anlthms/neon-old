from neon.transforms.sum_squared import SumSquaredDiffs as neonMSE
from neon.transforms.cross_entropy import CrossEntropy as neonXEnt
import numpy as np


class Cost:
    def __init__(self, backend, c=1.):
        self.backend = backend
        self.C = self.backend.wrap(c)

    def forward(self, x, y):
        return self.backend.multiply(self.C, self.cost(x, y))

    def backward(self, x, y):
        return self.backend.multiply(self.C, self.dcost(x, y))


class Mse(Cost, neonMSE):
    def cost(self, x, y, temp=None):
        return self.apply_function(self.backend, x, y, temp)

    def dcost(self, x, y, temp=None):
        return self.apply_derivative(self.backend, x, y, temp)


class CrossEntropy(Cost, neonXEnt):
    def cost(self, x, y, temp=None):
        return self.apply_function(self.backend, x, y, temp)

    def dcost(self, x, y, temp=None):
        return self.apply_function(self.backend, x, y, temp)


class HingeL2(Cost):
    def cost(self, x, y):
        raise NotImplementedError
        a = 1.-x*y
        a *= (a > 0.)
        return 0.5*(a**2.).sum(1)

    def dcost(self, x, y):
        raise NotImplementedError
        a = (1-x*y)
        a *= (a > 0.)
        a *= -y
        return a


class PartialHingeL2(Cost):
    def __init__(self, stop_idx, c=1.):
        raise NotImplementedError
        self.stop_idx = stop_idx
        self.C = c

    def cost(self, x, y):
        a = 1.-x*y
        a *= (a > 0.)
        a[:, self.stop_idx:] = 0.
        return 0.5*(a**2.).sum(1)

    def dcost(self, x, y):
        a = (1-x*y)
        a *= (a > 0.)
        a[:, self.stop_idx:] = 0.
        a *= -y
        return a


class XCov(Cost):
    def __init__(self, block_idx, c=1.):
        raise NotImplementedError
        self.bidx = block_idx
        self.C = c

    def xcov(self, x, y):
        n = x.shape[0]
        xc = x-x.mean(0, keepdims=True)
        yc = y-y.mean(0, keepdims=True)
        return xc.T.dot(yc)/n

    def cost(self, x, y):
        return 0.5*(self.xcov(x[:, :self.bidx], x[:, self.bidx:])**2.).sum()

    def dcost(self, x, y):
        cross_cov = self.xcov(x[:, :self.bidx], x[:, self.bidx:])
        n = x.shape[0]
        c1 = (x[:, self.bidx:]-x[:, self.bidx:].mean(0, keepdims=True)).T
        c2 = (x[:, :self.bidx]-x[:, :self.bidx].mean(0, keepdims=True))
        s = np.zeros_like(x)
        s[:, :self.bidx] = cross_cov.dot(c1).T/n
        s[:, self.bidx:] = c2.dot(cross_cov)/n
        return s


class MuXCov(Cost):
    def __init__(self, block_idx, c=1.):
        raise NotImplementedError
        self.bidx = block_idx
        self.C = c

    def xcov(self, x, y):
        n = x.shape[0]
        xc = x-x.mean(0, keepdims=True)
        yc = y-y.mean(0, keepdims=True)
        return xc.T.dot(yc)/n

    def cost(self, x, y):
        subx_a = x[:, :self.bidx]
        subx_b = x[:, self.bidx:]
        a = (self.xcov(subx_a, subx_b)**2.).sum() + (subx_b.mean(0)**2.).sum()
        return 0.5*a

    def dcost(self, x, y):
        subx_a = x[:, :self.bidx]
        subx_b = x[:, self.bidx:]
        cross_cov = self.xcov(subx_a, subx_b)
        n = x.shape[0]
        c1 = (subx_b - subx_b.mean(0, keepdims=True)).T
        c2 = (subx_a - subx_a.mean(0, keepdims=True))
        s = np.zeros_like(x)
        s[:, :self.bidx] = cross_cov.dot(c1).T/n
        s[:, self.bidx:] = c2.dot(cross_cov)/n + subx_b.mean(0, keepdims=True)
        return s


class ZCov(Cost):
    def __init__(self, block_idx, c=1.):
        raise NotImplementedError
        self.bidx = block_idx
        self.C = c

    def xcov(self, x, y):
        n = x.shape[0]
        xc = x-x.mean(0, keepdims=True)
        yc = y-y.mean(0, keepdims=True)
        return xc.T.dot(yc)/n

    def cost(self, x, y):
        cross_cov = self.xcov(x, x[:, self.bidx:])
        np.fill_diagonal(cross_cov[self.bidx:], 0.)
        return 0.5*(cross_cov**2.).sum()

    def dcost(self, x, y):
        cross_cov = self.xcov(x, x[:, self.bidx:])
        np.fill_diagonal(cross_cov[self.bidx:], 0.)
        n = x.shape[0]
        c1 = (x[:, self.bidx:]-x[:, self.bidx:].mean(0, keepdims=True)).T
        c2 = x-x.mean(0, keepdims=True)
        s = np.zeros_like(x)
        s[:, :self.bidx] = cross_cov[:self.bidx].dot(c1).T/n
        s[:, self.bidx:] = c2.dot(cross_cov)/n
        return s
