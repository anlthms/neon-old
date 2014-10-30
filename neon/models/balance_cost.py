from neon.transforms.sum_squared import SumSquaredDiffs as neonMSE
from neon.transforms.cross_entropy import CrossEntropy as neonXEnt

class Cost:
    def __init__(self, backend, C=1.):
        self.backend = backend
        self.C = self.backend.wrap(C)

    def forward(self, x, y):
        return self.backend.multiply(self.C, self.cost(x,y))

    def backward(self, x, y):
        return self.backend.multiply(self.C, self.dcost(x,y))

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
        a *= (a>0.)
        return 0.5*(a**2.).sum(1)

    def dcost(self, x, y):
        raise NotImplementedError
        a = (1-x*y)
        a *= (a>0.)
        a *= -y
        return a

class PartialHingeL2(Cost):
    def __init__(self, stop_idx, C=1.):
        raise NotImplementedError
        self.stop_idx = stop_idx
        self.C = C

    def cost(self, x, y):
        a = 1.-x*y
        a *= (a>0.)
        a[:,self.stop_idx:] = 0.
        return 0.5*(a**2.).sum(1)

    def dcost(self, x, y):
        a = (1-x*y)
        a *= (a>0.)
        a[:,self.stop_idx:] = 0.
        a *= -y
        return a

class XCov(Cost):
    def __init__(self, block_idx, C=1.):
        raise NotImplementedError
        self.block_idx = block_idx
        self.C = C

    def xcov(self, x, y):
        N = x.shape[0]
        xc = x-x.mean(0,keepdims=True)
        yc = y-y.mean(0,keepdims=True)
        return xc.T.dot(yc)/N

    def cost(self, x, y):
        return 0.5*(self.xcov(x[:,:self.block_idx],x[:,self.block_idx:])**2.).sum()

    def dcost(self, x, y):
        cross_cov = self.xcov(x[:,:self.block_idx],x[:,self.block_idx:])
        N = x.shape[0]
        C1 = (x[:,self.block_idx:]-x[:,self.block_idx:].mean(0,keepdims=True)).T
        C2 = (x[:,:self.block_idx]-x[:,:self.block_idx].mean(0,keepdims=True))
        s = np.zeros_like(x)
        s[:,:self.block_idx] = cross_cov.dot(C1).T/N
        s[:,self.block_idx:] = C2.dot(cross_cov)/N
        return s

class muXCov(Cost):
    def __init__(self, block_idx, C=1.):
        raise NotImplementedError
        self.block_idx = block_idx
        self.C = C

    def xcov(self, x, y):
        N = x.shape[0]
        xc = x-x.mean(0,keepdims=True)
        yc = y-y.mean(0,keepdims=True)
        return xc.T.dot(yc)/N

    def cost(self, x, y):
        return 0.5*(self.xcov(x[:,:self.block_idx],x[:,self.block_idx:])**2.).sum() + 
            0.5*(x[:,self.block_idx:].mean(0)**2.).sum()

    def dcost(self, x, y):
        cross_cov = self.xcov(x[:,:self.block_idx],x[:,self.block_idx:])
        N = x.shape[0]
        C1 = (x[:,self.block_idx:]-x[:,self.block_idx:].mean(0,keepdims=True)).T
        C2 = (x[:,:self.block_idx]-x[:,:self.block_idx].mean(0,keepdims=True))
        s = np.zeros_like(x)
        s[:,:self.block_idx] = cross_cov.dot(C1).T/N
        s[:,self.block_idx:] = C2.dot(cross_cov)/N + x[:,self.block_idx:].mean(0,keepdims=True)
        return s

class ZCov(Cost):
    def __init__(self, block_idx, C=1.):
        raise NotImplementedError
        self.block_idx = block_idx
        self.C = C

    def xcov(self, x, y):
        N = x.shape[0]
        xc = x-x.mean(0,keepdims=True)
        yc = y-y.mean(0,keepdims=True)
        return xc.T.dot(yc)/N

    def cost(self, x, y):
        cross_cov = self.xcov(x,x[:,self.block_idx:])
        np.fill_diagonal(cross_cov[self.block_idx:], 0.)
        return 0.5*(cross_cov**2.).sum()

    def dcost(self, x, y):
        cross_cov = self.xcov(x,x[:,self.block_idx:])
        np.fill_diagonal(cross_cov[self.block_idx:], 0.)
        N = x.shape[0]
        C1 = (x[:,self.block_idx:]-x[:,self.block_idx:].mean(0,keepdims=True)).T
        C2 = x-x.mean(0,keepdims=True)
        s = np.zeros_like(x)
        s[:,:self.block_idx] = cross_cov[:self.block_idx].dot(C1).T/N
        s[:,self.block_idx:] = C2.dot(cross_cov)/N
        return s
