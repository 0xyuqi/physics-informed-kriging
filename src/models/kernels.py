"""
Coastal kernels in flow frame.
FlowAlignedRBF, FlowAlignedRQ, FlowAlignedMatern32, MixKernel3, ShorelineAwareBarrier.
"""
from __future__ import annotations
import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from shapely.geometry import LineString, Polygon

def _segment_distance_to_boundary(p, q, coast: Polygon) -> float:
    seg = LineString([tuple(p), tuple(q)])
    return float(seg.distance(coast.boundary))

def _crosses_land(p, q, coast: Polygon) -> bool:
    if coast is None: return False
    ls = LineString([tuple(p), tuple(q)])
    return ls.crosses(coast) or ls.within(coast)

class _FlowFrameMixin:
    def __init__(self, angle_rad: float):
        self.theta = float(angle_rad)
    def _to_flow(self, X):
        X=np.asarray(X, float); x=X[:,0]; y=X[:,1]
        c,s=np.cos(self.theta), np.sin(self.theta)
        a =  c*x + s*y
        c0 = -s*x + c*y
        return np.stack([a, c0], 1)

class FlowAlignedRBF(Kernel, _FlowFrameMixin):
    def __init__(self, length_along=12.0, length_cross=4.0, angle_rad: float = 0.0, sigma2=1.0):
        _FlowFrameMixin.__init__(self, angle_rad)
        self.la=float(length_along); self.lc=float(length_cross); self.s2=float(sigma2)
    def __call__(self, X, Y=None, eval_gradient=False):
        A = self._to_flow(X); B = A if Y is None else self._to_flow(Y)
        d1 = A[:,[0]] - B[:,0][None,:]
        d2 = A[:,[1]] - B[:,1][None,:]
        sq = (d1**2)/(self.la**2) + (d2**2)/(self.lc**2)
        K = self.s2*np.exp(-0.5*sq)
        if eval_gradient: return K, np.empty((K.shape[0], K.shape[1], 0))
        return K
    def diag(self, X): return np.full(X.shape[0], self.s2, float)
    def is_stationary(self): return True

class FlowAlignedRQ(Kernel, _FlowFrameMixin):
    """
    RationalQuadratic in flow frame: k = s2 * (1 + r/(2*alpha))^{-alpha}
    Multi-scale behavior in one kernel.
    """
    def __init__(self, length_along=18.0, length_cross=6.0, alpha=1.2, angle_rad: float = 0.0, sigma2=1.0):
        _FlowFrameMixin.__init__(self, angle_rad)
        self.la=float(length_along); self.lc=float(length_cross); self.alpha=float(alpha); self.s2=float(sigma2)
    def __call__(self, X, Y=None, eval_gradient=False):
        A = self._to_flow(X); B = A if Y is None else self._to_flow(Y)
        d1 = A[:,[0]] - B[:,0][None,:]
        d2 = A[:,[1]] - B[:,1][None,:]
        r = (d1**2)/(self.la**2) + (d2**2)/(self.lc**2)
        base = (1.0 + 0.5*r/np.maximum(self.alpha,1e-9)) ** (-self.alpha)
        K = self.s2*base
        if eval_gradient: return K, np.empty((K.shape[0], K.shape[1], 0))
        return K
    def diag(self, X): return np.full(X.shape[0], self.s2, float)
    def is_stationary(self): return True

class FlowAlignedMatern32(Kernel, _FlowFrameMixin):
    """
    Matérn 3/2 in flow frame: k = s2 * (1 + sqrt(3) d) * exp(-sqrt(3) d)
    with anisotropic scaled distance d.
    """
    def __init__(self, length_along=14.0, length_cross=5.0, angle_rad: float = 0.0, sigma2=0.8):
        _FlowFrameMixin.__init__(self, angle_rad)
        self.la=float(length_along); self.lc=float(length_cross); self.s2=float(sigma2)
    def __call__(self, X, Y=None, eval_gradient=False):
        A = self._to_flow(X); B = A if Y is None else self._to_flow(Y)
        d1 = (A[:,[0]] - B[:,0][None,:])/self.la
        d2 = (A[:,[1]] - B[:,1][None,:])/self.lc
        d = np.sqrt(d1**2 + d2**2)
        cst = np.sqrt(3.0); base = (1.0 + cst*d)*np.exp(-cst*d)
        K = self.s2*base
        if eval_gradient: return K, np.empty((K.shape[0], K.shape[1], 0))
        return K
    def diag(self, X): return np.full(X.shape[0], self.s2, float)
    def is_stationary(self): return True

class MixKernel3(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel, k3: Kernel, w1=0.5, w2=0.3, w3=0.2):
        self.k1=k1; self.k2=k2; self.k3=k3
        s = float(w1)+float(w2)+float(w3)+1e-12
        self.w1=float(w1)/s; self.w2=float(w2)/s; self.w3=float(w3)/s
    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.w1*self.k1(X,Y,False) + self.w2*self.k2(X,Y,False) + self.w3*self.k3(X,Y,False)
        if eval_gradient: return K, np.empty((K.shape[0], K.shape[1], 0))
        return K
    def diag(self, X): return self.w1*self.k1.diag(X) + self.w2*self.k2.diag(X) + self.w3*self.k3.diag(X)
    def is_stationary(self): return self.k1.is_stationary() and self.k2.is_stationary() and self.k3.is_stationary()

class ShorelineAwareBarrier(Kernel):
    """
    Two effects:
    1) crossing land -> multiply by exp(-lam_cross)
    2) near shore    -> multiply by exp(-eta / (d + eps)) where d is distance to coastline
    """
    def __init__(self, base: Kernel, coast_poly: Polygon=None, lam_cross: float=3.2, eta: float=0.6, eps: float=1.0):
        self.base=base; self.coast=coast_poly
        self.lam=float(lam_cross); self.eta=float(eta); self.eps=float(eps)
    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.base(X, Y, False)
        if self.coast is None:
            return K if not eval_gradient else (K, np.empty((K.shape[0],K.shape[1],0)))
        if Y is None: Y = X
        X = np.asarray(X); Y = np.asarray(Y)
        pen = np.ones_like(K)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                p = X[i,:2]; q = Y[j,:2]
                if _crosses_land(p, q, self.coast):
                    pen[i,j] *= np.exp(-self.lam)
                if self.eta > 0.0:
                    d = _segment_distance_to_boundary(p, q, self.coast)
                    pen[i,j] *= np.exp(-self.eta / (d + self.eps))
        Kb = K * pen
        if eval_gradient: return Kb, np.empty((Kb.shape[0], Kb.shape[1], 0))
        return Kb
    def diag(self, X): return self.base.diag(X)
    def is_stationary(self): return False
