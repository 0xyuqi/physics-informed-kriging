"""
kernels.py — Advanced kernels for flow-driven fields.
- Anisotropic RBF aligned with flow (two scales)
- Time-free "advective" transform (rotation only, consistent with baseline)
- Barrier kernel to penalize cross-land correlations
"""

from __future__ import annotations
import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from shapely.geometry import LineString, Polygon

def _crosses_land(p, q, coast_poly: Polygon) -> bool:
    """Return True if segment p->q intersects the land polygon."""
    if coast_poly is None:
        return False
    ls = LineString([tuple(p), tuple(q)])
    return ls.crosses(coast_poly) or ls.within(coast_poly)

class FlowAlignedRBF(Kernel):
    """Anisotropic RBF in (along, cross) axes, obtained by rotating (x,y) by flow angle."""
    def __init__(self, length_along=12.0, length_cross=4.0, angle_rad: float = 0.0, sigma2=1.0):
        self.la=float(length_along); self.lc=float(length_cross)
        self.theta=float(angle_rad); self.s2=float(sigma2)

    def _to_flow(self, X):
        X=np.asarray(X, float); x=X[:,0]; y=X[:,1]
        c,s=np.cos(self.theta), np.sin(self.theta)
        # R: [x,y] -> [along, cross]
        a =  c*x + s*y
        c0 = -s*x + c*y
        return np.stack([a, c0], 1)

    def __call__(self, X, Y=None, eval_gradient=False):
        A = self._to_flow(X); B = A if Y is None else self._to_flow(Y)
        d1 = A[:,[0]] - B[:,0][None,:]
        d2 = A[:,[1]] - B[:,1][None,:]
        sq = (d1**2)/(self.la**2) + (d2**2)/(self.lc**2)
        K = self.s2*np.exp(-0.5*sq)
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):  # needed by sklearn
        return np.full(X.shape[0], self.s2, float)

    def is_stationary(self):  # due to rotation, still stationary in rotated frame
        return True

class SumKernel(Kernel):
    """k = w1*k1 + w2*k2 (two-scale mix)."""
    def __init__(self, k1: Kernel, k2: Kernel, w1=0.7, w2=0.3):
        self.k1=k1; self.k2=k2; self.w1=float(w1); self.w2=float(w2)

    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.w1*self.k1(X,Y,eval_gradient=False) + self.w2*self.k2(X,Y,eval_gradient=False)
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        return self.w1*self.k1.diag(X) + self.w2*self.k2.diag(X)

    def is_stationary(self):
        return self.k1.is_stationary() and self.k2.is_stationary()

class BarrierKernel(Kernel):
    """
    Multiply base kernel by exp(-lam) if the straight line between two points crosses land.
    """
    def __init__(self, base: Kernel, coast_poly: Polygon=None, lam: float=2.5):
        self.base=base; self.coast=coast_poly; self.lam=float(lam)

    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.base(X, Y, eval_gradient=False)
        if self.coast is None:
            return K if not eval_gradient else (K, np.empty((K.shape[0],K.shape[1],0)))
        if Y is None: Y = X
        X = np.asarray(X); Y = np.asarray(Y)
        pen = np.ones_like(K)
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if _crosses_land(X[i,:2], Y[j,:2], self.coast):
                    pen[i,j] = np.exp(-self.lam)
        Kb = K * pen
        if eval_gradient:
            return Kb, np.empty((Kb.shape[0], Kb.shape[1], 0))
        return Kb

    def diag(self, X): return self.base.diag(X)
    def is_stationary(self): return False
