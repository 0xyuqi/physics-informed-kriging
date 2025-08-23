"""
Physics-soft GP: minimize total = NLL + lambda_phys * RMSE(PDE residual).
Kernel = Mix3(RBF, RQ, Matérn32) in flow frame + ShorelineAwareBarrier.
Heteroscedastic alpha depends on shoreline distance (nearshore noisier).
"""
from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .kernels import FlowAlignedRBF, FlowAlignedRQ, FlowAlignedMatern32, MixKernel3, ShorelineAwareBarrier
from ..utils.physics import pde_residual_steady
from shapely.geometry import Polygon
from ..utils.geo import dist_to_coast

def build_phys_kernel(angle_rad: float,
                      coast_poly=None,
                      la_s=8.0,  lc_s=3.0,  s2_s=1.0,
                      la_rq=22.0, lc_rq=9.0,  alpha_rq=1.2, s2_rq=0.8,
                      la_m32=14.0, lc_m32=5.0, s2_m32=0.6,
                      w1=0.5, w2=0.3, w3=0.2,
                      lam_cross=3.2, eta_shore=0.6, eps=1.0):
    k_rbf = FlowAlignedRBF(length_along=la_s, length_cross=lc_s, angle_rad=angle_rad, sigma2=s2_s)
    k_rq  = FlowAlignedRQ(length_along=la_rq, length_cross=lc_rq, alpha=alpha_rq, angle_rad=angle_rad, sigma2=s2_rq)
    k_m32 = FlowAlignedMatern32(length_along=la_m32, length_cross=lc_m32, angle_rad=angle_rad, sigma2=s2_m32)
    k_mix = MixKernel3(k_rbf, k_rq, k_m32, w1=w1, w2=w2, w3=w3)
    return ShorelineAwareBarrier(k_mix, coast_poly=coast_poly, lam_cross=lam_cross, eta=eta_shore, eps=eps)

def fit_baseline(X, y, alpha=1e-6):
    gp = GaussianProcessRegressor(kernel=1.0*RBF(length_scale=[20.0, 20.0]),
                                  alpha=float(alpha), optimizer=None,
                                  normalize_y=True, random_state=0)
    gp.fit(X, y); return gp

def make_hetero_alpha(X, coast_poly: Polygon, base=1e-6, near_coef=5.0, scale=8.0):
    """
    Per-sample alpha ~ base + near_coef/(d/scale + 1)
    Nearshore (small d) gets larger noise to reflect turbulent zones.
    """
    if coast_poly is None:
        return float(base)
    d = dist_to_coast(X, coast_poly)
    return base + near_coef / (d/scale + 1.0)

def grid_search_phys(X_train, y_train, X_eval, grid_ijxy, vxy, kappa,
                     coast_poly=None, alpha_obs=1e-6, lam_phys=0.25, angle_deg0=30.0):
    flows = np.deg2rad([angle_deg0-10, angle_deg0, angle_deg0+10])
    mults = [0.8, 1.0, 1.25]
    best = None
    nx, ny, xs, ys = grid_ijxy
    alpha_vec = make_hetero_alpha(X_train, coast_poly, base=alpha_obs, near_coef=2e-6, scale=6.0)
    for ang in flows:
        for m in mults:
            la_s  = 8.0*m;  lc_s  = 3.0*m
            la_rq = 22.0*m; lc_rq = 9.0*m
            la_m  = 14.0*m; lc_m  = 5.0*m
            k = build_phys_kernel(ang, coast_poly=coast_poly,
                                  la_s=la_s, lc_s=lc_s, s2_s=1.0,
                                  la_rq=la_rq, lc_rq=lc_rq, alpha_rq=1.2, s2_rq=0.8,
                                  la_m32=la_m, lc_m32=lc_m, s2_m32=0.6,
                                  w1=0.5, w2=0.3, w3=0.2, lam_cross=3.2, eta_shore=0.6, eps=1.0)
            gp = GaussianProcessRegressor(kernel=k, alpha=alpha_vec,
                                          optimizer=None, normalize_y=True, random_state=0)
            gp.fit(X_train, y_train)
            nll = -gp.log_marginal_likelihood_value_
            mu = gp.predict(X_eval).reshape(ny, nx)
            R = pde_residual_steady(mu, xs, ys, vx=vxy[0], vy=vxy[1], kappa=kappa)
            phys = float(np.sqrt(np.nanmean(R**2)))
            total = float(nll + lam_phys*phys)
            if (best is None) or (total < best["total"]):
                best = {"gp": gp, "nll": float(nll), "phys": phys, "total": total,
                        "angle_rad": float(ang), "la_s": la_s, "lc_s": lc_s,
                        "la_rq": la_rq, "lc_rq": lc_rq, "la_m": la_m, "lc_m": lc_m}
    return best
