"""
Physics-soft training / model selection.
Objective:  NLL + λ · PDE_residual  (steady advection–diffusion without time term)
"""

from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from .kernels import FlowAlignedRBF, SumKernel, BarrierKernel
from ..utils.physics import pde_residual_steady

def build_phys_kernel(angle_rad: float,
                      coast_poly=None,
                      la_small=8.0, lc_small=3.0, s2_small=1.0,
                      la_large=25.0, lc_large=12.0, s2_large=0.5,
                      w1=0.7, w2=0.3, barrier_lambda=2.5):
    k_s = FlowAlignedRBF(la_small, lc_small, angle_rad=angle_rad, sigma2=s2_small)
    k_l = FlowAlignedRBF(la_large, lc_large, angle_rad=angle_rad, sigma2=s2_large)
    k_mix = SumKernel(k_s, k_l, w1=w1, w2=w2)
    return BarrierKernel(k_mix, coast_poly=coast_poly, lam=barrier_lambda)

def fit_baseline(X, y, alpha=1e-6):
    gp = GaussianProcessRegressor(kernel=1.0*RBF(length_scale=[20.0, 20.0]), alpha=float(alpha),
                                  optimizer=None, normalize_y=True, random_state=0)
    gp.fit(X, y)
    return gp

def grid_search_phys(X_train, y_train, X_eval, grid_ijxy, vxy, kappa,
                     coast_poly=None, alpha_obs=1e-6, lam_phys=0.2, angle_deg0=30.0):
    """
    Search around the flow angle and scale multipliers to minimize:
        total = NLL + lam_phys * RMSE(pde_residual_steady)
    grid_ijxy: (nx, ny, arrays of x,y) for residual computation
    vxy: (vx, vy)
    """
    flows = np.deg2rad([angle_deg0-10, angle_deg0, angle_deg0+10])
    mults = [0.7, 1.0, 1.3]
    best = None
    xs, ys = grid_ijxy[2], grid_ijxy[3]  # x_grid, y_grid

    for ang in flows:
        for m1 in mults:
            for m2 in mults:
                for m3 in mults:
                    la_s, lc_s = 8.0*m1, 3.0*m2
                    la_l, lc_l = 25.0*m1, 12.0*m2
                    k = build_phys_kernel(ang, coast_poly=coast_poly,
                                          la_small=la_s, lc_small=lc_s,
                                          la_large=la_l, lc_large=lc_l,
                                          w1=0.7, w2=0.3, barrier_lambda=2.5)
                    gp = GaussianProcessRegressor(kernel=k, alpha=float(alpha_obs),
                                                  optimizer=None, normalize_y=True, random_state=0)
                    gp.fit(X_train, y_train)
                    nll = -gp.log_marginal_likelihood_value_

                    mu = gp.predict(X_eval)
                    # project mean to grid for PDE residual
                    nx, ny = grid_ijxy[0], grid_ijxy[1]
                    Z = mu.reshape(ny, nx)
                    R = pde_residual_steady(Z, xs, ys, vx=vxy[0], vy=vxy[1], kappa=kappa)
                    phys = float(np.sqrt(np.nanmean(R**2)))
                    total = float(nll + lam_phys*phys)

                    if (best is None) or (total < best["total"]):
                        best = {"gp": gp, "nll": float(nll), "phys": phys, "total": total,
                                "angle_rad": float(ang), "la_s": la_s, "lc_s": lc_s, "la_l": la_l, "lc_l": lc_l}
    return best
