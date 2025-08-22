# RMSE/CRPS and directional variogram.
from __future__ import annotations
import numpy as np
from scipy.stats import norm

def rmse(y_true, y_pred) -> float:
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true-y_pred)**2)))

def crps_gaussian(y, mu, sigma) -> float:
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z*(2.0*norm.cdf(z)-1.0) + 2.0*norm.pdf(z) - 1.0/np.sqrt(np.pi))))

def dir_variogram(x, y, vals, angle_rad, tol=np.deg2rad(15), nbins=20, hmax=60.0):
    x=np.asarray(x); y=np.asarray(y); v=np.asarray(vals)
    dx=x[:,None]-x[None,:]; dy=y[:,None]-y[None,:]; dh=np.sqrt(dx**2+dy**2)
    th=np.arctan2(dy,dx)
    dth=np.minimum.reduce([np.abs(th-angle_rad),np.abs(th-angle_rad+2*np.pi),np.abs(th-angle_rad-2*np.pi)])
    m=(dh>0)&(dh<=hmax)&(dth<=tol)
    if not np.any(m): return np.array([]), np.array([])
    h=dh[m].ravel(); g=(0.5*(v[:,None]-v[None,:])**2)[m].ravel()
    bins=np.linspace(0,hmax,nbins+1); centers=0.5*(bins[:-1]+bins[1:])
    gv=np.array([np.nanmean(g[(h>=bins[i])&(h<bins[i+1])]) if np.any((h>=bins[i])&(h<bins[i+1])) else np.nan for i in range(nbins)])
    return centers, gv
