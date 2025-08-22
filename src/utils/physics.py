"""
PDE residuals on a regular grid (steady advection–diffusion).
Input Z: (ny, nx), xs: (nx,), ys: (ny,)
Residual: v·∇Z - κ ΔZ
"""
from __future__ import annotations
import numpy as np

def pde_residual_steady(Z, xs, ys, vx=0.4, vy=0.1, kappa=0.25):
    Z = np.asarray(Z, float); xs=np.asarray(xs,float); ys=np.asarray(ys,float)
    ny, nx = Z.shape
    dx = float(xs[1]-xs[0]) if nx>1 else 1.0
    dy = float(ys[1]-ys[0]) if ny>1 else 1.0

    # first derivatives
    fx = np.zeros_like(Z); fy = np.zeros_like(Z)
    fx[:,1:-1] = (Z[:,2:] - Z[:,:-2])/(2*dx)
    fx[:,0] = (Z[:,1] - Z[:,0])/dx
    fx[:,-1] = (Z[:,-1] - Z[:,-2])/dx
    fy[1:-1,:] = (Z[2:,:] - Z[:-2,:])/(2*dy)
    fy[0,:] = (Z[1,:] - Z[0,:])/dy
    fy[-1,:] = (Z[-1,:] - Z[-2,:])/dy

    # second derivatives
    fxx = np.zeros_like(Z); fyy = np.zeros_like(Z)
    fxx[:,1:-1] = (Z[:,2:] - 2*Z[:,1:-1] + Z[:,:-2])/(dx*dx)
    fxx[:,0] = (Z[:,1] - 2*Z[:,0] + Z[:,0])/(dx*dx)
    fxx[:,-1] = (Z[:,-1] - 2*Z[:,-1] + Z[:,-2])/(dx*dx)
    fyy[1:-1,:] = (Z[2:,:] - 2*Z[1:-1,:] + Z[:-2,:])/(dy*dy)
    fyy[0,:] = (Z[1,:] - 2*Z[0,:] + Z[0,:])/(dy*dy)
    fyy[-1,:] = (Z[-1,:] - 2*Z[-1,:] + Z[-2,:])/(dy*dy)

    return vx*fx + vy*fy - kappa*(fxx + fyy)
