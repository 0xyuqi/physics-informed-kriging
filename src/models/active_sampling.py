# cost-aware active design.
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def straddle(mu, std, tau, k=1.96):
    return k*np.asarray(std) - np.abs(np.asarray(mu)-float(tau))

def route_cost(points, port=(0.0,0.0), speed=10.0):
    p=np.asarray(points,float); d=np.sqrt((p[:,0]-port[0])**2 + (p[:,1]-port[1])**2)
    return d/max(speed,1e-6)

def pick_next_points(cands_xy, mu, std, tau, k_pick=20, min_dist=6.0, port=(0.0,0.0), lam_cost=0.3):
    base = straddle(mu, std, tau)
    cost = route_cost(cands_xy, port=tuple(port))
    score = base - lam_cost*cost
    order = np.argsort(-score)
    chosen = []
    for idx in order:
        if len(chosen) >= int(k_pick): break
        p = cands_xy[idx]
        if not chosen: chosen.append(p); continue
        if cdist(np.array(chosen), p[None,:]).min() >= float(min_dist):
            chosen.append(p)
    return np.array(chosen)
