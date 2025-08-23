"""
Cost-aware active sampling (hotspot-first).
Score = w_straddle*(k*std - |mu - tau|) + w_mi*log(std^2 + eps) - lam_cost * travel_time
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def straddle(mu, std, tau, k=1.96):
    return k*np.asarray(std) - np.abs(np.asarray(mu)-float(tau))

def info_gain_approx(std, eps=1e-6):
    s2 = np.asarray(std, float)**2
    return np.log(s2 + eps)

def route_cost(points, port=(0.0,0.0), speed=10.0):
    p=np.asarray(points,float); d=np.sqrt((p[:,0]-port[0])**2 + (p[:,1]-port[1])**2)
    return d/max(speed,1e-6)

def pick_next_points(cands_xy, mu, std, tau, k_pick=20, min_dist=6.0,
                     port=(0.0,0.0), lam_cost=0.3, w_straddle=0.7, w_mi=0.3):
    s = straddle(mu, std, tau)
    mi = info_gain_approx(std)
    cost = route_cost(cands_xy, port=tuple(port))
    score = w_straddle*s + w_mi*mi - lam_cost*cost
    order = np.argsort(-score)
    chosen = []
    for idx in order:
        if len(chosen) >= int(k_pick): break
        p = cands_xy[idx]
        if not chosen: chosen.append(p); continue
        if cdist(np.array(chosen), p[None,:]).min() >= float(min_dist):
            chosen.append(p)
    return np.array(chosen)
