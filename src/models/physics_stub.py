"""Templates for PDE soft constraints (to extend in Day 2-3).
Idea: Add virtual observations enforcing L[u](x)=0 with noise, where
L is the steady advection-diffusion operator:
    L[u] = vx * du/dx + vy * du/dy - kappa * (d2u/dx2 + d2u/dy2)
Approach A (simple): finite-difference residuals on a collocation grid.
Approach B (exact): use kernel derivatives of GP to form residual covariances (advanced).
"""

from dataclasses import dataclass

@dataclass
class PDEConfig:
    vx: float
    vy: float
    kappa: float = 1.0
    noise: float = 0.1  # residual noise for virtual obs

def sample_collocation_points(xmin, xmax, ymin, ymax, n):
    import numpy as np
    xs = np.random.uniform(xmin, xmax, size=n)
    ys = np.random.uniform(ymin, ymax, size=n)
    return np.column_stack([xs, ys])

# TODO: Implement:
# 1) Build finite-difference stencil for L[u] at collocation points.
# 2) Form augmented dataset: [real points; collocation points] with targets [z; 0] and per-point noise.
# 3) Fit GP normally; uncertainty should shrink where physics is informative.
