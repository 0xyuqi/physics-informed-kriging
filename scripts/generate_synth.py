"""
Generate a competition-ready synthetic plume dataset
Output: data/synth_points.csv, data/grid_coords.csv, data/flow_meta.json
"""

import argparse, json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# Construct orthogonal basis, decompose main flow transport
def flow_basis(vx: float, vy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    ex: unit vector along flow (downstream)
    ey: unit vector cross to flow (lateral)
    """
    v = np.array([vx, vy], dtype=float)
    n = np.linalg.norm(v) + 1e-12
    ex = v / n
    ey = np.array([-ex[1], ex[0]], dtype=float)  # 90° CCW rotation of ex
    return ex, ey


# Gaussian plume
"""
1. Gaussian shape:
In fluid, the concentration field of pollutants evolves over time,
mainly controlled by advection (transport by water flow) + dispersion (random disturbance).
Corresponds to the advection-diffusion equation, whose analytical solution is a Gaussian function.

2. Elliptical Gaussian: advection + anisotropic dispersion
"""
def gauss_elong(xy: np.ndarray, center: np.ndarray,
                ex: np.ndarray, ey: np.ndarray,
                L_par: float, L_perp: float, amp: float = 1.0):
    d = xy - center
    s = d @ ex
    t = d @ ey                             # Transform original coordinates into flow-aligned (s) and cross-flow (t)
    return amp * np.exp(-(s/L_par)**2 - (t/L_perp)**2)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # grid
    L = int(args.grid)
    xs = np.linspace(0.0, 100.0, L)
    ys = np.linspace(0.0, 100.0, L)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    XY = np.stack([xx, yy], axis=-1)

    ex, ey = flow_basis(args.vx, args.vy)   # Construct orthogonal basis vectors according to velocity components

    # Two elongated elliptical Gaussian plumes in the flow direction,
    # one strong & large, one weaker & smaller, more realistic
    s1, Lp1, Lt1, a1 = np.array([30.0, 40.0]), 35.0, 10.0, 1.0
    s2, Lp2, Lt2, a2 = np.array([65.0, 55.0]), 25.0, 8.0, 0.6

    z = gauss_elong(XY, s1, ex, ey, Lp1, Lt1, a1) + gauss_elong(XY, s2, ex, ey, Lp2, Lt2, a2)
    s_coord = (XY @ ex).astype(float)       # Flow-aligned coordinate s for each grid point
    z = z + 0.002 * s_coord                 # Add a linear background along flow

    # sample observations with noise
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(L * L, size=int(args.n_obs), replace=False)
    ix = idx % L
    iy = idx // L
    x_obs = xs[ix]
    y_obs = ys[iy]
    z_obs = z[iy, ix] + rng.normal(0.0, float(args.noise), size=int(args.n_obs))

    # --- Output files ---
    # 1. Observation points CSV
    df_pts = pd.DataFrame({"x": x_obs, "y": y_obs, "z": z_obs})
    (out_dir/"synth_points.csv").write_text(df_pts.to_csv(index=False))

    # 2. Grid coordinates CSV
    gi, gj = np.meshgrid(np.arange(L), np.arange(L), indexing="xy")
    df_grid = pd.DataFrame({
        "i": gi.ravel(), "j": gj.ravel(),
        "x": xx.ravel(), "y": yy.ravel()
    })
    (out_dir / "grid_coords.csv").write_text(df_grid.to_csv(index=False))

    # 3. Metadata JSON (flow direction, grid, noise, random seed)
    meta = {"vx": float(args.vx), "vy": float(args.vy),
            "grid": int(L), "noise": float(args.noise), "seed": int(args.seed)}
    (out_dir/"flow_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[OK] Saved {args.n_obs} obs to {out_dir/'synth_points.csv'}")
    print(f"[OK] Grid {L}x{L} at {out_dir/'grid_coords.csv'}")
    print(f"[OK] Flow (vx, vy)=({args.vx}, {args.vy}), noise={args.noise}, seed={args.seed}")


# Command line arguments
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs", type=int, default=40)
    ap.add_argument("--grid", type=int, default=80)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--vy", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()
    main(args)
