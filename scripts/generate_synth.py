import argparse, json
import numpy as np, pandas as pd
from pathlib import Path

def make_field(xx, yy, vx, vy, seed=0):
    rng = np.random.default_rng(seed)
    # Flow-aligned coordinates (approx): project onto flow unit vectors
    v = np.array([vx, vy]); v = v / (np.linalg.norm(v) + 1e-12)
    ex, ey = v, np.array([-v[1], v[0]])
    # Sources elongated along-flow
    s1 = np.array([30.0, 40.0]); Lpar1, Lperp1 = 35.0, 10.0
    s2 = np.array([65.0, 55.0]); Lpar2, Lperp2 = 25.0, 8.0
    def gauss_elong(xy, s, Lp, Lt, amp):
        d = xy - s
        s_coord = d @ ex; t_coord = d @ ey
        return amp * np.exp(-(s_coord/Lp)**2 - (t_coord/Lt)**2)
    XY = np.stack([xx, yy], axis=-1)
    z = gauss_elong(XY, s1, Lpar1, Lperp1, 1.0) + 0.6*gauss_elong(XY, s2, Lpar2, Lperp2, 1.0)
    # Add gentle downstream trend
    scoord = (XY @ ex).astype(float)
    z += 0.002 * scoord
    return z

def main(args):
    out_data = Path(args.out_dir); out_data.mkdir(parents=True, exist_ok=True)
    # Domain/grid
    L = args.grid
    xs = np.linspace(0, 100, L)
    ys = np.linspace(0, 100, L)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')

    vx, vy = args.vx, args.vy
    z = make_field(xx, yy, vx, vy, seed=args.seed)

    # Sample observation points
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(L*L, size=args.n_obs, replace=False)
    xi = idx % L; yi = idx // L
    x_obs = xs[xi]; y_obs = ys[yi]
    z_obs = z[yi, xi] + rng.normal(0.0, args.noise, size=args.n_obs)

    # Save training points
    pts = pd.DataFrame({'x': x_obs, 'y': y_obs, 'z': z_obs})
    pts.to_csv(out_data / 'synth_points.csv', index=False)

    # Save grid coords
    gi, gj = np.meshgrid(np.arange(L), np.arange(L), indexing='xy')
    grid = pd.DataFrame({'i': gi.ravel(), 'j': gj.ravel(), 'x': xx.ravel(), 'y': yy.ravel()})
    grid.to_csv(out_data / 'grid_coords.csv', index=False)

    # Metadata
    meta = {'vx': vx, 'vy': vy, 'grid': L, 'noise': args.noise, 'seed': args.seed}
    (out_data / 'flow_meta.json').write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(pts)} obs to {out_data/'synth_points.csv'} with flow (vx, vy)=({vx}, {vy}) and grid {L}x{L}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_obs', type=int, default=40)
    p.add_argument('--grid', type=int, default=100)
    p.add_argument('--noise', type=float, default=0.1)
    p.add_argument('--vx', type=float, default=1.0)
    p.add_argument('--vy', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='data')
    args = p.parse_args()
    main(args)
