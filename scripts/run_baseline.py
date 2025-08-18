import argparse, json
import numpy as np
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold
from src.utils.geo import rotate_to_flow

def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))

def load_data(data_dir: Path):
    pts = pd.read_csv(data_dir / "synth_points.csv")
    meta = json.loads((data_dir / "flow_meta.json").read_text())
    grid = pd.read_csv(data_dir / "grid_coords.csv")
    return pts, meta, grid


def build_gp(length_parallel, length_cross, alpha, optimizer=True, n_restarts=0):
    # ARD RBF in 2D (after rotation): length_scales initialized but learnable
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[length_parallel, length_cross], length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, optimizer=('fmin_l_bfgs_b' if optimizer else None), n_restarts_optimizer=n_restarts)
    return gp

def main(args):
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts, meta, grid = load_data(data_dir)
    vx, vy = meta['vx'], meta['vy']

    X_obs = pts[['x','y']].to_numpy()
    y_obs = pts['z'].to_numpy()

    # Rotate coords into (along, across)-flow
    Xp_obs = rotate_to_flow(X_obs, vx, vy)

    gp = build_gp(args.length_parallel, args.length_cross, args.alpha, optimizer=(not args.no_opt), n_restarts=1)
    gp.fit(Xp_obs, y_obs)

    # LOOCV (fast KFold with k=n, but use predictions on held-out folds)
    kf = KFold(n_splits=len(X_obs))
    y_pred_cv, y_std_cv = np.zeros_like(y_obs), np.zeros_like(y_obs)
    for train_idx, test_idx in kf.split(Xp_obs):
        gp_cv = build_gp(args.length_parallel, args.length_cross, args.alpha, optimizer=False, n_restarts=0)
        gp_cv.fit(Xp_obs[train_idx], y_obs[train_idx])
        mu, std = gp_cv.predict(Xp_obs[test_idx], return_std=True)
        y_pred_cv[test_idx] = mu
        y_std_cv[test_idx] = std

    mae = np.mean(np.abs(y_pred_cv - y_obs))
    rmse = np.sqrt(np.mean((y_pred_cv - y_obs)**2))
    crps = np.mean(crps_gaussian(y_obs, y_pred_cv, y_std_cv))

    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  CRPS={crps:.3f}")
    metrics = {'MAE': float(mae), 'RMSE': float(rmse), 'CRPS': float(crps)}
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # Predict on grid
    X_grid = grid[['x','y']].to_numpy()
    Xp_grid = rotate_to_flow(X_grid, vx, vy)
    mu, std = gp.predict(Xp_grid, return_std=True)

    # Save grid predictions
    grid_out = grid.copy()
    grid_out['mean'] = mu
    grid_out['std'] = std
    grid_out.to_csv(out_dir.parent / 'data' / 'grid_pred.csv', index=False)

    # Plots (single plot per figure, default colors)
    nx = int(grid['i'].max()+1)
    ny = int(grid['j'].max()+1)
    Zm = mu.reshape(ny, nx)
    Zs = std.reshape(ny, nx)

    plt.figure()
    plt.imshow(Zm, origin='lower', extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()])
    plt.scatter(pts['x'], pts['y'], s=10, alpha=0.7)
    plt.title('Posterior Mean (with training points)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / 'mean_map.png', dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(Zs, origin='lower', extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()])
    plt.scatter(pts['x'], pts['y'], s=10, alpha=0.7)
    plt.title('Posterior Std (uncertainty)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / 'std_map.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--out_dir', type=str, default='figures')
    p.add_argument('--length_parallel', type=float, default=30.0)
    p.add_argument('--length_cross', type=float, default=8.0)
    p.add_argument('--alpha', type=float, default=1e-6, help='Nugget for numerical stability')
    p.add_argument('--no_opt', action='store_true', help='Disable kernel hyperparameter optimization')
    args = p.parse_args()
    main(args)
