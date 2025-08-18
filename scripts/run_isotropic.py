
import argparse, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold

def crps_gaussian(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return sigma * (z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))

def load_data(data_dir: Path):
    pts = pd.read_csv(data_dir / "synth_points.csv")
    meta = json.loads((data_dir / "flow_meta.json").read_text())
    grid = pd.read_csv(data_dir / "grid_coords.csv")
    return pts, meta, grid

def build_gp(length_scale, alpha, optimizer=True, n_restarts=1):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, optimizer=('fmin_l_bfgs_b' if optimizer else None), n_restarts_optimizer=n_restarts)
    return gp

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    pts, meta, grid = load_data(data_dir)
    X_obs = pts[['x','y']].to_numpy()
    y_obs = pts['z'].to_numpy()

    gp = build_gp(args.length, args.alpha, optimizer=(not args.no_opt), n_restarts=1)

    # LOOCV
    kf = KFold(n_splits=len(X_obs))
    y_pred_cv, y_std_cv = np.zeros_like(y_obs), np.zeros_like(y_obs)
    for train_idx, test_idx in kf.split(X_obs):
        gp_cv = build_gp(args.length, args.alpha, optimizer=False, n_restarts=0)
        gp_cv.fit(X_obs[train_idx], y_obs[train_idx])
        mu, std = gp_cv.predict(X_obs[test_idx], return_std=True)
        y_pred_cv[test_idx] = mu; y_std_cv[test_idx] = std
    mae = float(np.mean(np.abs(y_pred_cv - y_obs)))
    rmse = float(np.sqrt(np.mean((y_pred_cv - y_obs)**2)))
    crps = float(np.mean(crps_gaussian(y_obs, y_pred_cv, y_std_cv)))
    metrics = {'MAE': mae, 'RMSE': rmse, 'CRPS': crps}
    (out_dir / 'metrics_iso.json').write_text(json.dumps(metrics, indent=2))
    print(f"ISO  MAE={mae:.3f} RMSE={rmse:.3f} CRPS={crps:.3f}")

    # Predict grid
    X_grid = grid[['x','y']].to_numpy()
    gp.fit(X_obs, y_obs)
    mu, std = gp.predict(X_grid, return_std=True)
    nx = int(grid['i'].max()+1); ny = int(grid['j'].max()+1)
    Zm = mu.reshape(ny, nx); Zs = std.reshape(ny, nx)
    plt.figure(); plt.imshow(Zm, origin='lower', extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()])
    plt.scatter(pts['x'], pts['y'], s=10, alpha=0.7); plt.title('Isotropic Mean'); plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir / 'mean_iso.png', dpi=150); plt.close()
    plt.figure(); plt.imshow(Zs, origin='lower', extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()])
    plt.scatter(pts['x'], pts['y'], s=10, alpha=0.7); plt.title('Isotropic Std'); plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir / 'std_iso.png', dpi=150); plt.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--out_dir', type=str, default='figures')
    p.add_argument('--length', type=float, default=12.0)
    p.add_argument('--alpha', type=float, default=1e-6)
    p.add_argument('--no_opt', action='store_true')
    args = p.parse_args()
    main(args)
