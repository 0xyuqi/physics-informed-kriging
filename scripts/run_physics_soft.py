
import argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold
from scipy.ndimage import convolve

import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]))
from src.utils.geo import rotate_to_flow

def finite_diffs_2d(field, dx, dy):
    # Central diff for first derivatives, 5-point Laplacian for second derivatives
    kx = np.array([[0,0,0],[ -0.5, 0, 0.5],[0,0,0]])
    ky = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]])
    kxx = np.array([[0,0,0],[1,-2,1],[0,0,0]])
    kyy = np.array([[0,1,0],[0,-2,0],[0,1,0]])
    ux = convolve(field, kx, mode='nearest')/dx
    uy = convolve(field, ky, mode='nearest')/dy
    uxx = convolve(field, kxx, mode='nearest')/(dx*dx)
    uyy = convolve(field, kyy, mode='nearest')/(dy*dy)
    return ux, uy, uxx, uyy

def build_gp_aniso(lp, lc, alpha):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[lp, lc], length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    return GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, optimizer=None, n_restarts_optimizer=0)

def cv_rmse(gp, X, y):
    kf = KFold(n_splits=len(X))
    errs = []
    for tr, te in kf.split(X):
        gp_cv = GaussianProcessRegressor(kernel=gp.kernel, alpha=gp.alpha, normalize_y=True, optimizer=None)
        gp_cv.fit(X[tr], y[tr])
        mu = gp_cv.predict(X[te])
        errs.append((mu - y[te])**2)
    return float(np.sqrt(np.mean(np.concatenate(errs))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--out_dir', type=str, default='figures')
    ap.add_argument('--length_parallel_list', type=float, nargs='+', default=[20,30,40])
    ap.add_argument('--length_cross_list', type=float, nargs='+', default=[6,8,12])
    ap.add_argument('--alpha', type=float, default=1e-6)
    ap.add_argument('--lambda_phys', type=float, default=1.0)
    ap.add_argument('--kappa', type=float, default=1.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(data_dir/'synth_points.csv')
    meta = json.loads((data_dir/'flow_meta.json').read_text())
    grid = pd.read_csv(data_dir/'grid_coords.csv')
    vx, vy = meta['vx'], meta['vy']

    X_obs = pts[['x','y']].to_numpy()
    y_obs = pts['z'].to_numpy()

    # Grid geometry
    xs = np.sort(grid['x'].unique()); ys = np.sort(grid['y'].unique())
    nx, ny = len(xs), len(ys)
    dx = xs[1]-xs[0]; dy = ys[1]-ys[0]
    Xg = grid[['x','y']].to_numpy()

    # Rotate coordinates once for aniso GP
    Xp_obs = rotate_to_flow(X_obs, vx, vy)
    Xp_grid = rotate_to_flow(Xg, vx, vy)

    rows = []
    best = None

    for lp in args.length_parallel_list:
        for lc in args.length_cross_list:
            gp = build_gp_aniso(lp, lc, args.alpha)
            # CV on rotated obs
            rmse = cv_rmse(gp, Xp_obs, y_obs)
            # Fit on all obs to get grid mean
            gp.fit(Xp_obs, y_obs)
            mu = gp.predict(Xp_grid).reshape(ny, nx)

            ux, uy, uxx, uyy = finite_diffs_2d(mu, dx, dy)
            resid = vx*ux + vy*uy - args.kappa*(uxx+uyy)
            phys_pen = float(np.sqrt(np.mean(resid**2)))  # RMS residual
            score = rmse + args.lambda_phys * phys_pen

            rows.append({'length_parallel':lp, 'length_cross':lc, 'RMSE':rmse, 'phys_pen':phys_pen, 'score':score})
            if (best is None) or (score < best['score']):
                best = {'lp':lp, 'lc':lc, 'rmse':rmse, 'phys_pen':phys_pen, 'score':score, 'mu':mu}

    df = pd.DataFrame(rows)
    df.to_csv(out_dir/'physics_soft_gridsearch.csv', index=False)

    # Save best mean map and residual map
    plt.figure()
    plt.imshow(best['mu'], origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    plt.scatter(pts['x'], pts['y'], s=10, alpha=0.7); plt.title(f'Physics-Soft Mean (lp={best["lp"]}, lc={best["lc"]})')
    plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir/'mean_physics_soft.png', dpi=150); plt.close()

    ux, uy, uxx, uyy = finite_diffs_2d(best['mu'], dx, dy)
    resid = vx*ux + vy*uy - args.kappa*(uxx+uyy)
    plt.figure()
    plt.imshow(np.abs(resid), origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    plt.title('Physics Residual |L[u]| (RMS={:.3f})'.format(best['phys_pen']))
    plt.colorbar(); plt.tight_layout()
    plt.savefig(out_dir/'physics_residual.png', dpi=150); plt.close()

    # Save summary metrics
    summary = {'best_length_parallel': best['lp'], 'best_length_cross': best['lc'], 'RMSE': best['rmse'], 'phys_pen': best['phys_pen'], 'score': best['score']}
    (out_dir/'metrics_physics_soft.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
