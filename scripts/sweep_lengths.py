
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import KFold
from src.utils.geo import rotate_to_flow

def cv_rmse(lp, lc, Xp, y):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[lp, lc], length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    kf = KFold(n_splits=len(Xp))
    errs = []
    for tr, te in kf.split(Xp):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None)
        gp.fit(Xp[tr], y[tr])
        mu = gp.predict(Xp[te])
        errs.append((mu - y[te])**2)
    return float(np.sqrt(np.mean(np.concatenate(errs))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--out_csv', type=str, default='figures/length_sweep.csv')
    ap.add_argument('--lp_list', type=float, nargs='+', default=[20,30,40])
    ap.add_argument('--lc_list', type=float, nargs='+', default=[6,8,12])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pts = pd.read_csv(data_dir/'synth_points.csv')
    meta = json.loads((data_dir/'flow_meta.json').read_text())
    vx, vy = meta['vx'], meta['vy']

    X = pts[['x','y']].to_numpy(); y = pts['z'].to_numpy()
    Xp = rotate_to_flow(X, vx, vy)

    rows = []
    for lp in args.lp_list:
        for lc in args.lc_list:
            rmse = cv_rmse(lp, lc, Xp, y)
            rows.append({'length_parallel':lp, 'length_cross':lc, 'RMSE':rmse})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved sweep to {args.out_csv}")

if __name__ == '__main__':
    main()
