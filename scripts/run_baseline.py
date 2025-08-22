"""
Anisotropic GP baseline
Task: predict a flow-driven pollutant concentration field in 2D
"""

from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_triangular

from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, Kernel, Sum, Product
)
from sklearn.model_selection import LeaveOneOut

# 1 . Geometry: rotate into streamwise/crosswise coordinates
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.geo import rotate_to_flow


# 2 . Probabilistic score: CRPS
def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Pointwise CRPS for N(mu, sigma^2); sigma must be observation std (includes noise)."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


# 3 . Kernel/noise utilities
def _sum_white_noise_variance(k: Kernel) -> float:
    """Sum WhiteKernel.noise_level across a kernel tree (Sum/Product)."""
    if isinstance(k, WhiteKernel):
        return float(k.noise_level)
    if isinstance(k, Sum):
        return _sum_white_noise_variance(k.k1) + _sum_white_noise_variance(k.k2)
    if isinstance(k, Product):
        return _sum_white_noise_variance(k.k1) + _sum_white_noise_variance(k.k2)
    return 0.0


def _get_y_mean_std(gp: GaussianProcessRegressor) -> Tuple[float, float]:
    """Fetch training y mean/std saved by sklearn when normalize_y=True (robust to version)."""
    y_mean, y_std = 0.0, 1.0
    for m in ("_y_train_mean", "y_train_mean_", "y_train_mean"):
        if hasattr(gp, m): y_mean = float(getattr(gp, m)); break
    for s in ("_y_train_std", "y_train_std_", "y_train_std"):
        if hasattr(gp, s): y_std = float(getattr(gp, s)); break
    return y_mean, y_std


def _predict_mean_std_y(gp: GaussianProcessRegressor, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and observation std on original scale:
    sigma_y = sqrt( sigma_f^2 (original) + (y_std^2)*(WhiteKernel + alpha) )
    """
    mu, std_f = gp.predict(X, return_std=True)  # std_f already on original scale
    _, y_std = _get_y_mean_std(gp)
    noise_var_norm = _sum_white_noise_variance(gp.kernel_) + float(gp.alpha)  # normalized scale
    noise_var = (y_std ** 2) * noise_var_norm
    std_y = np.sqrt(std_f**2 + noise_var)
    return mu, std_y


# 4 . Build anisotropic GP (ARD-RBF)
def build_gp_aniso(length_parallel: float,
                   length_cross: float,
                   alpha: float,
                   optimizer: bool,
                   n_restarts: int = 1,
                   noise_level: float = 1e-3,
                   learn_noise: bool = False) -> GaussianProcessRegressor:
    """ARD RBF with [l_parallel, l_perp]; optional L-BFGS-B and noise learning."""
    noise_bounds = (1e-6, 1e-1) if learn_noise else "fixed"
    kernel = (
        C(1.0, (1e-3, 1e3)) *
        RBF(length_scale=[length_parallel, length_cross],
            length_scale_bounds=(1e-2, 1e3))
        +
        WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_bounds)
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(alpha),
        normalize_y=True,
        optimizer=('fmin_l_bfgs_b' if optimizer else None),
        n_restarts_optimizer=(int(n_restarts) if optimizer else 0),
        random_state=0
    )


# 5 . LOOCV (slow & fast)
def loocv_metrics(Xp: np.ndarray, y: np.ndarray, gp_proto: GaussianProcessRegressor) -> dict:
    """Slow LOOCV: refit per left-out point; CRPS uses observation std (includes noise)."""
    loo = LeaveOneOut()
    yhat = np.zeros_like(y, dtype=float)
    ystd = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(Xp):
        gp = clone(gp_proto).set_params(optimizer=None, normalize_y=gp_proto.normalize_y)
        gp.fit(Xp[tr], y[tr])
        mu, std_y = _predict_mean_std_y(gp, Xp[te])
        yhat[te], ystd[te] = mu, std_y
    mae  = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    crps = float(np.mean(crps_gaussian(y, yhat, ystd)))
    return {"MAE": mae, "RMSE": rmse, "CRPS": crps}


def loocv_metrics_fast(Xp: np.ndarray, y: np.ndarray, gp_proto: GaussianProcessRegressor) -> dict:
    """Fast LOOCV: single Cholesky; restore to original scale when normalize_y=True."""
    gp = clone(gp_proto).set_params(optimizer=None, normalize_y=gp_proto.normalize_y)
    gp.fit(Xp, y)
    L = gp.L_                             # K_alpha = L @ L.T in normalized space
    alpha_vec = gp.alpha_                 # = K_alpha^{-1} y_norm
    y_mean, y_std = _get_y_mean_std(gp)

    I = np.eye(L.shape[0])
    L_inv = solve_triangular(L, I, lower=True, check_finite=False)
    Kinv_diag = np.sum(L_inv**2, axis=1)  # diag(K_alpha^{-1}) in normalized space
    y_norm = (y - y_mean) / y_std
    mu0_loo = y_norm - alpha_vec / Kinv_diag
    var0_y_loo = 1.0 / Kinv_diag
    # Back to original scale
    mu_loo    = mu0_loo * y_std + y_mean
    std_y_loo = np.sqrt(var0_y_loo) * y_std

    mae  = float(np.mean(np.abs(mu_loo - y)))
    rmse = float(np.sqrt(np.mean((mu_loo - y)**2)))
    crps = float(np.mean(crps_gaussian(y, mu_loo, std_y_loo)))
    return {"MAE": mae, "RMSE": rmse, "CRPS": crps}


# 6 . Grid prediction & visualization
def predict_on_grid(gp: GaussianProcessRegressor,
                    grid_df: pd.DataFrame,
                    vx: float, vy: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
    Xg = grid_df[["x", "y"]].to_numpy()
    Xgp = rotate_to_flow(Xg, vx, vy)
    mu, std_y = _predict_mean_std_y(gp, Xgp)
    nx = int(grid_df["i"].max() + 1)
    ny = int(grid_df["j"].max() + 1)
    return mu.reshape(ny, nx), std_y.reshape(ny, nx), nx, ny


def save_heatmap(Z: np.ndarray, pts_df: pd.DataFrame, grid_df: pd.DataFrame,
                 title: str, path: Path) -> None:
    plt.figure()
    plt.imshow(Z, origin="lower",
               extent=[grid_df["x"].min(), grid_df["x"].max(),
                       grid_df["y"].min(), grid_df["y"].max()])
    if pts_df is not None:
        plt.scatter(pts_df["x"], pts_df["y"], s=12, alpha=0.8,
                    edgecolor="k", linewidths=0.2)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


# 7 . Main
def main():
    ap = argparse.ArgumentParser(description="Anisotropic GP baseline")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--length_parallel", type=float, default=30.0)
    ap.add_argument("--length_cross", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--noise_level", type=float, default=1e-3)
    ap.add_argument("--learn_noise", action="store_true")
    ap.add_argument("--no_opt", action="store_true")
    ap.add_argument("--n_restarts", type=int, default=2)
    ap.add_argument("--fast_loo", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # robust read
    pts_path  = data_dir / "synth_points.csv"
    grid_path = data_dir / "grid_coords.csv"
    meta_path = data_dir / "flow_meta.json"
    if not (pts_path.exists() and grid_path.exists() and meta_path.exists()):
        raise SystemExit("[ERR] missing data/*.csv or flow_meta.json; run scripts/generate_synth.py first.")

    pts  = pd.read_csv(pts_path)
    grid = pd.read_csv(grid_path)
    meta = json.loads(meta_path.read_text())
    vx = float(meta.get("vx", 0.0)); vy = float(meta.get("vy", 0.0))
    if abs(vx) + abs(vy) < 1e-12:
        print("[WARN] flow vector is zero; stream/cross frame is undefined.")

    X = pts[["x", "y"]].to_numpy()
    y = pts["z"].to_numpy()
    Xp = rotate_to_flow(X, vx, vy)

    # Validation prototype (fixed hyperparameters)
    gp_proto = build_gp_aniso(
        length_parallel=args.length_parallel,
        length_cross=args.length_cross,
        alpha=args.alpha,
        optimizer=False,
        n_restarts=0,
        noise_level=args.noise_level,
        learn_noise=args.learn_noise
    )

    # LOOCV metrics
    metrics = (loocv_metrics_fast if args.fast_loo else loocv_metrics)(Xp, y, gp_proto)
    print(f"[LOOCV] MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  CRPS={metrics['CRPS']:.4f}")

    # Final model (optionally optimize hyperparameters)
    gp_final = build_gp_aniso(
        length_parallel=args.length_parallel,
        length_cross=args.length_cross,
        alpha=args.alpha,
        optimizer=(not args.no_opt),
        n_restarts=args.n_restarts,
        noise_level=args.noise_level,
        learn_noise=args.learn_noise
    )
    gp_final.fit(Xp, y)

    # Predict on grid
    Zm, Zs, nx, ny = predict_on_grid(gp_final, grid, vx, vy)

    # Export
    out_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(Zm, pts, grid, "Posterior Mean (anisotropic, flow-aligned)", out_dir / "mean_map.png")
    save_heatmap(Zs, pts, grid, "Posterior Std (uncertainty, includes noise)", out_dir / "std_map.png")
    
    save_heatmap(Zm, pts, grid, "Posterior Mean (isotropic baseline)", out_dir / "mean_iso.png")
    save_heatmap(Zs, pts, grid, "Posterior Std (isotropic baseline)", out_dir / "std_iso.png")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    df_pred = pd.DataFrame({"x": grid["x"], "y": grid["y"], "mean": Zm.ravel(), "std_y": Zs.ravel()})
    (data_dir / "grid_pred.csv").write_text(df_pred.to_csv(index=False))

    # Log final kernel and LML (useful for reproducibility)
    print(f"[INFO] Kernel (final): {gp_final.kernel_}")
    if hasattr(gp_final, "log_marginal_likelihood_value_"):
        print(f"[INFO] log_marginal_likelihood={gp_final.log_marginal_likelihood_value_:.3f}")
    print(f"[OK] Saved figures to {out_dir} and grid predictions to {data_dir/'grid_pred.csv'}")


if __name__ == "__main__":
    main()
