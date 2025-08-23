"""
Physics-soft GP with Malaysia coastline barrier.
Includes: flow-frame Mix3 kernel, shoreline-aware penalty, heteroscedastic alpha, conformal CI, active sampling.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.tri as mtri
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.geo import prepare_malaysia_barrier, load_geojson_polygon, rotate_to_flow
from src.utils.metrics import rmse, crps_gaussian, dir_variogram
from src.utils.physics import pde_residual_steady
from src.models.phys_soft import build_phys_kernel, fit_baseline, grid_search_phys, make_hetero_alpha
from src.models.conformal import calibrate, coverage
from src.models.active_sampling import pick_next_points
from sklearn.gaussian_process import GaussianProcessRegressor

CFG = {
    "alpha_obs": 1e-6,
    "physics_lambda": 0.25,
    "kappa": 0.25,
    "angle_deg0": 30.0,
    "next_k": 20,
    "min_dist": 6.0,
    "port_xy": [20.0, 20.0],
    "lam_cost": 0.3,
    "simplify_tol": 0.02
}

def fit_trend_along(X, y, vx, vy):
    Xp = rotate_to_flow(X, vx, vy)
    along = Xp[:,0]
    A = np.stack([np.ones_like(along), along], 1)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(theta[0]), float(theta[1])

def apply_trend_along(X, vx, vy, b0, b1):
    Xp = rotate_to_flow(X, vx, vy)
    along = Xp[:,0]
    return b0 + b1 * along

def main():
    ap = argparse.ArgumentParser(description="Physics-soft GP (Malaysia barrier)")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--coast_path", type=str, default="", help="optional prebuilt coastline geojson")
    ap.add_argument("--use_trend", action="store_true", help="flow-aligned linear trend")
    args = ap.parse_args()

    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    pts_path, grid_path, meta_path = data_dir/"synth_points.csv", data_dir/"grid_coords.csv", data_dir/"flow_meta.json"
    if not (pts_path.exists() and grid_path.exists() and meta_path.exists()):
        raise SystemExit("[ERR] missing data/*.csv or flow_meta.json; run scripts/generate_synth.py first.")

    pts, grid = pd.read_csv(pts_path), pd.read_csv(grid_path)
    meta = json.loads(meta_path.read_text()); vx = float(meta.get("vx", 1.0)); vy = float(meta.get("vy", 0.3))
    X = pts[["x","y"]].to_numpy(); y = pts["z"].to_numpy()

    xs = grid.drop_duplicates("i").sort_values("i")["x"].to_numpy()
    ys = grid.drop_duplicates("j").sort_values("j")["y"].to_numpy()
    if args.coast_path:
        coast_poly = load_geojson_polygon(args.coast_path)
    else:
        coast_fp = prepare_malaysia_barrier(str(data_dir), xs, ys, simplify_tol=CFG["simplify_tol"])
        coast_poly = load_geojson_polygon(coast_fp)

    if args.use_trend:
        b0, b1 = fit_trend_along(X, y, vx, vy)
        y_res = y - apply_trend_along(X, vx, vy, b0, b1)
    else:
        b0, b1 = 0.0, 0.0
        y_res = y

    nx = int(grid["i"].max() + 1); ny = int(grid["j"].max() + 1)
    X_eval = grid[["x","y"]].to_numpy()

    best = grid_search_phys(X, y_res, X_eval, (nx,ny,xs,ys), (vx,vy), CFG["kappa"],
                            coast_poly=coast_poly, alpha_obs=CFG["alpha_obs"],
                            lam_phys=CFG["physics_lambda"], angle_deg0=CFG["angle_deg0"])
    gp = best["gp"]

    # report vs baseline
    base = fit_baseline(X, y_res, alpha=CFG["alpha_obs"])
    mu,std = gp.predict(X, return_std=True)
    mu_b,std_b = base.predict(X, return_std=True)
    metrics = {
        "rmse_phys": rmse(y_res, mu),
        "rmse_base": rmse(y_res, mu_b),
        "crps_phys": crps_gaussian(y_res, mu, std),
        "crps_base": crps_gaussian(y_res, mu_b, std_b),
        "best_cfg": {k: float(v) for k,v in best.items() if k in ("angle_rad","la_s","lc_s","la_rq","lc_rq","la_m","lc_m")}
    }
    metrics["skill_vs_base"] = 1.0 - metrics["crps_phys"]/(metrics["crps_base"] + 1e-9)

    mu_g, std_g = gp.predict(X_eval, return_std=True)
    if args.use_trend:
        mu_g = mu_g + apply_trend_along(X_eval, vx, vy, b0, b1)
    Zm = mu_g.reshape(ny, nx); Zs = std_g.reshape(ny, nx)

    out_dir.mkdir(parents=True, exist_ok=True)
    tri = mtri.Triangulation(grid["x"].to_numpy(), grid["y"].to_numpy())
    plt.figure(); plt.tricontourf(tri, mu_g, levels=20); plt.colorbar(); plt.title("Mean (physics-soft, MY barrier)")
    plt.scatter(pts["x"], pts["y"], s=8, c="k", alpha=0.6, linewidths=0.2)
    plt.tight_layout(); plt.savefig(out_dir/"mean_physics_soft.png", dpi=180); plt.close()
    plt.figure(); plt.tricontourf(tri, std_g, levels=20); plt.colorbar(); plt.title("Std (physics-soft)")
    plt.scatter(pts["x"], pts["y"], s=8, c="k", alpha=0.6, linewidths=0.2)
    plt.tight_layout(); plt.savefig(out_dir/"std_map.png", dpi=180); plt.close()

    R = pde_residual_steady(Zm, xs, ys, vx=vx, vy=vy, kappa=CFG["kappa"])
    plt.figure(); plt.hist(R.ravel(), bins=60); plt.title("PDE residual (steady)")
    plt.tight_layout(); plt.savefig(out_dir/"physics_residual.png", dpi=180); plt.close()

    h1,g1 = dir_variogram(grid["x"].to_numpy(), grid["y"].to_numpy(), mu_g, angle_rad=np.arctan2(vy,vx))
    h2,g2 = dir_variogram(grid["x"].to_numpy(), grid["y"].to_numpy(), mu_g, angle_rad=np.arctan2(vy,vx)+np.pi/2)
    plt.figure()
    if h1.size>0: plt.plot(h1,g1,label="‖ flow")
    if h2.size>0: plt.plot(h2,g2,label="⊥ flow")
    plt.legend(); plt.xlabel("lag"); plt.ylabel("γ(h)"); plt.title("Directional variogram")
    plt.tight_layout(); plt.savefig(out_dir/"variogram.png", dpi=180); plt.close()

    rng = np.random.default_rng(0)
    cal_idx = rng.choice(len(X), size=max(1,int(0.3*len(X))), replace=False)
    mu_c, std_c = gp.predict(X[cal_idx], return_std=True)
    qhat = calibrate(y_res[cal_idx], mu_c, std_c, alpha=0.1)
    cov = coverage(y_res, mu, std, qhat)
    metrics["conformal_qhat"] = float(qhat); metrics["coverage_approx"] = float(cov)

    tau = float(np.quantile(y, 0.85))
    cands = grid[["x","y"]].to_numpy()
    next_pts = pick_next_points(cands, mu_g, std_g, tau,
                                k_pick=CFG["next_k"], min_dist=CFG["min_dist"],
                                port=tuple(CFG["port_xy"]), lam_cost=CFG["lam_cost"])
    np.savetxt(out_dir/"next_points.csv", next_pts, delimiter=",", header="x,y", comments="")

    (out_dir/"metrics_physics_soft.json").write_text(json.dumps(metrics, indent=2))
    print("[OK] saved mean/std/metrics_physics_soft.json/physics_residual.png/variogram.png/next_points.csv")

if __name__ == "__main__":
    main()
