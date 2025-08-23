# Physics-Informed Kriging (Coastal, Malaysia shoreline)

This repo provides a runnable baseline for **anisotropic Gaussian Process (GP)** with a flow-aligned prior (rotate into along- and cross-flow coordinates), plus a **physics-soft** variant that respects **real Malaysia shoreline**.

It includes:
- Synthetic data generator (advection-shaped field) — `scripts/generate_synth.py`
- Baseline anisotropic GP (ARD RBF) in flow-aligned coords, optional along-flow trend, fast LOO — `scripts/run_baseline.py`
- Physics-soft GP with **Malaysia shoreline barrier** (cross-land + near-shore penalties), **multi-scale kernel mix (RBF + RQ + Matérn-3/2)**, heteroscedastic noise, conformal intervals, and active sampling — `scripts/run_physics_soft.py`
- Metrics: MAE / RMSE / CRPS, conformal coverage, directional variogram
- Artifacts: heatmaps, PDE-residual histogram, next-round sampling CSV

## Quickstart
```bash
# 0) Install deps
pip install -r requirements.txt

# 1) Generate synthetic data (flow vector, grid, noise are configurable)
python scripts/generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 7

# 2) Run baseline (anisotropic GP in flow-aligned coords, with fast LOO + along-flow trend)
python scripts/run_baseline.py --fast_loo --use_trend
# optional: tune kernel scales
# python scripts/run_baseline.py --fast_loo --use_trend --length_parallel 30 --length_cross 8 --alpha 1e-6

# 3) Run physics-soft (Malaysia shoreline automatically fit to grid; or pass your own)
python scripts/run_physics_soft.py --use_trend
# optional: reuse a local shoreline file to avoid network
# python scripts/run_physics_soft.py --use_trend --coast_path data/malaysia_barrier.geojson
````

## Artifacts

* `figures/mean_physics_soft.png` — posterior mean (physics-soft, Malaysia barrier)
* `figures/std_map.png` — posterior std (uncertainty)
* `figures/physics_residual.png` — PDE residual histogram
* `figures/variogram.png` — directional variogram (‖ / ⟂ to flow)
* `figures/metrics_physics_soft.json` — key metrics + best hyperparameters
* `figures/next_points.csv` — next-round sampling points (x, y)
* `figures/mean_map.png`, `figures/std_map.png` — from the baseline
* `data/grid_pred.csv` — dense grid predictions (baseline)

## Notes

* **Flow-aligned frame**: rotate (x, y) → (along, cross) using `(vx, vy)`.
* **Kernel mix (flow frame)**: `RBF` for hotspots + `RationalQuadratic` for multi-scale background + `Matérn-3/2` for roughness robustness.
* **Shoreline barrier**: multiply base kernel by

  * `exp(-λ_cross)` if the segment crosses land;
  * `exp(-η / (d + ε))` as near-shore decay (d = distance to coastline).
* **Physics-soft objective**: minimize `NLL + λ_phys · RMSE(v·∇Z − κΔZ)` on the grid.
* **Heteroscedastic noise**: `alpha_i = base + c / (dist_to_coast/scale + 1)` (near-shore noisier).
* **Conformal intervals**: calibrate a z-quantile on a holdout slice for reliable coverage.
* **Active sampling**: straddle score + approximate info gain − travel-cost penalty.

## How to reproduce

```bash
# Generate data
python scripts/generate_synth.py

# Baseline (report MAE/RMSE/CRPS via fast LOO; writes mean/std maps)
python scripts/run_baseline.py --fast_loo --use_trend

# Physics-soft full run (kernel mix + Malaysia shoreline + PDE penalty + conformal + active sampling)
python scripts/run_physics_soft.py --use_trend

# Physics-soft with custom shoreline (no download)
python scripts/run_physics_soft.py --use_trend --coast_path data/malaysia_barrier.geojson
```

## File map

```
src/
  utils/{geo.py, physics.py, metrics.py}
  models/{kernels.py, phys_soft.py, conformal.py, active_sampling.py}
scripts/
  {generate_synth.py, run_baseline.py, run_physics_soft.py}
data/      # generated: synth_points.csv, grid_coords.csv, flow_meta.json, malaysia_barrier.geojson
figures/   # outputs: *.png, metrics_physics_soft.json, next_points.csv
requirements.txt
README.md
```

Tips

* On Windows PowerShell, run commands one-by-one (no `&&`).
* If `shapely` install fails, upgrade build tools first: `pip install --upgrade pip setuptools wheel`.

```
