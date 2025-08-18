# Physics-Informed Kriging 

This repo gives you a  runnable baseline for **anisotropic Gaussian Process (GP)**
with a flow-aligned prior (rotate into along- and cross-flow coordinates). It includes:

- Synthetic data generator (advection-shaped field) — `scripts/generate_synth.py`
- Baseline anisotropic GP (ARD RBF) in flow-aligned coords — `scripts/run_baseline.py`
- Metrics: RMSE / MAE / CRPS, LOOCV, grid prediction to heatmaps
- Stubs for PDE soft-constraint (to extend on Day 2-3) — `src/models/physics_stub.py`

## Quickstart

```bash
# 1) (Optional) create venv and install deps
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

# 2) Generate synthetic data
python scripts/generate_synth.py --n_obs 40 --grid 100 --noise 0.1 --seed 42

# 3) Run baseline anisotropic GP (flow-aligned)
python scripts/run_baseline.py --length_parallel 30.0 --length_cross 8.0 --alpha 1e-6 --seed 42
```

Artifacts are saved into `figures/` and `data/`:
- `figures/mean_map.png` — posterior mean
- `figures/std_map.png` — posterior std (uncertainty)
- `data/synth_points.csv` — training points (x, y, z)
- `data/grid_pred.csv` — dense grid predictions and uncertainty

## Roadmap (for A)

**Today (done here):** baseline anisotropic GP + metrics + visualizations.  
**Next:** implement *PDE soft-constraints* by adding virtual residual observations for a steady
advection-diffusion operator at collocation points (see `src/models/physics_stub.py`).

## Notes

- We align coordinates to the **flow vector** (vx, vy). RBF kernel with ARD length-scales then
  expresses different correlation ranges **along** vs **across** the flow.
- You can let the GP optimizer tune length-scales or fix/scan values via CLI.


## Role A — Final Deliverable (How to reproduce)

### Isotropic baseline
```bash
python scripts/run_isotropic.py --length 12 --alpha 1e-6 --no_opt
```

### Anisotropic (flow-aligned) baseline
```bash
python scripts/run_baseline.py --length_parallel 30 --length_cross 8 --alpha 1e-6 --no_opt
```

### Physics soft-constraint (residual-penalty search)
```bash
python scripts/run_physics_soft.py --length_parallel_list 20 30 40 --length_cross_list 6 8 12 --lambda_phys 1.0 --kappa 1.0
```

### Parameter sweep (RMSE heatmap data)
```bash
python scripts/sweep_lengths.py --lp_list 20 30 40 --lc_list 6 8 12
```

### Four-panel comparison
```bash
python scripts/make_fourpanel.py
```
Artifacts are saved in `figures/`.
