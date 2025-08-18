
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def main():
    root = Path('.')
    grid = pd.read_csv(root/'data'/'grid_coords.csv')
    xs = np.sort(grid['x'].unique()); ys = np.sort(grid['y'].unique())
    nx, ny = len(xs), len(ys)

    # Load anisotropic mean/std
    aniso_mean = plt.imread(root/'figures'/'mean_map.png')  # fallback if raw arrays absent
    # Better: recompute from grid_pred.csv
    gp_grid = pd.read_csv(root/'data'/'grid_pred.csv')
    Zm = gp_grid['mean'].to_numpy().reshape(ny, nx)
    Zs = gp_grid['std'].to_numpy().reshape(ny, nx)

    # Load isotropic maps if exist
    mean_iso = plt.imread(root/'figures'/'mean_iso.png') if (root/'figures'/'mean_iso.png').exists() else None
    std_iso  = plt.imread(root/'figures'/'std_iso.png') if (root/'figures'/'std_iso.png').exists() else None

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    im0 = axs[0,0].imshow(Zm, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    axs[0,0].set_title('Anisotropic Mean'); fig.colorbar(im0, ax=axs[0,0])
    im1 = axs[1,0].imshow(Zs, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    axs[1,0].set_title('Anisotropic Std'); fig.colorbar(im1, ax=axs[1,0])

    if mean_iso is None:
        # placeholder using aniso maps if iso not available
        im2 = axs[0,1].imshow(Zm, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
        axs[0,1].set_title('Isotropic Mean (placeholder)'); fig.colorbar(im2, ax=axs[0,1])
        im3 = axs[1,1].imshow(Zs, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
        axs[1,1].set_title('Isotropic Std (placeholder)'); fig.colorbar(im3, ax=axs[1,1])
    else:
        axs[0,1].imshow(mean_iso); axs[0,1].set_title('Isotropic Mean (img)')
        axs[1,1].imshow(std_iso);  axs[1,1].set_title('Isotropic Std (img)')

    for ax in axs.ravel():
        ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig(root/'figures'/'four_panel_compare.png', dpi=150)

if __name__ == '__main__':
    main()
