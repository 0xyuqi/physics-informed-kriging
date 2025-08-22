
#simple two-sided conformal calibration.
from __future__ import annotations
import numpy as np

def calibrate(y_cal, mu_cal, std_cal, alpha=0.1) -> float:
    z = np.abs((np.asarray(y_cal)-np.asarray(mu_cal)) / np.clip(np.asarray(std_cal), 1e-9, None))
    return float(np.quantile(z, 1.0 - alpha))

def coverage(y, mu, std, qhat) -> float:
    lower = mu - qhat*std; upper = mu + qhat*std
    return float(np.mean((y >= lower) & (y <= upper)))
