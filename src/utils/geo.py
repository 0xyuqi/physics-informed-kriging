import numpy as np

def angle_from_flow(vx: float, vy: float) -> float:
    """Return rotation angle (radians) to align x'-axis with flow vector (vx, vy)."""
    return np.arctan2(vy, vx + 1e-12)

def rotate_to_flow(X: np.ndarray, vx: float, vy: float) -> np.ndarray:
    """Rotate 2D coords (N,2) so that x' is along flow, y' is cross-flow."""
    theta = angle_from_flow(vx, vy)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])  # maps [x,y] -> [x', y']
    return X @ R.T
