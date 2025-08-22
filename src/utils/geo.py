"""Utility functions to transform coordinates between original (x, y) space
   and a flow-aligned system (along-flow, cross-flow)."""

from __future__ import annotations
import numpy as np

def angle_from_flow(vx: float, vy: float) -> float:
    norm = float(np.hypot(vx, vy))
    return float(np.arctan2(vy, vx))

def _ensure_nx2(arr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Ensure last dim is 2; reshape back on return."""
    A = np.asarray(arr, dtype=float)
    if A.shape == (2,):
        original_shape = (2,)
        A2 = A.reshape(1, 2)
    else:
        if A.ndim < 2 or A.shape[-1] != 2:
            raise ValueError("Input must have shape (..., 2).")
        original_shape = A.shape
        A2 = A.reshape(-1, 2)
    return A2, original_shape

def rotate_to_flow(X: np.ndarray, vx: float, vy: float) -> np.ndarray:
    """
    Map original coords (x,y) -> flow-aligned (along, cross).
    """
    X2, orig_shape = _ensure_nx2(X)
    theta = angle_from_flow(vx, vy)
    c, s = np.cos(theta), np.sin(theta)
    # R maps [x, y] -> [along, cross]
    R = np.array([[ c,  s],
                  [-s,  c]], dtype=float)
    Y = X2 @ R.T
    return Y.reshape(orig_shape)

def rotate_from_flow(Xp: np.ndarray, vx: float, vy: float) -> np.ndarray:
    Xp2, orig_shape = _ensure_nx2(Xp)
    theta = angle_from_flow(vx, vy)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[ c,  s],
                  [-s,  c]], dtype=float)
    R_inv = R.T  # == [[ c, -s], [ s, c]]
    Y = Xp2 @ R_inv.T
    return Y.reshape(orig_shape)
