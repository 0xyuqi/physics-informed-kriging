"""
Coordinate transforms (flow-aligned) + Malaysia coastline utilities.
Style: short docstrings + clear inline comments.
"""
from __future__ import annotations
import os, io, json, urllib.request
from typing import Tuple
import numpy as np

try:
    from shapely.geometry import shape, mapping, LineString, Polygon, Point
    from shapely.ops import unary_union
    from shapely.affinity import scale, translate
except Exception:
    shape = mapping = LineString = Polygon = Point = unary_union = scale = translate = None

def angle_from_flow(vx: float, vy: float) -> float:
    return float(np.arctan2(vy, vx))

def _ensure_nx2(arr: np.ndarray):
    A = np.asarray(arr, dtype=float)
    if A.shape == (2,):
        original_shape = (2,); A2 = A.reshape(1, 2)
    else:
        if A.ndim < 2 or A.shape[-1] != 2:
            raise ValueError("Input must have shape (..., 2).")
        original_shape = A.shape; A2 = A.reshape(-1, 2)
    return A2, original_shape

def rotate_to_flow(X: np.ndarray, vx: float, vy: float) -> np.ndarray:
    """
    Map (x,y) -> (along, cross) by rotating with the flow angle.
    along =  cosθ x + sinθ y
    cross = -sinθ x + cosθ y
    """
    X2, orig_shape = _ensure_nx2(X)
    th = angle_from_flow(vx, vy)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[ c,  s], [-s,  c]], dtype=float)
    Y = X2 @ R.T
    return Y.reshape(orig_shape)

def rotate_from_flow(Xp: np.ndarray, vx: float, vy: float) -> np.ndarray:
    Xp2, orig_shape = _ensure_nx2(Xp)
    th = angle_from_flow(vx, vy)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[ c,  s], [-s,  c]], dtype=float)
    R_inv = R.T
    Y = Xp2 @ R_inv.T
    return Y.reshape(orig_shape)

_COUNTRIES_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

def _require_shapely():
    if shape is None or unary_union is None or scale is None:
        raise RuntimeError("shapely is required for coastline functions.")

def _download_countries() -> dict:
    with urllib.request.urlopen(_COUNTRIES_URL, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _build_malaysia_union() -> 'Polygon':
    _require_shapely()
    gj = _download_countries()
    geoms = []
    for f in gj["features"]:
        iso = (f["properties"].get("ISO_A3") or f["properties"].get("iso_a3"))
        if str(iso).upper() == "MYS":
            geoms.append(shape(f["geometry"]).buffer(0))
    if not geoms:
        raise RuntimeError("Malaysia geometry not found.")
    return unary_union(geoms).buffer(0)

def _fit_polygon_to_bbox(poly: 'Polygon',
                         source_bounds: Tuple[float,float,float,float],
                         target_bounds: Tuple[float,float,float,float]) -> 'Polygon':
    _require_shapely()
    (sx0, sy0, sx1, sy1) = source_bounds
    (tx0, ty0, tx1, ty1) = target_bounds
    sw = max(sx1 - sx0, 1e-9); sh = max(sy1 - sy0, 1e-9)
    poly0 = translate(poly, xoff=-sx0, yoff=-sy0)
    poly1 = scale(poly0, xfact=1.0/sw, yfact=1.0/sh, origin=(0,0))
    tw = max(tx1 - tx0, 1e-9); th = max(ty1 - ty0, 1e-9)
    poly2 = scale(poly1, xfact=tw, yfact=th, origin=(0,0))
    poly3 = translate(poly2, xoff=tx0, yoff=ty0)
    return poly3

def prepare_malaysia_barrier(data_dir: str, xs: np.ndarray, ys: np.ndarray, simplify_tol: float = 0.02) -> str:
    """
    Build Malaysia coastline and fit to data grid bbox; save to geojson.
    """
    _require_shapely()
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "malaysia_barrier.geojson")
    poly_ll = _build_malaysia_union()
    poly_ll = poly_ll.simplify(simplify_tol, preserve_topology=True)
    sx0, sy0, sx1, sy1 = poly_ll.bounds
    tx0, tx1 = float(np.min(xs)), float(np.max(xs))
    ty0, ty1 = float(np.min(ys)), float(np.max(ys))
    poly_fit = _fit_polygon_to_bbox(poly_ll, (sx0,sy0,sx1,sy1), (tx0,ty0,tx1,ty1))
    gj_out = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{"name":"Malaysia (fitted)"},"geometry": mapping(poly_fit)}]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj_out, f)
    return out_path

def load_geojson_polygon(path: str) -> 'Polygon':
    _require_shapely()
    with open(path,"r",encoding="utf-8") as f: gj=json.load(f)
    return shape(gj["features"][0]["geometry"]).buffer(0)

def segment_crosses_land(p: np.ndarray, q: np.ndarray, coast_poly: 'Polygon') -> bool:
    _require_shapely()
    ls = LineString([tuple(np.asarray(p,float)), tuple(np.asarray(q,float))])
    return ls.crosses(coast_poly) or ls.within(coast_poly)

def dist_to_coast(points: np.ndarray, coast_poly: 'Polygon') -> np.ndarray:
    """
    Distance from points to coastline boundary; used to build heteroscedastic alpha.
    """
    _require_shapely()
    P = np.asarray(points, float)
    return np.array([Point(float(px), float(py)).distance(coast_poly.boundary) for px,py in P], dtype=float)
