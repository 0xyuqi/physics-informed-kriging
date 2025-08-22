"""Utility functions to transform coordinates between original (x, y) space
   and a flow-aligned system (along-flow, cross-flow); plus coastal geometry helpers.
"""

from __future__ import annotations
import os, io, json, zipfile, urllib.request
import numpy as np

# shapely only used for coastline / barrier helpers
try:
    from shapely.geometry import shape, mapping, LineString
    from shapely.ops import unary_union
except Exception as _e:
    shape = mapping = LineString = unary_union = None  # lazy check in functions


# 1 . Flow-aligned transforms (keep original API & behavior)
def angle_from_flow(vx: float, vy: float) -> float:
    norm = float(np.hypot(vx, vy))  # keep for consistency; not used directly
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


# 2 . Coastal geometry helpers (real coastline + barrier checks)
#    Saved to data/coastline_malaysia.geojson on first use.
_NE_LAND_ZIP = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_land.zip"
_COUNTRIES_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
_TARGET_ISOS = {"MYS", "SGP", "IDN", "THA", "BRN"}  # Malaysia + neighbors (barrier continuity)

def _require_shapely():
    if shape is None or unary_union is None:
        raise RuntimeError("shapely is required for coastline functions. Please `pip install shapely`.")

def ensure_coastline(data_dir: str) -> str:
    """
    Ensure a coastline GeoJSON exists at <data_dir>/coastline_malaysia.geojson.
    Prefer Natural Earth 10m Land; fallback to 'geo-countries' union (MYS/SGP/IDN/THA/BRN).
    If offline and no cache, try <data_dir>/coastline_placeholder.geojson.
    Returns the path to the GeoJSON file.
    """
    _require_shapely()
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "coastline_malaysia.geojson")
    if os.path.exists(out_path):
        return out_path

    try:
        # Try Natural Earth Land (10m). Some packages ship shapefiles; many mirrors provide GeoJSON in the zip.
        gj = None
        with urllib.request.urlopen(_NE_LAND_ZIP, timeout=30) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
            for name in zf.namelist():
                if name.endswith(".geojson") or name.endswith(".json"):
                    gj = json.loads(zf.read(name).decode("utf-8"))
                    break
        geom = None
        if gj is not None:
            polys = [shape(f["geometry"]).buffer(0) for f in gj["features"]]
            geom = unary_union(polys).buffer(0)

        if geom is None:
            # Fallback: union of neighboring countries (keeps barrier realistic around Malaysia)
            with urllib.request.urlopen(_COUNTRIES_URL, timeout=30) as r:
                gj = json.loads(r.read().decode("utf-8"))
            geoms = []
            for f in gj["features"]:
                iso = (f["properties"].get("ISO_A3") or f["properties"].get("iso_a3"))
                if iso in _TARGET_ISOS:
                    geoms.append(shape(f["geometry"]).buffer(0))
            if not geoms:
                raise RuntimeError("No country geometries found in fallback dataset.")
            geom = unary_union(geoms).buffer(0)

        gj_out = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature",
                          "properties": {"source": "NE10m/countries-fallback"},
                          "geometry": mapping(geom)}]
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gj_out, f)
        return out_path

    except Exception:
        # Offline fallback
        alt = os.path.join(data_dir, "coastline_placeholder.geojson")
        if os.path.exists(alt):
            return alt
        # If even placeholder is missing, bubble up a clear error
        raise RuntimeError("No coastline available (network error and no placeholder at data/coastline_placeholder.geojson).")


def load_geojson_polygon(path: str):
    """
    Load a (multi)polygon from GeoJSON (first feature), return a shapely geometry (cleaned by buffer(0)).
    """
    _require_shapely()
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    return shape(gj["features"][0]["geometry"]).buffer(0)


def segment_crosses_land(p: np.ndarray, q: np.ndarray, coast_poly) -> bool:
    """
    Return True iff the straight segment p->q crosses land polygon (barrier event).
    """
    _require_shapely()
    ls = LineString([tuple(np.asarray(p, float)), tuple(np.asarray(q, float))])
    return ls.crosses(coast_poly) or ls.within(coast_poly)


def ocean_mask(points_xy: np.ndarray, coast_poly) -> np.ndarray:
    """
    Boolean mask for points that are in the ocean (i.e., not inside land polygon).
    points_xy: array of shape (N,2)
    """
    _require_shapely()
    pts = np.asarray(points_xy, float)
    # simple vectorized check by sampling; for big N consider STRtree
    out = np.ones(len(pts), dtype=bool)
    for i, (x, y) in enumerate(pts):
        out[i] = not coast_poly.contains(shape({"type": "Point", "coordinates": (float(x), float(y))}))
    return out
