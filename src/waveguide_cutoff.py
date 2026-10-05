"""A shallow pool or river as a waveguide: what the water column carries.

propagation.tl_water_db spreads the hydrophone's signal spherically out to
one water depth and cylindrically beyond it, at every frequency.  Water a
few metres deep under a pressure-release surface is a waveguide, and a
waveguide has a lowest frequency it will carry.  This module puts numbers
on that with two textbook results (Jensen, Kuperman, Porter & Schmidt,
Computational Ocean Acoustics, ch. 2 and 5):

  * cut-off of the first mode, for a rigid floor and for a fluid (Pekeris)
    bottom;
  * the exact field of a point source between a pressure-release surface
    and a rigid floor, as a normal-mode sum.  Below cut-off every mode is
    evanescent and the sum is the whole field.  The sum is checked here
    against an independent image-source series.

Run from the repository root:

    python src/waveguide_cutoff.py        # writes results/waveguide_cutoff.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import hankel1, k0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from propagation import C_WATER, tl_water_db  # noqa: E402

DB_PER_NEPER = 20.0 / np.log(10.0)          # 8.686


def cutoff_rigid_hz(depth_m: float, mode: int = 1) -> float:
    """Cut-off of mode n: pressure-release surface, rigid floor."""
    return (2 * mode - 1) * C_WATER / (4.0 * depth_m)


def cutoff_fluid_bottom_hz(depth_m: float, c_bottom: float, mode: int = 1) -> float:
    """Cut-off of mode n over a fluid half-space faster than water (Pekeris)."""
    return cutoff_rigid_hz(depth_m, mode) / np.sqrt(1.0 - (C_WATER / c_bottom) ** 2)


def evanescent_db_per_m(f_hz: float, depth_m: float) -> float:
    """Decay of the slowest (first) mode below the rigid-floor cut-off."""
    kz = np.pi / (2.0 * depth_m)
    k = 2.0 * np.pi * f_hz / C_WATER
    return 0.0 if k >= kz else float(DB_PER_NEPER * np.sqrt(kz * kz - k * k))


def green_modes(f_hz, depth_m, z_src, z_rec, range_m, n_modes: int = 400) -> complex:
    """Point-source Green's function (free field = exp(ikR)/(4 pi R)) between a
    pressure-release surface at z = 0 and a rigid floor at z = depth_m."""
    k = 2.0 * np.pi * f_hz / C_WATER
    n = np.arange(1, n_modes + 1)
    kz = (n - 0.5) * np.pi / depth_m
    shape = np.sin(kz * z_src) * np.sin(kz * z_rec)
    kr2 = k * k - kz * kz
    out = 0.0 + 0.0j
    prop = kr2 > 0
    if prop.any():
        out += (1j / (2.0 * depth_m)) * np.sum(shape[prop] * hankel1(0, np.sqrt(kr2[prop]) * range_m))
    if (~prop).any():
        out += (1.0 / (np.pi * depth_m)) * np.sum(shape[~prop] * k0(np.sqrt(-kr2[~prop]) * range_m))
    return complex(out)


def green_images(f_hz, depth_m, z_src, z_rec, range_m, n_cells: int = 4000) -> complex:
    """The same field as a series of image sources (surface reflects with -1,
    rigid floor with +1), averaged over partial sums.  Independent of the
    modal sum; used only as a check."""
    k = 2.0 * np.pi * f_hz / C_WATER
    m = np.arange(-n_cells, n_cells + 1)[:, None] * 4.0 * depth_m
    z_img = m + np.array([z_src, -z_src, 2 * depth_m - z_src, 2 * depth_m + z_src])[None, :]
    sign = np.array([1.0, -1.0, 1.0, -1.0])[None, :]
    R = np.sqrt(range_m ** 2 + (z_rec - z_img) ** 2)
    cell = (sign * np.exp(1j * k * R) / (4.0 * np.pi * R)).sum(axis=1)
    order = np.argsort(np.abs(np.arange(-n_cells, n_cells + 1)), kind="stable")
    partial = np.cumsum(cell[order])
    return complex(partial[len(partial) // 2:].mean())      # Cesaro mean of the tail


def tl_exact_db(f_hz, depth_m, z_src, z_rec, range_m) -> float:
    """Transmission loss re 1 m in free field, ideal rigid-floor waveguide."""
    return float(-20.0 * np.log10(4.0 * np.pi * abs(green_modes(f_hz, depth_m, z_src, z_rec, range_m))))


def range_where_exact_is_below_law(f_hz, depth_m, z_src, z_rec, gap_db, r_max=200.0):
    """Smallest range (>= one depth) at which the exact level is gap_db under
    the spreading law; None if it never is within r_max."""
    for r in np.arange(depth_m, r_max, 0.01):
        if tl_exact_db(f_hz, depth_m, z_src, z_rec, r) - float(tl_water_db(r, depth_m)) >= gap_db:
            return round(float(r), 2)
    return None


def main():
    depths = [1.0, 1.5, 2.0, 3.0, 5.0]
    c_sand = 1700.0
    out = {
        "c_water_m_s": C_WATER,
        "c_bottom_saturated_sand_m_s": c_sand,
        "cutoff_hz": {str(h): {"rigid_floor": cutoff_rigid_hz(h),
                               "fluid_sand_bottom": cutoff_fluid_bottom_hz(h, c_sand)} for h in depths},
        "evanescent_db_per_m_rigid_floor": {
            str(h): {str(f): evanescent_db_per_m(f, h) for f in (10, 20, 50, 100)} for h in (2.0, 3.0)},
        "low_frequency_limit_db_per_depth": float(DB_PER_NEPER * np.pi / 2.0),
    }
    tables = {}
    for h in (3.0, 2.0):                               # 3 m is the depth the range figure assumes
        zs = zr = h / 2.0
        ranges = [3.0, 5.0, 10.0, 30.0] if h == 3.0 else [2.0, 5.0, 10.0, 20.0]
        t = {"source_depth_m": zs, "hydrophone_depth_m": zr, "ranges_m": ranges,
             "spreading_law_db": [float(tl_water_db(r, h)) for r in ranges], "exact_db": {}}
        for f in (20, 50, 100, 205, 400):
            t["exact_db"][str(f)] = [tl_exact_db(f, h, zs, zr, r) for r in ranges]
        t["range_m_where_exact_is_10_db_under_the_law"] = {
            str(f): range_where_exact_is_below_law(f, h, zs, zr, 10.0) for f in (20, 50, 100)}
        t["range_m_where_exact_is_20_db_under_the_law"] = {
            str(f): range_where_exact_is_below_law(f, h, zs, zr, 20.0) for f in (20, 50, 100)}
        tables[str(h)] = t
    out["transmission_loss_re_1m"] = tables
    chk = []
    for f, h, r in ((50.0, 2.0, 3.0), (50.0, 3.0, 5.0), (100.0, 3.0, 4.0), (20.0, 2.0, 2.0)):
        gm, gi = green_modes(f, h, h / 2, h / 2, r), green_images(f, h, h / 2, h / 2, r)
        chk.append({"f_hz": f, "depth_m": h, "range_m": r, "modes": abs(gm), "images": abs(gi),
                    "relative_difference": abs(gm - gi) / abs(gm)})
    out["modes_against_image_series"] = chk
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "waveguide_cutoff.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
