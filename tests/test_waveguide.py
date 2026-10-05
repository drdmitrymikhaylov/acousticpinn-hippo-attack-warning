"""Checks on the shallow-water waveguide section of the project page.

The physics is recomputed from src/waveguide_cutoff.py, compared with
results/waveguide_cutoff.json, and the rows and sentences on the page are
rebuilt from those numbers.  Run: python -m pytest tests/test_waveguide.py
(numpy and scipy only).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import waveguide_cutoff as W  # noqa: E402
from propagation import C_WATER, tl_water_db  # noqa: E402

RES = json.load(open(os.path.join(ROOT, "results", "waveguide_cutoff.json")))


def readme():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        return " ".join(f.read().split())          # line wraps must not matter


def half_up(x, nd=0):
    q = 10 ** nd
    v = int(abs(x) * q + 0.5) / q
    v = -v if x < 0 else v
    return f"{v:.{nd}f}"


def depth_label(h):
    return f"{h:g} m"


def cutoff_row(h):
    return (f"| {depth_label(h)} | {half_up(W.cutoff_rigid_hz(h))} Hz | "
            f"{half_up(W.cutoff_fluid_bottom_hz(h, 1700.0))} Hz |")


def tl_row(f, h=3.0, ranges=(3.0, 5.0, 10.0, 30.0)):
    v = [W.tl_exact_db(f, h, h / 2, h / 2, r) for r in ranges]
    return v, " | ".join(half_up(x, 1) for x in v)


def test_cutoff_table_is_c_over_4h_and_the_pekeris_factor():
    t = readme()
    for h in (1.0, 1.5, 2.0, 3.0, 5.0):
        assert abs(W.cutoff_rigid_hz(h) - C_WATER / (4 * h)) < 1e-9
        assert abs(RES["cutoff_hz"][str(h)]["rigid_floor"] - W.cutoff_rigid_hz(h)) < 1e-9
        assert abs(RES["cutoff_hz"][str(h)]["fluid_sand_bottom"]
                   - W.cutoff_fluid_bottom_hz(h, 1700.0)) < 1e-9
        assert cutoff_row(h) in t, cutoff_row(h)
    # a faster fluid bottom is closer to rigid, and none brings the cut-off below c/4H
    assert W.cutoff_fluid_bottom_hz(3.0, 1700.0) > W.cutoff_fluid_bottom_hz(3.0, 5000.0) > W.cutoff_rigid_hz(3.0)


def test_modal_sum_agrees_with_an_independent_image_series():
    for f, h, r in ((50.0, 2.0, 3.0), (50.0, 3.0, 5.0), (100.0, 3.0, 4.0), (20.0, 2.0, 2.0)):
        gm = W.green_modes(f, h, h / 2, h / 2, r)
        gi = W.green_images(f, h, h / 2, h / 2, r)
        assert abs(gm - gi) / abs(gm) < 1e-5
    assert max(c["relative_difference"] for c in RES["modes_against_image_series"]) < 1e-6
    assert "agreement to better than 1e-6" in readme()


def test_field_meets_both_boundary_conditions():
    f, h, zs, r = 50.0, 3.0, 1.5, 4.0
    mid = abs(W.green_modes(f, h, zs, 1.5, r))
    assert abs(W.green_modes(f, h, zs, 0.0, r)) < 1e-12 * mid          # pressure release
    dz = 1e-4                                                          # rigid floor: dp/dz = 0
    slope = abs(W.green_modes(f, h, zs, h, r) - W.green_modes(f, h, zs, h - dz, r)) / dz
    assert slope * h < 1e-3 * mid


def test_transmission_loss_table_and_the_spreading_law_row():
    t = readme()
    tab = RES["transmission_loss_re_1m"]["3.0"]
    assert tab["ranges_m"] == [3.0, 5.0, 10.0, 30.0]
    law = [float(tl_water_db(r, 3.0)) for r in tab["ranges_m"]]
    assert "| spreading law used for the range figure | " + " | ".join(
        half_up(x, 1) + " dB" for x in law) + " |" in t
    for f, tag in ((20, "20 Hz"), (50, "50 Hz"), (100, "100 Hz"), (205, "205 Hz (above cut-off)")):
        v, cells = tl_row(f)
        assert np.allclose(v, tab["exact_db"][str(f)], atol=1e-6)
        assert f"| exact, {tag} | {cells} |" in t, (f, cells)
    v50, _ = tl_row(50)
    assert half_up(v50[2] - law[2]) == "35" and half_up(v50[3] - law[3]) == "118"
    assert "arrives 35 dB below what the range budget assumes, and at thirty metres 118 dB below" in t
    v205, _ = tl_row(205)
    gain = [a - b for a, b in zip(law, v205)]                          # above cut-off: kinder
    assert 4.0 < min(gain) and max(gain) < 5.0
    assert "4 to 5 dB kinder than the spreading law" in t


def test_decay_rates_below_cutoff():
    t = readme()
    assert half_up(W.evanescent_db_per_m(50, 3.0), 1) == "4.2"
    assert half_up(W.evanescent_db_per_m(50, 2.0), 1) == "6.6"
    assert half_up(RES["low_frequency_limit_db_per_depth"], 1) == "13.6"
    assert abs(RES["low_frequency_limit_db_per_depth"] - 20 / np.log(10) * np.pi / 2) < 1e-12
    assert "4.2 dB/m at 50 Hz in 3 m of water, 6.6 dB/m in 2 m" in t
    assert "13.6 dB for every water depth of range" in t
    # far from the source the exact field falls at the first mode's rate
    slope = W.tl_exact_db(50, 3.0, 1.5, 1.5, 21.0) - W.tl_exact_db(50, 3.0, 1.5, 1.5, 20.0)
    assert abs(slope - W.evanescent_db_per_m(50, 3.0)) < 0.25
    assert W.evanescent_db_per_m(205, 3.0) == 0.0                      # above cut-off
    for h in ("2.0", "3.0"):
        for f in ("10", "20", "50", "100"):
            assert abs(RES["evanescent_db_per_m_rigid_floor"][h][f]
                       - W.evanescent_db_per_m(float(f), float(h))) < 1e-9


def test_ranges_at_which_the_low_band_is_10_and_20_db_under_the_law():
    t = readme()
    got = {}
    for h in (3.0, 2.0):
        tab = RES["transmission_loss_re_1m"][str(h)]
        for gap, key in ((10.0, "range_m_where_exact_is_10_db_under_the_law"),
                         (20.0, "range_m_where_exact_is_20_db_under_the_law")):
            r = tab[key]["50"]
            lo = W.tl_exact_db(50, h, h / 2, h / 2, r - 0.02) - float(tl_water_db(r - 0.02, h))
            hi = W.tl_exact_db(50, h, h / 2, h / 2, r) - float(tl_water_db(r, h))
            assert lo < gap <= hi
            got[(h, gap)] = half_up(r, 1)
    assert got == {(3.0, 10.0): "3.9", (3.0, 20.0): "6.3", (2.0, 10.0): "2.4", (2.0, 20.0): "3.9"}
    assert ("10 dB under the spreading law by 3.9 m (50 Hz, 3 m of water) and 20 dB under it "
            "by 6.3 m; in 2 m of water by 2.4 m and 3.9 m") in t
    # "about two water depths": 20 dB under the law by roughly 2H in both pools
    for h in (3.0, 2.0):
        r20 = RES["transmission_loss_re_1m"][str(h)]["range_m_where_exact_is_20_db_under_the_law"]["50"]
        assert 1.9 < r20 / h < 2.2
    assert "That is about two water depths in both cases." in t
