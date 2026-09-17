"""Pin the simulator numbers quoted on the project page to results/.

results/cv_fmin5, results/cv_fmin200: one JSON per seed x fold (15 each).
results/stream_eval.json: the 12 h stream operating curve.
results/paired_checks.json: the paired ablation and the curve ends.
Run: python -m pytest tests/test_results.py   (numpy only)
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def folds(tag):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, f"cv_{tag}", "*.json")))]
    assert len(rows) == 15
    assert sorted((r["seed"], r["fold"]) for r in rows) == [(s, f) for s in range(3) for f in range(5)]
    assert all(r["n_test"] == 320 for r in rows)
    return {(r["seed"], r["fold"]): r for r in rows}


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), v.std(ddof=1)


def test_front_end_from_5hz_0995_pm_0002_and_0960_pm_0014():
    r = folds("fmin5")
    m, s = mean_sd([x["auc"] for x in r.values()])
    assert abs(m - 0.995) < 0.0005 and abs(s - 0.002) < 0.0005
    m, s = mean_sd([x["bal_acc"] for x in r.values()])
    assert abs(m - 0.960) < 0.0005 and abs(s - 0.014) < 0.0005


def test_front_end_from_200hz_0989_pm_0004_and_0944_pm_0012():
    r = folds("fmin200")
    m, s = mean_sd([x["auc"] for x in r.values()])
    assert abs(m - 0.989) < 0.0005 and abs(s - 0.004) < 0.0005
    m, s = mean_sd([x["bal_acc"] for x in r.values()])
    assert abs(m - 0.944) < 0.0005 and abs(s - 0.012) < 0.0005


def test_low_band_paired_gain_is_15_of_15_folds():
    a, b = folds("fmin5"), folds("fmin200")
    d = np.array([a[k]["auc"] - b[k]["auc"] for k in sorted(a)])
    assert (d > 0).all()
    assert abs(d.mean() - 0.0063) < 0.0001
    assert abs(1.96 * d.std(ddof=1) / np.sqrt(15) - 0.0013) < 0.0001
    assert abs(d.min() - 0.0018) < 0.0001
    db = np.array([a[k]["bal_acc"] - b[k]["bal_acc"] for k in sorted(a)])
    assert (db > 0).sum() == 14 and abs(db.min() - (-0.0062)) < 0.0001
    p = json.load(open(os.path.join(RES, "paired_checks.json")))["paired_ablation"]
    assert p["auc"]["fmin5_wins"] == 15 and p["bal_acc"]["fmin5_wins"] == 14
    assert abs(p["auc"]["diff_mean"] - d.mean()) < 1e-9
    assert abs(p["errors_per_fold_from_bal_acc"]["fmin5"] - 12.9) < 0.05
    assert abs(p["errors_per_fold_from_bal_acc"]["fmin200"] - 17.9) < 0.05


def test_stream_trigger_and_operating_table():
    se = json.load(open(os.path.join(RES, "stream_eval.json")))
    assert se["hours"] == 12.0 and se["bouts"] == 111
    assert abs(se["trigger_recall_of_bouts"] - 0.97) < 0.005
    assert se["trigger_false_alarms_per_hour"] == 57.0
    fa = np.array([c["false_alarms_per_hour"] for c in se["curve"]])
    rc = np.array([c["recall_of_bouts"] for c in se["curve"]])

    def recall_at(x):
        return rc[fa <= x + 1e-12].max()

    for x, want in ((5, 0.93), (2, 0.75), (1, 0.47), (0.5, 0.28)):
        assert abs(recall_at(x) - want) < 0.005, (x, recall_at(x))
        assert abs(se[f"recall_at_{x}_fa_per_h"] - recall_at(x)) < 1e-9


def test_operating_curve_ends_one_false_alarm_a_night():
    e = json.load(open(os.path.join(RES, "paired_checks.json")))["operating_curve_ends"]
    assert e["fa_per_h_resolution"] == 1 / 12
    assert e["bouts_caught_at_fa_per_h"]["1_per_night_8h_0.125"] == 8
    assert e["bouts_caught_at_fa_per_h"]["0"] == 5
    assert e["bouts_caught_at_fa_per_h"]["0.5"] == 31
    assert e["false_alarms_in_stream_at_fa_per_h"]["0.5"] == 6
    assert abs(e["min_fa_per_h_for_recall"]["0.9"] - 3.33) < 0.005
    assert abs(e["min_fa_per_h_for_recall"]["0.95"] - 6.5) < 0.005
    se = json.load(open(os.path.join(RES, "stream_eval.json")))
    fa = np.array([c["false_alarms_per_hour"] for c in se["curve"]])
    rc = np.array([c["recall_of_bouts"] for c in se["curve"]])
    assert abs(rc[fa <= 0.125].max() - 8 / 111) < 1e-9
