# Changelog

## 2026-10-19

- LICENSE now holds the standard MIT text and LICENSE-CONTENT the standard CC BY 4.0 text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-10-05

- **The water column is a waveguide** (new README section "A shallow pool
  does not carry the low band", `src/waveguide_cutoff.py`,
  `results/waveguide_cutoff.json`, `tests/test_waveguide.py`, 6 tests). The
  range budget spread every frequency alike. With a rigid floor the first
  mode cuts on at c/4H: 185 Hz in 2 m of water, 123 Hz in 3 m; over
  saturated sand 376 and 251 Hz. Below cut-off the exact field (normal-mode
  sum, checked against an image-source series to better than 1e-6) falls
  4.2 dB/m at 50 Hz in 3 m of water: 35 dB under the budget at 10 m, 118 dB
  at 30 m.
- **Qualified:** the low band (the 5 Hz front end, the hydrophone flat below
  10 Hz) is a short-range signal, about two water depths; the range curve
  holds for the part of the call above cut-off, where the ideal guide is 4
  to 5 dB kinder than the spreading law at 205 Hz.
- Not shown: an elastic floor can carry what the water cannot; only the
  cut-off is given for the sand bottom.
- Housekeeping in the same commit: eight macOS `._*` files removed and
  ignored.

## 2026-09-17

- Low-band ablation read as 15 paired seed×fold differences: the 5 Hz
  front end wins 15/15 pairs in AUC (+0.0063 ± 0.0025, worst +0.0018) and
  14/15 in balanced accuracy; ≈13 vs 18 errors per 320-clip fold.
- Operating curve extended to the rates a keeper would tolerate: one false
  alarm a night catches 8 of 111 bouts (7 %); 90 % of bouts needs 3.3 false
  alarms/h. Stated the 1/12-per-hour resolution of the 12 h stream.
- Correction: the ± on balanced accuracy was the population sd; now the
  sample sd over 15 runs — 0.960 ± 0.013 → ± 0.014, 0.944 ± 0.011 → ± 0.012.
  AUC rows unchanged. Stated what ± and n mean under the table.
- Per-run results (`results/cv_fmin5/`, `results/cv_fmin200/`,
  `results/stream_eval.json`, `results/paired_checks.json`) published;
  `tests/test_results.py` pins every simulator number on the page to them.
