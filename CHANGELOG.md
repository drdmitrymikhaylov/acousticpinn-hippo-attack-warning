# Changelog

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
