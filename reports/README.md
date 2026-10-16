# reports/

Small result files produced by the notebooks. They are committed because they document the measured results
(they contain aggregate numbers only, no M5 rows) and are shown in the dashboard.

| File | Produced by | Content |
|---|---|---|
| `holdout_runs.jsonl` | Notebook 5 | append-only log of the final-holdout runs; **line 1 is the official result** |
| `inventory_results.json` | Notebook 6 | simulated inventory backtest (assumption-based) |
| `comparison_results_h28.json` | Notebook 4 | LightGBM vs Chronos-2 vs TFT |
