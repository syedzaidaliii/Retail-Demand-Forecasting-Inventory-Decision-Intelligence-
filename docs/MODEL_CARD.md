# Model card: LightGBM 28-day demand model (`lgbm_h28`)

## Intended use
Daily unit-demand forecasts for item-store series, 1 to 28 days ahead, and a simulated order recommendation.
A portfolio / learning project. **Not validated for real purchasing decisions.**

## Data
M5 (Walmart) sales, calendar and prices from the Kaggle competition *M5 Forecasting - Accuracy*.
A stratified sample of 999 item-store series (about 3.3% of the 30,490 series) from the four California stores.
Training window: the last 730 days up to 2016-04-24. Final holdout: 2016-04-25 to 2016-05-22 (never used for any decision).

## Model
LightGBM with a Tweedie objective (`tweedie_variance_power` 1.3), a direct 28-day horizon, 37 features
(lags of at least 28 days, rolling statistics, intermittency, calendar, price, series identity).
Hyper-parameters were tuned on two validation folds and confirmed on a third.

## Measured performance (final holdout, 999 series)
| | LightGBM | 28-day moving average |
|---|---|---|
| WAPE (daily) | 0.744 | 0.789 |
| WAPE (7-day sums) | 0.370 | 0.416 |
| WAPE (28-day sums) | 0.263 | 0.314 |

The daily improvement is 5.7% (95% bootstrap CI 2.9% to 9.1%). Total demand was under-forecast by 6.9%.
The model helps on regular, high-volume series (smooth/erratic classes, +10% to +14%) and is about equal to the
baseline on sparse series (intermittent class +1%, low-volume tier -0.4%).

## Alternatives that were evaluated
Chronos-2 (zero-shot) had a better daily WAPE on the folds (0.712) but under-forecast total demand by about 31% and was worse on
7-day and 28-day sums; TFT (one configuration, one seed) performed poorly. LightGBM was selected using rules fixed before the comparison.

## Limitations
* One state and about 3% of the series; no test on other states.
* Items not seen in training (new items) are not supported reliably (`item_id` is a feature).
* Point forecasts only; no prediction intervals are served.
* Prices and calendar are assumed to be known in advance.
* Systematic under-forecast (about 7% in total on the holdout), larger on Mondays and on low-volume series.
* Inventory recommendations rest on assumptions chosen by the user (lead time, service level); in the simulated backtest the better
  forecast changed total cost by only about 1.7% (and by about 0% at equal service level).

## Monitoring suggestions
Track WAPE on 7-day sums and the bias once actuals arrive, by volume tier and store; re-train or fall back to the moving-average baseline
if the improvement over the baseline disappears. The final holdout is used up: a new model needs a new, later evaluation period.
