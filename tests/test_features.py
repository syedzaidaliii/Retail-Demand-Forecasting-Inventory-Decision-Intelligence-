"""Feature tests: no leakage, correct values, structural zeros handled as NaN."""

import numpy as np
import pandas as pd

from retail_forecast.features import build_features

H = 28
SALES_BASED = ["lags", "rolling", "intermittency"]


def test_feature_groups_are_stable(panel):
    _, groups = build_features(panel, H)
    assert set(groups) == {"lags", "rolling", "intermittency", "calendar", "price", "series_info"}
    assert sum(len(v) for v in groups.values()) == 37
    assert groups["lags"] == ["lag_28", "lag_35", "lag_42", "lag_49"]


def test_sales_features_do_not_depend_on_the_future(panel):
    """Replace all sales after a cut-off with random numbers: features up to cut-off + h must not change."""
    cutoff = panel["date"].max() - pd.Timedelta(days=100)
    a, groups = build_features(panel, H)

    corrupted = panel.copy()
    corrupted["sales"] = corrupted["sales"].astype("float32")
    mask = corrupted["date"] > cutoff
    corrupted.loc[mask, "sales"] = np.random.default_rng(0).integers(0, 100, int(mask.sum())).astype("float32")
    b, _ = build_features(corrupted, H)

    rows = (a["date"] <= cutoff + pd.Timedelta(days=H)).to_numpy()
    for group in SALES_BASED:
        for col in groups[group]:
            assert np.allclose(a.loc[rows, col].to_numpy(float), b.loc[rows, col].to_numpy(float), equal_nan=True), col


def test_a_leaky_feature_would_be_detected(panel):
    """The leakage test must fail for an un-shifted rolling mean (so that the test above means something)."""
    s = panel[panel["id"] == panel["id"].iloc[0]].set_index("date")["sales"].astype(float)
    cutoff = s.index[300]
    corrupted = s.copy()
    corrupted[corrupted.index > cutoff] = 99.0
    rows = s.index <= cutoff + pd.Timedelta(days=H)
    leaky = s.rolling(7, min_periods=1).mean()
    leaky_bad = corrupted.rolling(7, min_periods=1).mean()
    assert not np.allclose(leaky[rows], leaky_bad[rows], equal_nan=True)


def test_lag_and_rolling_match_a_manual_calculation(panel):
    feats, _ = build_features(panel, H)
    sid = panel["id"].iloc[0]
    g = panel[panel["id"] == sid].sort_values("date").reset_index(drop=True)
    s = g["sales"].where(g["is_available"])
    i = 250
    row = feats[(feats["id"] == sid) & (feats["date"] == g.loc[i, "date"])].iloc[0]
    assert np.isclose(row["lag_28"], s.iloc[i - 28], equal_nan=True)
    assert np.isclose(row["lag_35"], s.iloc[i - 35], equal_nan=True)
    assert np.isclose(row["roll_mean_7"], s.shift(H).rolling(7, min_periods=1).mean().iloc[i], equal_nan=True)
    assert np.isclose(row["roll_mean_28"], s.shift(H).rolling(28, min_periods=1).mean().iloc[i], equal_nan=True)


def test_unavailable_days_are_nan_and_not_zero(panel):
    """Pre-launch days carry sales = 0 in the raw data, but must never enter a lag as a zero."""
    unavailable = panel.groupby("id")["is_available"].apply(lambda s: (~s).sum())
    sid = unavailable[unavailable >= 30].index[0]
    feats, _ = build_features(panel, H)
    g = feats[feats["id"] == sid].reset_index(drop=True)
    raw = panel[panel["id"] == sid].sort_values("date").reset_index(drop=True)
    launch = int((~raw["is_available"]).sum())
    j = launch + 5                                           # its lag_28 points into the pre-launch period
    assert raw.loc[j - 28, "sales"] == 0 and not raw.loc[j - 28, "is_available"]
    assert np.isnan(g.loc[j, "lag_28"])


def test_forecast_window_rows_have_features(panel):
    feats, _ = build_features(panel, H)
    origin = panel.loc[panel["sales"].notna(), "date"].max()
    future = feats[feats["date"] > origin]
    assert future["id"].nunique() == panel["id"].nunique()
    assert future["lag_28"].notna().mean() > 0.9           # lag_28 of a future day lies inside the observed history
