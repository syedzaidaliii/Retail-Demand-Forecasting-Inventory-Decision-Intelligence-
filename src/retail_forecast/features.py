"""Feature engineering for the 28-day (direct) demand model.

This module is the single source of truth for the features. It is used for training (notebooks),
for the holdout evaluation and for online inference, so that training and serving cannot drift apart.

Key rules (see Notebook 2):
* every sales-based feature is shifted by at least ``h`` days (no leakage from the forecast window),
* unavailable (pre-launch) days are ``NaN``, never zero,
* calendar and price features are assumed to be known in advance.
"""

import warnings

import numpy as np
import pandas as pd

__all__ = ["build_features", "model_rows", "FEATURE_GROUPS_DOC"]

FEATURE_GROUPS_DOC = {
    "lags": "sales of the same weekday h, h+7, h+14, h+21 days earlier",
    "rolling": "rolling mean/std over 7/14/28/56 days (after shifting by h) and derived ratios",
    "intermittency": "zero-share, days since last sale, size of the average sale",
    "calendar": "weekday, month, day of month, week of year, SNAP, events, distance to events",
    "price": "price, weekly price change, price relative to its history",
    "series_info": "item, department, category, store, age of the series",
}


CAL_CAP = 14   # event distances are capped at 14 days

def _to_wide(values, ids, dates):
    """Long (id-major) 1D values -> matrix (dates x ids)."""
    arr = np.asarray(values, dtype="float32").reshape(len(ids), len(dates)).T
    return pd.DataFrame(arr, index=dates, columns=ids)

def _to_flat(W):
    """Matrix (dates x ids) -> long (id-major) 1D array."""
    return W.to_numpy(dtype="float32").T.ravel()


def build_features(long, h, lag_weeks=4, windows=(7, 14, 28, 56)):
    """
    long : long table from the EDA notebook (id, date, sales, is_available, sell_price, snap, events ...)
    h    : forecast horizon in days. All sales-based features are at least h days old.
    return: (features_df, groups_dict)
    """
    # ---------- 0) Sort the panel and verify alignment ----------
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    ids   = long["id"].drop_duplicates().tolist()
    dates = pd.DatetimeIndex(np.sort(long["date"].unique()))
    n_ids, n_dates = len(ids), len(dates)
    assert len(long) == n_ids * n_dates, "Incomplete panel: some series are missing days"
    assert (dates[-1] - dates[0]).days == n_dates - 1, "Dates are not daily-continuous"
    assert (long["date"].to_numpy() == np.tile(dates.to_numpy(), n_ids)).all(), "Date alignment is wrong"
    assert (long["id"].to_numpy() == np.repeat(np.array(ids, dtype=object), n_dates)).all(), "ID alignment is wrong"

    F, G = {}, {}

    # ---------- 1) Sales matrix: unavailable = NaN (not zero!) ----------
    S  = _to_wide(long["sales"].where(long["is_available"]), ids, dates)
    Sh = S.shift(h)                       # <-- the key line: data that is at least h days old

    # ---------- 2) Lags (multiples of 7 => same weekday) ----------
    G["lags"] = []
    for k in range(lag_weeks):
        L = h + 7 * k
        F[f"lag_{L}"] = _to_flat(S.shift(L))
        G["lags"].append(f"lag_{L}")

    # ---------- 3) Rolling mean/std (shift FIRST, then roll) ----------
    G["rolling"] = []
    for w in windows:
        F[f"roll_mean_{w}"] = _to_flat(Sh.rolling(w, min_periods=1).mean())
        F[f"roll_std_{w}"]  = _to_flat(Sh.rolling(w, min_periods=2).std())
        G["rolling"] += [f"roll_mean_{w}", f"roll_std_{w}"]

    # ---------- 4) Derived ----------
    lag_cols = G["lags"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        F["same_weekday_mean"] = np.nanmean(np.vstack([F[c] for c in lag_cols]), axis=0)
    F["trend_ratio"] = F["roll_mean_14"] / (F["roll_mean_56"] + 0.01)
    G["rolling"] += ["same_weekday_mean", "trend_ratio"]

    # ---------- 5) Intermittency ----------
    Z  = (S == 0).astype("float32").where(S.notna())     # 1 = zero-sales day, 0 = sale occurred, NaN = unavailable
    Zh = Z.shift(h)
    F["zero_share_28"]    = _to_flat(Zh.rolling(28, min_periods=1).mean())
    F["zero_share_56"]    = _to_flat(Zh.rolling(56, min_periods=1).mean())
    F["nonzero_mean_56"]  = _to_flat(Sh.where(Sh > 0).rolling(56, min_periods=1).mean())   # average size of a sale when one occurs
    arr  = Sh.to_numpy()
    pos  = np.arange(n_dates)[:, None]
    last = pd.DataFrame(np.where(arr > 0, pos, np.nan)).ffill().to_numpy()   # position of the last sale
    F["days_since_last_sale"] = (pos - last).T.ravel().astype("float32")
    G["intermittency"] = ["zero_share_28", "zero_share_56", "nonzero_mean_56", "days_since_last_sale"]

    # ---------- 6) Calendar (known in the future => no shift needed) ----------
    d = long["date"]
    F["dow"]         = d.dt.dayofweek.to_numpy().astype("int8")
    F["month"]       = d.dt.month.to_numpy().astype("int8")
    F["dom"]         = d.dt.day.to_numpy().astype("int8")
    F["weekofyear"]  = d.dt.isocalendar().week.to_numpy().astype("int8")
    F["is_weekend"]  = (F["dow"] >= 5).astype("int8")
    F["is_christmas"] = ((F["month"] == 12) & (F["dom"] == 25)).astype("int8")
    F["snap"]        = long["snap"].to_numpy().astype("int8")
    F["has_event"]   = long["event_name_1"].notna().to_numpy().astype("int8")

    # event distance: computed at date level (1,913 days), then tiled for every series
    ev_days = np.sort(((pd.DatetimeIndex(long.loc[long["event_name_1"].notna(), "date"].unique()) - dates[0]).days).to_numpy())
    d_days  = np.arange(n_dates)
    if len(ev_days):
        nxt_i = np.searchsorted(ev_days, d_days, side="left")
        nxt   = np.where(nxt_i < len(ev_days), ev_days[np.minimum(nxt_i, len(ev_days) - 1)] - d_days, CAL_CAP)
        prv_i = np.searchsorted(ev_days, d_days, side="right") - 1
        prv   = np.where(prv_i >= 0, d_days - ev_days[np.maximum(prv_i, 0)], CAL_CAP)
    else:
        nxt = prv = np.full(n_dates, CAL_CAP)
    F["days_to_next_event"]   = np.tile(np.minimum(nxt, CAL_CAP).astype("int8"), n_ids)
    F["days_since_last_event"] = np.tile(np.minimum(prv, CAL_CAP).astype("int8"), n_ids)
    G["calendar"] = ["dow", "month", "dom", "weekofyear", "is_weekend", "is_christmas", "snap",
                     "has_event", "event_type", "days_to_next_event", "days_since_last_event"]

    # ---------- 7) Price ----------
    P = _to_wide(long["sell_price"], ids, dates)
    F["sell_price"]         = _to_flat(P)
    F["price_change_1w"]    = _to_flat(P / P.shift(7) - 1)
    F["price_vs_past_mean"] = _to_flat(P / P.shift(h).expanding(min_periods=1).mean() - 1)
    G["price"] = ["sell_price", "price_change_1w", "price_vs_past_mean"]

    # ---------- 8) Series info ----------
    first_av = long["date"].where(long["is_available"]).groupby(long["id"]).transform("min")
    F["age_days"] = (long["date"] - first_av).dt.days.to_numpy().astype("float32")
    G["series_info"] = ["item_id", "dept_id", "cat_id", "store_id", "age_days"]

    # ---------- Assemble ----------
    out = pd.DataFrame({
        "id": long["id"], "date": long["date"],
        "sales": long["sales"].astype("float32"),
        "is_available": long["is_available"],
        "vol_tier": long["vol_tier"],                       # for error analysis only, not a feature
        "event_type": long["event_type_1"].fillna("none").astype("category"),
        "item_id": long["item_id"].astype("category"),
        "dept_id": long["dept_id"].astype("category"),
        "cat_id": long["cat_id"].astype("category"),
        "store_id": long["store_id"].astype("category"),
    })
    out = pd.concat([out, pd.DataFrame(F)], axis=1)
    return out, G


def model_rows(df, h):
    """Rows used for training/evaluation: available rows only, after removing the burn-in period where lags cannot exist yet."""
    burn_end = df["date"].min() + pd.Timedelta(days=h + 56)
    return df[df["is_available"] & (df["date"] >= burn_end)].reset_index(drop=True)
