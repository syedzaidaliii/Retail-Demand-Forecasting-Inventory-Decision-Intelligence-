"""The serving panel: one row per series per day, history plus the next ``horizon`` days.

History rows contain the observed ``sales``. Future rows (the forecast window) have ``sales = NaN``;
their calendar and price columns are known in advance. :func:`validate_panel` is the data contract
that both the preparation script and the service rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "vol_tier", "date", "event_name_1",
    "event_type_1", "snap", "sell_price", "is_available", "sales",
]


def validate_panel(df: pd.DataFrame, horizon: int | None = None) -> list[str]:
    """Return a list of problems (empty list = the panel satisfies the contract)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    problems: list[str] = []
    if df.duplicated(["id", "date"]).any():
        problems.append("duplicate (id, date) rows")
    if df.groupby("id").size().nunique() != 1:
        problems.append("series have different numbers of days")

    dates = pd.DatetimeIndex(np.sort(df["date"].unique()))
    if len(dates) > 1 and (dates[-1] - dates[0]).days != len(dates) - 1:
        problems.append("dates are not daily-continuous")

    sales = df["sales"]
    if (sales.dropna() < 0).any():
        problems.append("negative sales")
    if (df["sell_price"].dropna() < 0).any():
        problems.append("negative prices")
    if sales.notna().sum() == 0:
        problems.append("no observed sales")
        return problems

    origin = df.loc[sales.notna(), "date"].max()
    if df[(df["date"] <= origin) & sales.isna()].shape[0] > 0:
        problems.append("missing sales before the forecast origin")
    if df[(df["date"] > origin) & sales.notna()].shape[0] > 0:
        problems.append("future rows must not contain sales (leakage)")
    n_future = int((dates > origin).sum())
    if horizon is not None and n_future != horizon:
        problems.append(f"expected {horizon} future days after {origin.date()}, found {n_future}")
    return problems


def load_panel(path: str | Path, horizon: int | None = None) -> pd.DataFrame:
    """Read, type and validate a serving panel (raises ``ValueError`` if the contract is violated)."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["id"] = df["id"].astype(str)
    problems = validate_panel(df, horizon=horizon)
    if problems:
        raise ValueError("invalid serving panel: " + "; ".join(problems))
    return df.sort_values(["id", "date"]).reset_index(drop=True)
