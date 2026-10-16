"""Build the serving panel: the observed history plus the next 28 days of calendar and prices.

Inputs: the extended long table written by Notebook 5 (``m5_sample_long_ext.parquet``) and the original
M5 ``calendar.csv`` / ``sell_prices.csv``. M5 data may not be redistributed, so the output stays in ``data/``
(which is git-ignored).

    python scripts/prepare_serving_data.py --ext data/processed/m5_sample_long_ext.parquet ^
        --calendar data/raw/calendar.csv --prices data/raw/sell_prices.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.data import REQUIRED_COLUMNS, validate_panel  # noqa: E402


def build_panel(ext: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    ext = ext.copy()
    ext["date"] = pd.to_datetime(ext["date"])
    ext["id"] = ext["id"].astype(str)
    origin = ext["date"].max()

    states = ext["state_id"].unique()
    if len(states) != 1:
        raise ValueError(f"expected exactly one state, found {list(states)}")
    snap_col = f"snap_{states[0]}"

    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    fut_cal = cal[(cal["date"] > origin) & (cal["date"] <= origin + pd.Timedelta(days=horizon))]
    if len(fut_cal) != horizon:
        raise ValueError(f"calendar covers {len(fut_cal)} of the {horizon} future days after {origin.date()}")
    fut_cal = fut_cal[["date", "wm_yr_wk", "event_name_1", "event_type_1", snap_col]].rename(columns={snap_col: "snap"})

    meta = ext.drop_duplicates("id")[["id", "item_id", "dept_id", "cat_id", "store_id", "vol_tier"]]
    future = meta.merge(fut_cal, how="cross")
    pr = prices[prices["store_id"].isin(meta["store_id"].unique()) & prices["item_id"].isin(meta["item_id"].unique())]
    n0 = len(future)
    future = future.merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    assert len(future) == n0, "price join changed the number of rows"
    future["is_available"] = future["sell_price"].notna()
    future["sales"] = float("nan")

    hist = ext[REQUIRED_COLUMNS].copy()
    hist["sales"] = hist["sales"].astype("float32")
    panel = pd.concat([hist, future[REQUIRED_COLUMNS]], ignore_index=True)
    panel["sales"] = panel["sales"].astype("float32")
    return panel.sort_values(["id", "date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext", default="data/processed/m5_sample_long_ext.parquet")
    ap.add_argument("--calendar", default="data/raw/calendar.csv")
    ap.add_argument("--prices", default="data/raw/sell_prices.csv")
    ap.add_argument("--horizon", type=int, default=28)
    ap.add_argument("--out", default="data/serving/serving_panel.parquet")
    a = ap.parse_args()

    panel = build_panel(pd.read_parquet(a.ext), pd.read_csv(a.calendar), pd.read_csv(a.prices), a.horizon)
    problems = validate_panel(panel, horizon=a.horizon)
    if problems:
        raise SystemExit("serving panel violates the data contract: " + "; ".join(problems))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    origin = panel.loc[panel["sales"].notna(), "date"].max()
    print(f"saved {out}  |  series: {panel['id'].nunique()}  |  origin: {origin.date()}  |  "
          f"forecast window: {origin.date() + pd.Timedelta(days=1)} .. {panel['date'].max().date()}")


if __name__ == "__main__":
    main()
