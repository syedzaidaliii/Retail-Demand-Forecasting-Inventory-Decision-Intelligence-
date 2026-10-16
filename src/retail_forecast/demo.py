"""Synthetic M5-like data and a tiny model.

Used by the tests and by ``scripts/make_demo_artifacts.py`` so the whole stack (API, dashboard, Docker)
can be run without the real M5 files. **The data is synthetic; its numbers mean nothing.**
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features, model_rows

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "event_type"]


def make_synthetic_panel(n_items_per_cat: int = 4, n_days: int = 420, horizon: int = 28, seed: int = 0,
                         start: str = "2015-01-01") -> pd.DataFrame:
    """A serving panel (history + ``horizon`` future days) with the M5 column layout."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days + horizon)
    origin_idx = n_days - 1
    dow_effect = np.array([0.9, 0.85, 0.85, 0.9, 1.0, 1.35, 1.4])[dates.dayofweek]
    snap = (dates.day <= 10).astype("int64")
    event = np.where((dates.month == 12) & (dates.day == 25), "Christmas", None)
    event_type = np.where(event == "Christmas", "National", None)

    rows = []
    for cat in ["FOODS", "HOBBIES"]:
        for k in range(n_items_per_cat):
            item = f"{cat}_1_{k:03d}"
            base = float(rng.choice([0.3, 1.0, 3.0, 8.0]))
            launch = int(rng.integers(40, 120)) if k % 3 == 2 else 0
            price0 = float(rng.uniform(1.0, 12.0))
            for store in ["CA_1", "CA_2"]:
                lam = base * dow_effect * (1 + 0.0004 * np.arange(len(dates)))
                sales = rng.poisson(lam).astype("float64")
                sales[(dates.month == 12) & (dates.day == 25)] = 0
                week = np.arange(len(dates)) // 7
                discount = np.where(rng.random(week.max() + 1) < 0.05, 0.85, 1.0)[week]
                price = np.round(price0 * discount, 2)
                available = np.arange(len(dates)) >= launch
                price = np.where(available, price, np.nan)
                sales = np.where(np.arange(len(dates)) < launch, 0.0, sales)
                sales[origin_idx + 1:] = np.nan                      # the forecast window is unknown
                rows.append(pd.DataFrame({
                    "id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": f"{cat}_1", "cat_id": cat,
                    "store_id": store, "vol_tier": "high" if base >= 3 else ("mid" if base >= 1 else "low"),
                    "date": dates, "event_name_1": event, "event_type_1": event_type, "snap": snap,
                    "sell_price": price, "is_available": available & ~np.isnan(price), "sales": sales,
                }))
    panel = pd.concat(rows, ignore_index=True)
    panel["sales"] = panel["sales"].astype("float32")
    return panel.sort_values(["id", "date"]).reset_index(drop=True)


def train_demo_model(panel: pd.DataFrame, out_dir: str | Path, horizon: int = 28, n_estimators: int = 40,
                     name: str = "demo_lgbm") -> dict[str, Path]:
    """Train a small LightGBM on the synthetic panel and write the same artefact files as the real model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = panel.loc[panel["sales"].notna(), "date"].max()

    feats, groups = build_features(panel, horizon)
    feature_names = [c for g in groups.values() for c in g]
    rows = model_rows(feats, horizon)
    rows = rows[rows["date"] <= origin]
    cats = [c for c in feature_names if c in CATEGORICAL]

    params = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
                  n_estimators=n_estimators, random_state=42, verbose=-1)
    model = lgb.LGBMRegressor(**params)
    model.fit(rows[feature_names], rows["sales"], categorical_feature=cats)

    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))
    config = {"horizon": horizon, "objective": "tweedie", "train_days": int(rows["date"].nunique()), "params": params,
              "final_variant": "demo", "features": feature_names, "categorical": cats}
    config_path = out_dir / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    metadata = {"name": name, "version": f"{name}-synthetic", "synthetic": True, "trained_through": str(origin.date()),
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "note": "Trained on synthetic data. Not a real forecasting model."}
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return {"model": model_path, "config": config_path, "metadata": meta_path}
