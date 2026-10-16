"""The forecasting service: features -> model -> forecast, plus the inventory recommendation.

The service is built once from a validated serving panel and a :class:`~retail_forecast.models.Forecaster`.
Forecasts are computed per series on demand (features are rebuilt for that series only) and cached.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import validate_panel
from .features import build_features
from .inventory import OrderRecommendation, recommend_order
from .models import Forecaster


class SeriesNotFound(KeyError):
    """The requested series id is not in the serving panel."""


class HorizonError(ValueError):
    """The requested horizon is outside what the model supports."""


@dataclass(frozen=True)
class ForecastResult:
    series_id: str
    origin: pd.Timestamp
    dates: list[pd.Timestamp]
    values: list[float]
    available: list[bool]
    model_name: str
    model_version: str


class ForecastService:
    def __init__(self, panel: pd.DataFrame, model: Forecaster, sigma_days: int = 56):
        problems = validate_panel(panel, horizon=model.horizon)
        if problems:
            raise ValueError("invalid serving panel: " + "; ".join(problems))
        self.model = model
        self.sigma_days = sigma_days
        panel = panel.sort_values(["id", "date"]).reset_index(drop=True)
        self.origin: pd.Timestamp = panel.loc[panel["sales"].notna(), "date"].max()
        self._groups = {k: g.reset_index(drop=True) for k, g in panel.groupby("id", sort=False)}
        self._series = (panel.drop_duplicates("id")[["id", "item_id", "dept_id", "cat_id", "store_id", "vol_tier"]]
                        .astype(str).reset_index(drop=True))
        self._full_forecast = functools.lru_cache(maxsize=512)(self._compute_full_forecast)

    # ------------------------------------------------------------------ lookups
    def series_ids(self) -> list[str]:
        return list(self._groups)

    def series_table(self) -> pd.DataFrame:
        return self._series.copy()

    def _group(self, series_id: str) -> pd.DataFrame:
        try:
            return self._groups[series_id]
        except KeyError as exc:
            raise SeriesNotFound(series_id) from exc

    # ------------------------------------------------------------------ forecasting
    def _compute_full_forecast(self, series_id: str):
        g = self._group(series_id)
        feats, _ = build_features(g, self.model.horizon)
        fut = feats[feats["date"] > self.origin]
        values = self.model.predict(fut)
        available = fut["is_available"].to_numpy(bool)
        values = np.where(available, values, 0.0)
        return list(fut["date"]), [float(v) for v in values], [bool(a) for a in available]

    def forecast(self, series_id: str, horizon: int | None = None) -> ForecastResult:
        horizon = self.model.horizon if horizon is None else int(horizon)
        if not 1 <= horizon <= self.model.horizon:
            raise HorizonError(f"horizon must be between 1 and {self.model.horizon}")
        dates, values, available = self._full_forecast(series_id)
        return ForecastResult(series_id=series_id, origin=self.origin, dates=dates[:horizon], values=values[:horizon],
                              available=available[:horizon], model_name=self.model.name, model_version=self.model.version)

    # ------------------------------------------------------------------ history and variability
    def history(self, series_id: str, days: int = 120) -> pd.DataFrame:
        g = self._group(series_id)
        h = g[(g["date"] <= self.origin) & g["is_available"]]
        return h[["date", "sales"]].tail(int(days)).reset_index(drop=True)

    def sigma_daily(self, series_id: str) -> float:
        g = self._group(series_id)
        recent = g[(g["date"] <= self.origin) & g["is_available"]].tail(self.sigma_days)["sales"]
        return float(recent.std(ddof=1)) if len(recent) >= 2 else 0.0

    # ------------------------------------------------------------------ inventory
    def recommend(self, series_id: str, on_hand: float, on_order: float, lead_time_days: int,
                  review_period_days: int, service_level: float) -> tuple[OrderRecommendation, ForecastResult]:
        result = self.forecast(series_id)
        rec = recommend_order(result.values, on_hand, on_order, lead_time_days, review_period_days, service_level,
                              self.sigma_daily(series_id))
        return rec, result

    # ------------------------------------------------------------------ metadata
    def info(self) -> dict:
        return {"origin": str(self.origin.date()), "n_series": len(self._groups), "model": self.model.info()}
