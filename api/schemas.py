"""Typed request/response contracts of the API (validated by Pydantic)."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when model and data are loaded, otherwise 'degraded'")
    model_loaded: bool
    data_loaded: bool
    active_model: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model: dict
    data_origin: dt.date = Field(description="Last day with observed sales; forecasts start the day after")
    n_series: int
    limitations: list[str]


class SeriesItem(BaseModel):
    id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    vol_tier: str


class SeriesListResponse(BaseModel):
    total: int
    items: list[SeriesItem]


class HistoryPoint(BaseModel):
    date: dt.date
    sales: float


class HistoryResponse(BaseModel):
    series_id: str
    points: list[HistoryPoint]


class ForecastRequest(BaseModel):
    series_id: str = Field(min_length=1)
    horizon: int = Field(default=28, ge=1, le=28, description="Days ahead (1-28)")


class ForecastPoint(BaseModel):
    date: dt.date
    forecast: float = Field(ge=0)
    available: bool = Field(description="False if the item has no price on that day (not on sale); forecast is 0")


class ForecastResponse(BaseModel):
    series_id: str
    origin: dt.date
    horizon: int
    model_name: str
    model_version: str
    total_forecast: float
    forecast: list[ForecastPoint]
    notes: list[str]


class InventoryRequest(BaseModel):
    series_id: str = Field(min_length=1)
    on_hand: float = Field(default=0, ge=0, description="Units in stock now")
    on_order: float = Field(default=0, ge=0, description="Units already ordered and not yet received")
    lead_time_days: int = Field(default=7, ge=0, le=60)
    review_period_days: int = Field(default=7, ge=1, le=60)
    service_level: float = Field(default=0.95, ge=0.5, lt=1.0)


class InventoryResponse(BaseModel):
    series_id: str
    as_of: dt.date
    order_quantity: int
    expected_arrival_date: dt.date
    target_level: float
    expected_demand: float
    safety_stock: float
    inventory_position: float
    z_value: float
    protection_period_days: int
    sigma_daily: float
    assumptions: dict
    notes: list[str]
    model_name: str
    model_version: str
