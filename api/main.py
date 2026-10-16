"""FastAPI service.

Run locally:   ``uvicorn api.main:app --reload``
Interactive docs: http://localhost:8000/docs

The service starts even when the model or the data are missing (``/health`` then reports ``degraded``
and the forecasting endpoints return 503), which makes container health checks and deployments easier to debug.
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from api import schemas
from retail_forecast import __version__
from retail_forecast.config import Settings, load_settings
from retail_forecast.data import load_panel
from retail_forecast.models import load_forecaster
from retail_forecast.predict import ForecastService, HorizonError, SeriesNotFound

LIMITATIONS = [
    "The model was trained on a stratified subset of California item-store series of the M5 dataset (about 3% of the series).",
    "Items that were not in the training data (new items) are not supported reliably.",
    "Forecasts are point forecasts of daily units; this version returns no prediction intervals.",
    "Prices and the calendar are assumed to be known in advance.",
    "Inventory recommendations use assumptions chosen by the caller (lead time, service level); M5 contains no inventory data.",
    "On the final holdout the model under-forecast total demand by about 7%.",
]
FORECAST_NOTES = [
    "Daily unit forecasts from a direct 28-day LightGBM model (Tweedie objective).",
    "Days on which the item has no price are marked available=false and forecast as 0.",
]


def create_app(settings: Settings | None = None, service: ForecastService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        app.state.error = None
        app.state.settings = settings
        if service is None:
            try:
                cfg = settings or load_settings()
                app.state.settings = cfg
                model = load_forecaster(cfg.active_spec)
                panel = load_panel(cfg.serving_panel, horizon=model.horizon)
                app.state.service = ForecastService(panel, model)
            except Exception as exc:  # keep the API alive in degraded mode
                app.state.error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="Retail Demand Forecasting API", version=__version__, lifespan=lifespan,
                  description="28-day demand forecasts and order recommendations for M5 item-store series.")

    def get_service(request: Request) -> ForecastService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail=f"service not ready: {request.app.state.error or 'model or data not loaded'}")
        return svc

    @app.get("/health", response_model=schemas.HealthResponse, tags=["system"])
    def health(request: Request):
        svc = request.app.state.service
        cfg = request.app.state.settings
        return schemas.HealthResponse(
            status="ok" if svc is not None else "degraded", model_loaded=svc is not None, data_loaded=svc is not None,
            active_model=(svc.model.name if svc else (cfg.active_model if cfg else None)), detail=request.app.state.error)

    @app.get("/model-info", response_model=schemas.ModelInfoResponse, tags=["system"])
    def model_info(svc: ForecastService = Depends(get_service)):
        info = svc.info()
        return schemas.ModelInfoResponse(model=info["model"], data_origin=svc.origin.date(), n_series=info["n_series"],
                                         limitations=LIMITATIONS)

    @app.get("/series", response_model=schemas.SeriesListResponse, tags=["data"])
    def list_series(q: str | None = Query(None, description="substring of the series id"), store_id: str | None = None,
                    cat_id: str | None = None, limit: int = Query(50, ge=1, le=500), svc: ForecastService = Depends(get_service)):
        t = svc.series_table()
        if q:
            t = t[t["id"].str.contains(q, case=False, regex=False)]
        if store_id:
            t = t[t["store_id"] == store_id]
        if cat_id:
            t = t[t["cat_id"] == cat_id]
        return schemas.SeriesListResponse(total=len(t), items=[schemas.SeriesItem(**r) for r in t.head(limit).to_dict("records")])

    @app.get("/history/{series_id}", response_model=schemas.HistoryResponse, tags=["data"])
    def history(series_id: str, days: int = Query(120, ge=7, le=730), svc: ForecastService = Depends(get_service)):
        try:
            h = svc.history(series_id, days)
        except SeriesNotFound:
            raise HTTPException(status_code=404, detail=f"unknown series '{series_id}'") from None
        return schemas.HistoryResponse(series_id=series_id, points=[
            schemas.HistoryPoint(date=r.date.date(), sales=float(r.sales)) for r in h.itertuples()])

    @app.post("/forecast", response_model=schemas.ForecastResponse, tags=["forecast"])
    def forecast(req: schemas.ForecastRequest, svc: ForecastService = Depends(get_service)):
        try:
            res = svc.forecast(req.series_id, req.horizon)
        except SeriesNotFound:
            raise HTTPException(status_code=404, detail=f"unknown series '{req.series_id}'") from None
        except HorizonError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        points = [schemas.ForecastPoint(date=d.date(), forecast=v, available=a)
                  for d, v, a in zip(res.dates, res.values, res.available)]
        return schemas.ForecastResponse(series_id=res.series_id, origin=res.origin.date(), horizon=len(points),
                                        model_name=res.model_name, model_version=res.model_version,
                                        total_forecast=float(sum(res.values)), forecast=points, notes=FORECAST_NOTES)

    @app.post("/inventory/recommendation", response_model=schemas.InventoryResponse, tags=["inventory"])
    def inventory_recommendation(req: schemas.InventoryRequest, svc: ForecastService = Depends(get_service)):
        try:
            rec, res = svc.recommend(req.series_id, req.on_hand, req.on_order, req.lead_time_days,
                                     req.review_period_days, req.service_level)
        except SeriesNotFound:
            raise HTTPException(status_code=404, detail=f"unknown series '{req.series_id}'") from None
        order_day = res.dates[0]
        notes = [
            "Order-up-to policy: target = expected demand over (lead time + review period) + safety stock.",
            f"Safety stock = z x sigma_daily x sqrt(L + R) = {rec.z_value:.3f} x {rec.sigma_daily:.3f} x sqrt({rec.protection_period_days}).",
            "Order quantity = ceil(max(0, target - (on_hand + on_order))).",
            "sigma_daily is the standard deviation of daily sales over the last 56 observed days (normal approximation; "
            "it fits intermittent demand only roughly).",
            "Lead time is assumed fixed; unmet demand is assumed lost. These are assumptions, not observed values.",
        ]
        return schemas.InventoryResponse(
            series_id=req.series_id, as_of=res.origin.date(), order_quantity=rec.order_quantity,
            expected_arrival_date=(order_day + dt.timedelta(days=req.lead_time_days)).date(), target_level=rec.target_level,
            expected_demand=rec.expected_demand, safety_stock=rec.safety_stock, inventory_position=rec.inventory_position,
            z_value=rec.z_value, protection_period_days=rec.protection_period_days, sigma_daily=rec.sigma_daily,
            assumptions=req.model_dump(exclude={"series_id"}), notes=notes, model_name=res.model_name,
            model_version=res.model_version)

    return app


app = create_app()
