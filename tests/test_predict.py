import numpy as np
import pandas as pd
import pytest

from retail_forecast.features import build_features
from retail_forecast.predict import ForecastService, HorizonError, SeriesNotFound


def test_forecast_has_the_requested_length_and_contiguous_dates(service, series_id):
    res = service.forecast(series_id, 14)
    assert len(res.values) == len(res.dates) == len(res.available) == 14
    assert res.dates[0] == res.origin + pd.Timedelta(days=1)
    assert all((b - a).days == 1 for a, b in zip(res.dates, res.dates[1:]))


def test_forecast_values_are_finite_and_non_negative(service):
    for sid in service.series_ids():
        v = np.array(service.forecast(sid).values)
        assert np.isfinite(v).all() and (v >= 0).all()


def test_shorter_horizons_are_prefixes_of_the_full_forecast(service, series_id):
    full = service.forecast(series_id, 28).values
    assert service.forecast(series_id, 7).values == full[:7]


def test_unknown_series_and_bad_horizons_are_rejected(service, series_id):
    with pytest.raises(SeriesNotFound):
        service.forecast("does-not-exist")
    for h in (0, 29, -1):
        with pytest.raises(HorizonError):
            service.forecast(series_id, h)


def test_per_series_inference_equals_batch_features(service, forecaster, panel, series_id):
    """Building the features for one series must give the same forecast as building them for the whole panel."""
    feats, _ = build_features(panel, forecaster.horizon)
    fut = feats[(feats["id"] == series_id) & (feats["date"] > service.origin)]
    expected = np.where(fut["is_available"].to_numpy(bool), forecaster.predict(fut), 0.0)
    assert np.allclose(service.forecast(series_id).values, expected)


def test_history_ends_at_the_origin(service, series_id):
    h = service.history(series_id, 60)
    assert len(h) == 60 and h["date"].max() <= service.origin
    assert (h["sales"] >= 0).all()


def test_sigma_is_non_negative(service, series_id):
    assert service.sigma_daily(series_id) >= 0


def test_recommendation_is_consistent_with_the_forecast(service, series_id):
    rec, res = service.recommend(series_id, on_hand=0, on_order=0, lead_time_days=7, review_period_days=7, service_level=0.95)
    assert rec.order_quantity >= 0
    assert rec.expected_demand == pytest.approx(sum(res.values[:14]))


def test_service_rejects_a_panel_with_future_sales(panel, forecaster):
    bad = panel.copy()
    bad.loc[bad.index[-1], "sales"] = 3.0
    with pytest.raises(ValueError, match="invalid serving panel"):
        ForecastService(bad, forecaster)


def test_forecasts_do_not_use_information_after_the_origin(panel, forecaster):
    """Changing the future calendar/price of a *different* day range is allowed, but sales must not matter:
    the forecast of the same service built on a panel whose future sales are NaN is the production setting."""
    svc = ForecastService(panel, forecaster)
    assert panel.loc[panel["date"] > svc.origin, "sales"].isna().all()
