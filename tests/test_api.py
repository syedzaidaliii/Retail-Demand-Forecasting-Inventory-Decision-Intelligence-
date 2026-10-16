import datetime as dt

import yaml
from fastapi.testclient import TestClient

from api.main import create_app
from retail_forecast.config import load_settings


def test_health_is_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["data_loaded"]


def test_model_info_exposes_metadata_and_limitations(client):
    body = client.get("/model-info").json()
    assert body["model"]["horizon_days"] == 28
    assert body["model"]["sha256"] and body["n_series"] > 0
    assert len(body["limitations"]) >= 3


def test_series_listing_and_filters(client):
    all_items = client.get("/series", params={"limit": 500}).json()
    assert all_items["total"] == len(all_items["items"]) > 0
    store = all_items["items"][0]["store_id"]
    only = client.get("/series", params={"store_id": store, "limit": 500}).json()
    assert 0 < only["total"] <= all_items["total"]
    assert all(i["store_id"] == store for i in only["items"])
    assert client.get("/series", params={"q": "zzz-no-match"}).json()["total"] == 0


def test_history_endpoint(client, series_id):
    body = client.get(f"/history/{series_id}", params={"days": 30}).json()
    assert len(body["points"]) == 30
    assert client.get("/history/unknown").status_code == 404


def test_forecast_endpoint(client, series_id):
    r = client.post("/forecast", json={"series_id": series_id, "horizon": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == 10 and len(body["forecast"]) == 10
    assert abs(body["total_forecast"] - sum(p["forecast"] for p in body["forecast"])) < 1e-6
    assert dt.date.fromisoformat(body["forecast"][0]["date"]) == dt.date.fromisoformat(body["origin"]) + dt.timedelta(days=1)


def test_forecast_errors(client, series_id):
    assert client.post("/forecast", json={"series_id": "nope"}).status_code == 404
    assert client.post("/forecast", json={"series_id": series_id, "horizon": 0}).status_code == 422
    assert client.post("/forecast", json={"series_id": series_id, "horizon": 29}).status_code == 422
    assert client.post("/forecast", json={}).status_code == 422


def test_inventory_recommendation(client, series_id):
    payload = {"series_id": series_id, "on_hand": 4, "on_order": 2, "lead_time_days": 5, "review_period_days": 7, "service_level": 0.9}
    r = client.post("/inventory/recommendation", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["order_quantity"] >= 0 and body["protection_period_days"] == 12
    assert body["inventory_position"] == 6
    assert body["target_level"] >= body["expected_demand"]
    first_day = dt.date.fromisoformat(body["as_of"]) + dt.timedelta(days=1)
    assert dt.date.fromisoformat(body["expected_arrival_date"]) == first_day + dt.timedelta(days=5)
    assert body["assumptions"]["service_level"] == 0.9 and len(body["notes"]) >= 3


def test_inventory_validation_errors(client, series_id):
    assert client.post("/inventory/recommendation", json={"series_id": series_id, "service_level": 1.0}).status_code == 422
    assert client.post("/inventory/recommendation", json={"series_id": series_id, "on_hand": -1}).status_code == 422
    assert client.post("/inventory/recommendation", json={"series_id": "nope"}).status_code == 404


def test_api_starts_in_degraded_mode_when_artefacts_are_missing(tmp_path, monkeypatch):
    cfg = {"active_model": "m", "models": {"m": {"type": "lightgbm", "path": "missing/model.txt", "config": "missing/c.json", "horizon": 28}},
           "data": {"serving_panel": "missing/panel.parquet"}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("RETAIL_CONFIG", str(path))
    with TestClient(create_app(settings=load_settings())) as c:
        health = c.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "degraded"
        assert health.json()["detail"]
        assert c.post("/forecast", json={"series_id": "x"}).status_code == 503
        assert c.get("/model-info").status_code == 503
