"""Shared fixtures: a small synthetic panel, a tiny trained model and an API client. No M5 data is needed."""

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from retail_forecast.config import ModelSpec
from retail_forecast.demo import make_synthetic_panel, train_demo_model
from retail_forecast.models import load_forecaster
from retail_forecast.predict import ForecastService


@pytest.fixture(scope="session")
def panel():
    return make_synthetic_panel(n_items_per_cat=4, n_days=420, horizon=28, seed=1)


@pytest.fixture(scope="session")
def model_files(panel, tmp_path_factory):
    return train_demo_model(panel, tmp_path_factory.mktemp("model"), horizon=28, n_estimators=30)


@pytest.fixture(scope="session")
def forecaster(model_files):
    spec = ModelSpec("test_model", "lightgbm", model_files["model"], model_files["config"], model_files["metadata"], 28)
    return load_forecaster(spec)


@pytest.fixture(scope="session")
def service(panel, forecaster):
    return ForecastService(panel, forecaster)


@pytest.fixture(scope="session")
def client(service):
    with TestClient(create_app(service=service)) as c:
        yield c


@pytest.fixture(scope="session")
def series_id(service):
    return service.series_ids()[0]
