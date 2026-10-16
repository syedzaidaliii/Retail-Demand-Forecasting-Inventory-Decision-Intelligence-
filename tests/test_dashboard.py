"""Smoke test of the Streamlit dashboard against the in-process API."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


@pytest.fixture()
def dashboard(client, monkeypatch):
    import sys
    sys.path.insert(0, str(APP.parent))
    import api_client
    monkeypatch.setattr(api_client, "get_client", lambda: _Wrapper(client))
    at = AppTest.from_file(str(APP), default_timeout=60)
    return at


class _Wrapper:
    """Make the shared TestClient usable as a context manager without closing it."""

    def __init__(self, client):
        self.client = client

    def __enter__(self):
        return self.client

    def __exit__(self, *exc):
        return False


def test_dashboard_renders_forecast_and_recommendation(dashboard):
    at = dashboard.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert any(label.startswith("Total forecast") for label in labels)
    assert "Order quantity today" in labels


def test_dashboard_reacts_to_the_service_level(dashboard):
    at = dashboard.run()
    base = [m.value for m in at.metric if m.label == "Order quantity today"][0]
    slider = [s for s in at.sidebar.slider if s.label == "Target service level"][0]
    slider.set_value(0.99)
    at = at.run()
    higher = [m.value for m in at.metric if m.label == "Order quantity today"][0]
    assert int(higher.split()[0]) >= int(base.split()[0])
