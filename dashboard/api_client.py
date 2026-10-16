"""HTTP client used by the dashboard. The API address comes from the ``API_URL`` environment variable."""

import os

import httpx


def get_client() -> httpx.Client:
    return httpx.Client(base_url=os.environ.get("API_URL", "http://localhost:8000"), timeout=30.0)
