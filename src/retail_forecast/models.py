"""Model plug-in layer.

The service only talks to the :class:`Forecaster` interface. To serve a different model (another
LightGBM, an ensemble, a foundation model, ...) implement the interface, register the class in
:data:`REGISTRY` and point ``configs/model.yaml`` to the new artefact. Nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ModelSpec


class Forecaster(Protocol):
    name: str
    version: str
    horizon: int
    feature_names: list[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return one non-negative forecast per row of ``features``."""

    def info(self) -> dict:
        """Metadata that is exposed through the API."""


class LightGBMForecaster:
    """A LightGBM booster trained on the features of :mod:`retail_forecast.features`."""

    def __init__(self, spec: ModelSpec):
        self.booster = lgb.Booster(model_file=str(spec.path))
        self.config = json.loads(spec.config.read_text(encoding="utf-8"))
        self.metadata = json.loads(spec.metadata.read_text(encoding="utf-8")) if spec.metadata and spec.metadata.exists() else {}

        self.feature_names = list(self.config["features"])
        if list(self.booster.feature_name()) != self.feature_names:
            raise ValueError("model file and model config disagree about the feature list")

        self.name = spec.name
        self.version = str(self.metadata.get("version", spec.name))
        self.horizon = int(self.config.get("horizon", spec.horizon))
        self.sha256 = hashlib.sha256(spec.path.read_bytes()).hexdigest()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in features.columns]
        if missing:
            raise KeyError(f"features missing for the model: {missing}")
        return np.clip(self.booster.predict(features[self.feature_names]), 0.0, None)

    def info(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "type": "lightgbm",
            "horizon_days": self.horizon,
            "objective": self.config.get("objective"),
            "n_features": len(self.feature_names),
            "training_window_days": self.config.get("train_days"),
            "sha256": self.sha256,
            "metadata": self.metadata,
        }


REGISTRY = {"lightgbm": LightGBMForecaster}


def load_forecaster(spec: ModelSpec) -> Forecaster:
    if spec.type not in REGISTRY:
        raise ValueError(f"unknown model type '{spec.type}' (known: {sorted(REGISTRY)})")
    return REGISTRY[spec.type](spec)
