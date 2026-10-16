"""Compatibility tests for every registered model artefact.

When a new model is registered (``scripts/register_model.py``) these tests check that it still fits the
feature pipeline, so an incompatible model cannot be merged and deployed. Artefacts that are not present in
the checkout (for example the real model on a fresh clone) are skipped.
"""

import numpy as np
import pytest

from retail_forecast.config import PROJECT_ROOT, load_settings
from retail_forecast.features import build_features
from retail_forecast.models import load_forecaster

SETTINGS = load_settings(PROJECT_ROOT / "configs" / "model.yaml")
SPECS = [s for s in SETTINGS.models.values() if s.path.exists() and s.config.exists()]


@pytest.mark.skipif(not SPECS, reason="no model artefacts registered in this checkout")
@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_registered_model_is_compatible_with_the_feature_pipeline(spec, panel):
    model = load_forecaster(spec)
    feats, _ = build_features(panel, model.horizon)
    missing = [c for c in model.feature_names if c not in feats.columns]
    assert not missing, f"{spec.name} needs features the pipeline does not produce: {missing}"
    pred = model.predict(feats.tail(200))
    assert np.isfinite(pred).all() and (pred >= 0).all()


def test_the_model_registry_knows_the_configured_types():
    from retail_forecast.models import REGISTRY
    assert all(s.type in REGISTRY for s in SETTINGS.models.values())
