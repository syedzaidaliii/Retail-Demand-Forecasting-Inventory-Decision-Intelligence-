"""Configuration: which model is active, where the artefacts and the serving data live.

Settings come from a YAML file (``configs/model.yaml`` by default) and can be overridden with
environment variables, which is what Docker and CI use:

* ``RETAIL_CONFIG``         path to another YAML file
* ``RETAIL_ACTIVE_MODEL``   name of the model to serve
* ``RETAIL_SERVING_PANEL``  path of the serving panel (parquet)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    type: str
    path: Path
    config: Path
    metadata: Path | None
    horizon: int


@dataclass(frozen=True)
class Settings:
    config_path: Path
    active_model: str
    models: dict[str, ModelSpec]
    serving_panel: Path
    reports_dir: Path

    @property
    def active_spec(self) -> ModelSpec:
        if self.active_model not in self.models:
            raise KeyError(f"active model '{self.active_model}' is not defined in {self.config_path}")
        return self.models[self.active_model]


def _resolve(p: str | os.PathLike) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_path: str | os.PathLike | None = None) -> Settings:
    config_path = _resolve(config_path or os.environ.get("RETAIL_CONFIG", "configs/model.yaml"))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    models: dict[str, ModelSpec] = {}
    for name, m in raw["models"].items():
        meta = m.get("metadata")
        models[name] = ModelSpec(
            name=name,
            type=m["type"],
            path=_resolve(m["path"]),
            config=_resolve(m["config"]),
            metadata=_resolve(meta) if meta else None,
            horizon=int(m.get("horizon", 28)),
        )

    data = raw.get("data", {})
    return Settings(
        config_path=config_path,
        active_model=os.environ.get("RETAIL_ACTIVE_MODEL", raw["active_model"]),
        models=models,
        serving_panel=_resolve(os.environ.get("RETAIL_SERVING_PANEL", data.get("serving_panel", "data/serving/serving_panel.parquet"))),
        reports_dir=_resolve(data.get("reports_dir", "reports")),
    )
