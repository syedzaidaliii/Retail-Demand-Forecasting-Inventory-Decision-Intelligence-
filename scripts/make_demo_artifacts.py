"""Create a synthetic serving panel and a tiny model so the stack can run without the M5 files.

    python scripts/make_demo_artifacts.py

Then use the demo configuration (PowerShell):  $env:RETAIL_CONFIG = "configs/demo.yaml"
The data and the model are synthetic; the numbers are meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.data import validate_panel  # noqa: E402
from retail_forecast.demo import make_synthetic_panel, train_demo_model  # noqa: E402


def main() -> None:
    panel = make_synthetic_panel(n_items_per_cat=6, n_days=500)
    problems = validate_panel(panel, horizon=28)
    if problems:
        raise SystemExit(f"synthetic panel violates the data contract: {problems}")
    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "serving_panel.parquet", index=False)
    paths = train_demo_model(panel, ROOT / "artifacts" / "demo")
    print(f"serving panel: {out / 'serving_panel.parquet'}  ({panel['id'].nunique()} series)")
    for k, v in paths.items():
        print(f"{k}: {v}")
    print("Use it with:  RETAIL_CONFIG=configs/demo.yaml")


if __name__ == "__main__":
    main()
