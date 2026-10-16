"""Register a trained model as a versioned artefact and (optionally) make it the active model.

This is the "model update" step: after a new model is trained and evaluated (Notebooks 3-5), run

    python scripts/register_model.py --name lgbm_h28_v2 --model-file <model.txt> --config-file <config.json> ^
        --holdout-log reports/holdout_runs.jsonl --activate

The script copies the files to ``artifacts/<name>/``, writes ``metadata.json`` (checksum, date, holdout metrics),
verifies that the model can be loaded, and updates ``configs/model.yaml``. Commit the result; CI then runs the
model-compatibility tests against the new artefact before it can be merged and deployed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retail_forecast.config import ModelSpec  # noqa: E402
from retail_forecast.models import load_forecaster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="artefact name, e.g. lgbm_h28_v2")
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--version", default=None)
    ap.add_argument("--trained-through", default=None, help="last training day, e.g. 2016-04-24")
    ap.add_argument("--holdout-log", default=None, help="holdout_runs.jsonl; the first line is the official result")
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--config", default="configs/model.yaml")
    a = ap.parse_args()

    dest = ROOT / "artifacts" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.model_file, dest / "model.txt")
    shutil.copy(a.config_file, dest / "model_config.json")

    meta = {"name": a.name, "version": a.version or a.name, "trained_through": a.trained_through,
            "sha256": hashlib.sha256((dest / "model.txt").read_bytes()).hexdigest(),
            "registered_at_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
            "source_files": [Path(a.model_file).name, Path(a.config_file).name]}
    if a.holdout_log:
        first = json.loads(Path(a.holdout_log).read_text().splitlines()[0])
        meta["holdout"] = {k: first[k] for k in ("holdout", "WAPE_lgbm", "WAPE_ma28", "improvement_%", "model_sha256", "run_number") if k in first}
        if first.get("model_sha256") and first["model_sha256"] != meta["sha256"]:
            print("WARNING: the holdout log was produced by a different model file (checksum mismatch).")
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2))

    spec = ModelSpec(a.name, "lightgbm", dest / "model.txt", dest / "model_config.json", dest / "metadata.json", 28)
    model = load_forecaster(spec)           # raises if model file and config disagree
    print(f"model loads correctly: {len(model.feature_names)} features, horizon {model.horizon}")

    cfg_path = ROOT / a.config
    cfg = yaml.safe_load(cfg_path.read_text())
    rel = f"artifacts/{a.name}"
    cfg["models"][a.name] = {"type": "lightgbm", "path": f"{rel}/model.txt", "config": f"{rel}/model_config.json",
                             "metadata": f"{rel}/metadata.json", "horizon": model.horizon}
    if a.activate:
        cfg["active_model"] = a.name
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"registered '{a.name}'" + (" and set it as the active model" if a.activate else ""))


if __name__ == "__main__":
    main()
