import numpy as np
import pandas as pd
import pytest

from retail_forecast.data import load_panel, validate_panel


def test_valid_panel_has_no_problems(panel):
    assert validate_panel(panel, horizon=28) == []


def test_missing_column_is_reported(panel):
    assert "missing columns" in validate_panel(panel.drop(columns=["sell_price"]))[0]


def test_duplicates_are_reported(panel):
    bad = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    assert any("duplicate" in p for p in validate_panel(bad))


def test_a_missing_day_is_reported(panel):
    bad = panel.drop(index=panel.index[100])
    problems = validate_panel(bad)
    assert any("different numbers of days" in p for p in problems)


def test_negative_sales_are_reported(panel):
    bad = panel.copy()
    bad.loc[bad.index[10], "sales"] = -1.0
    assert "negative sales" in validate_panel(bad)


def test_future_rows_must_not_contain_sales(panel):
    bad = panel.copy()
    bad.loc[bad.index[-1], "sales"] = 5.0
    assert any("future rows" in p or "missing sales" in p for p in validate_panel(bad))


def test_wrong_horizon_is_reported(panel):
    assert any("expected 14 future days" in p for p in validate_panel(panel, horizon=14))


def test_load_panel_rejects_invalid_files(panel, tmp_path):
    bad = panel.copy()
    bad.loc[bad.index[5], "sales"] = -3.0
    path = tmp_path / "bad.parquet"
    bad.to_parquet(path)
    with pytest.raises(ValueError, match="invalid serving panel"):
        load_panel(path)


def test_load_panel_round_trip(panel, tmp_path):
    path = tmp_path / "ok.parquet"
    panel.to_parquet(path)
    loaded = load_panel(path, horizon=28)
    assert len(loaded) == len(panel)
    assert np.issubdtype(loaded["date"].dtype, np.datetime64)
