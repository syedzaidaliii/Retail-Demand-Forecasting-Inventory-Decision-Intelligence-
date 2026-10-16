"""Streamlit dashboard: demand forecast, order recommendation and the backtest results.

Run:  streamlit run dashboard/app.py        (the API must be running; set API_URL if it is not on localhost:8000)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_client  # noqa: E402

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports"))

st.set_page_config(page_title="Retail demand forecasting", layout="wide")
st.title("Retail demand forecasting and order recommendation")
st.caption("M5 (Walmart) data, California subset. Forecasts come from a LightGBM model; inventory numbers are simulated assumptions.")


def api(method: str, path: str, **kwargs):
    """Call the API and return (json, error_message)."""
    try:
        with api_client.get_client() as client:
            r = client.request(method, path, timeout=30.0, **kwargs)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            return None, f"{r.status_code}: {detail}"
        return r.json(), None
    except Exception as exc:  # network problems, API down, ...
        return None, f"{type(exc).__name__}: {exc}"


health, err = api("GET", "/health")
if err or health is None:
    st.error(f"Cannot reach the API. {err or ''}")
    st.stop()
if health["status"] != "ok":
    st.warning(f"The API is running but not ready: {health.get('detail')}")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Series")
store = st.sidebar.selectbox("Store", ["(all)", "CA_1", "CA_2", "CA_3", "CA_4"])
cat = st.sidebar.selectbox("Category", ["(all)", "FOODS", "HOBBIES", "HOUSEHOLD"])
query = st.sidebar.text_input("Search in id", "")
params = {"limit": 200}
if store != "(all)":
    params["store_id"] = store
if cat != "(all)":
    params["cat_id"] = cat
if query:
    params["q"] = query
series, err = api("GET", "/series", params=params)
if err or not series or not series["items"]:
    st.sidebar.warning("No series match the filters.")
    st.stop()
st.sidebar.caption(f"{series['total']} series match (showing up to 200)")
series_id = st.sidebar.selectbox("Series", [s["id"] for s in series["items"]])

st.sidebar.header("Forecast")
horizon = st.sidebar.slider("Horizon (days)", 7, 28, 28)

st.sidebar.header("Inventory assumptions")
lead = st.sidebar.slider("Lead time (days)", 0, 28, 7)
review = st.sidebar.slider("Review period (days)", 1, 28, 7)
service = st.sidebar.slider("Target service level", 0.80, 0.99, 0.95, 0.01)
on_hand = st.sidebar.number_input("On hand (units)", min_value=0.0, value=0.0, step=1.0)
on_order = st.sidebar.number_input("On order (units)", min_value=0.0, value=0.0, step=1.0)

tab_fc, tab_inv, tab_bt, tab_about = st.tabs(["Forecast", "Order recommendation", "Backtest results", "About and limitations"])

# ----------------------------------------------------------------------------- forecast
with tab_fc:
    fc, err = api("POST", "/forecast", json={"series_id": series_id, "horizon": horizon})
    hist, err_h = api("GET", f"/history/{series_id}", params={"days": 90})
    if err or fc is None:
        st.error(f"Forecast failed: {err}")
    else:
        f = pd.DataFrame(fc["forecast"])
        f["date"] = pd.to_datetime(f["date"])
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Total forecast, next {fc['horizon']} days", f"{fc['total_forecast']:.1f} units")
        c2.metric("Average per day", f"{fc['total_forecast'] / max(fc['horizon'], 1):.2f} units")
        c3.metric("Forecast starts", str(f['date'].min().date()))
        frames = []
        if hist and hist["points"]:
            h = pd.DataFrame(hist["points"])
            h["date"] = pd.to_datetime(h["date"])
            frames.append(h.rename(columns={"sales": "observed"}).set_index("date")["observed"])
        frames.append(f.set_index("date")["forecast"])
        st.line_chart(pd.concat(frames, axis=1))
        with st.expander("Forecast table"):
            st.dataframe(f.assign(date=f["date"].dt.date), hide_index=True, use_container_width=True)
        st.caption("Point forecasts of daily units. This version has no prediction interval. " + " ".join(fc["notes"]))

# ----------------------------------------------------------------------------- inventory
with tab_inv:
    rec, err = api("POST", "/inventory/recommendation", json={
        "series_id": series_id, "on_hand": on_hand, "on_order": on_order, "lead_time_days": lead,
        "review_period_days": review, "service_level": service})
    if err or rec is None:
        st.error(f"Recommendation failed: {err}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Order quantity today", f"{rec['order_quantity']} units")
        c2.metric("Target stock level", f"{rec['target_level']:.1f}")
        c3.metric("Expected demand (L+R days)", f"{rec['expected_demand']:.1f}")
        c4.metric("Safety stock", f"{rec['safety_stock']:.1f}")
        st.write(f"Inventory position: **{rec['inventory_position']:.1f}** units. "
                 f"Order arrives on **{rec['expected_arrival_date']}**. "
                 f"Protection period: **{rec['protection_period_days']}** days; daily sales std: **{rec['sigma_daily']:.2f}**.")
        with st.expander("How this is calculated"):
            for n in rec["notes"]:
                st.markdown(f"- {n}")
        st.info("These are simulation assumptions, not observed supplier or cost data.")

# ----------------------------------------------------------------------------- backtest
with tab_bt:
    log = REPORTS_DIR / "holdout_runs.jsonl"
    inv = REPORTS_DIR / "inventory_results.json"
    if log.exists():
        first = json.loads(log.read_text().splitlines()[0])
        st.subheader("Final holdout (official, run 1)")
        c1, c2, c3 = st.columns(3)
        c1.metric("LightGBM WAPE", f"{first['WAPE_lgbm']:.3f}")
        c2.metric("28-day moving average WAPE", f"{first['WAPE_ma28']:.3f}")
        c3.metric("Improvement", f"{first['improvement_%']:.1f}%")
        st.caption(f"Holdout period {first.get('holdout')}, {first.get('series')} series. Model checksum {str(first.get('model_sha256', ''))[:12]}...")
    else:
        st.info("Copy `holdout_runs.jsonl` from the notebooks into the `reports/` folder to show the holdout result here.")
    if inv.exists():
        res = json.loads(inv.read_text())
        st.subheader("Simulated inventory backtest (assumption-based)")
        fa, fb = pd.DataFrame(res["frontier_A"]).T, pd.DataFrame(res["frontier_B"]).T
        chart = pd.DataFrame({"A: moving average": fa["total_cost"].to_numpy(), "B: LightGBM": fb["total_cost"].to_numpy()},
                             index=[f"{float(i):.3f}" for i in fa.index])
        st.line_chart(chart)
        st.caption("Total simulated cost by target service level (lower is better). The curves nearly coincide: "
                   "a 5.7% better forecast gave only a small simulated cost difference.")
    else:
        st.info("Copy `inventory_results.json` into `reports/` to show the simulated inventory backtest.")

# ----------------------------------------------------------------------------- about
with tab_about:
    info, err = api("GET", "/model-info")
    if err or info is None:
        st.error(err)
    else:
        m = info["model"]
        st.write(f"**Model:** {m['name']} (version {m['version']}), horizon {m['horizon_days']} days, "
                 f"{m['n_features']} features. **Data up to:** {info['data_origin']}. **Series:** {info['n_series']}.")
        st.subheader("Limitations")
        for lim in info["limitations"]:
            st.markdown(f"- {lim}")
