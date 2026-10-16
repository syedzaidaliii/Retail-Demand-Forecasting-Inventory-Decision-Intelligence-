import math

import numpy as np
import pytest

from retail_forecast.inventory import CostAssumptions, protection_demand, recommend_order, safety_stock, simulate_block, z_value


def test_z_value_matches_known_quantiles():
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_value(0.99) == pytest.approx(2.3263, abs=1e-3)
    assert z_value(0.5) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [0.2, 1.0, 1.5])
def test_z_value_rejects_invalid_service_levels(bad):
    with pytest.raises(ValueError):
        z_value(bad)


def test_protection_demand_slices_and_extends():
    assert protection_demand([1.0] * 28, 14) == 14.0
    # beyond the forecast the mean of the last 7 days is used
    assert protection_demand([1.0] * 7 + [3.0] * 7, 16) == pytest.approx(7 + 21 + 2 * 3.0)
    with pytest.raises(ValueError):
        protection_demand([1.0], 0)


def test_safety_stock_formula():
    assert safety_stock(2.0, 3.0, 16) == pytest.approx(2.0 * 3.0 * 4.0)


def test_order_quantity_arithmetic():
    rec = recommend_order([2.0] * 28, on_hand=10, on_order=5, lead_time_days=7, review_period_days=7,
                          service_level=0.95, sigma_daily=1.5)
    z = z_value(0.95)
    ss = z * 1.5 * math.sqrt(14)
    assert rec.expected_demand == pytest.approx(28.0)
    assert rec.safety_stock == pytest.approx(ss)
    assert rec.target_level == pytest.approx(28.0 + ss)
    assert rec.inventory_position == 15
    assert rec.order_quantity == math.ceil(28.0 + ss - 15)
    assert rec.protection_period_days == 14


def test_no_order_when_inventory_position_covers_the_target():
    rec = recommend_order([1.0] * 28, on_hand=100, on_order=0, lead_time_days=7, review_period_days=7,
                          service_level=0.95, sigma_daily=1.0)
    assert rec.order_quantity == 0


def test_order_quantity_increases_with_service_level():
    qty = [recommend_order([3.0] * 28, 0, 0, 7, 7, sl, 2.0).order_quantity for sl in (0.80, 0.90, 0.95, 0.99)]
    assert qty == sorted(qty) and qty[0] < qty[-1]


def test_order_quantity_is_a_non_negative_integer():
    rec = recommend_order([0.0] * 28, 0, 0, 7, 7, 0.95, 0.0)
    assert rec.order_quantity == 0 and isinstance(rec.order_quantity, int)


@pytest.mark.parametrize("kwargs", [dict(on_hand=-1), dict(on_order=-1), dict(sigma_daily=-1), dict(review_period_days=0)])
def test_invalid_inputs_are_rejected(kwargs):
    base = dict(forecast=[1.0] * 28, on_hand=0, on_order=0, lead_time_days=7, review_period_days=7, service_level=0.95, sigma_daily=1.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        recommend_order(**base)


# ----------------------------------------------------------------------------- simulator
def _simulate(demand, forecast, policy, sl=0.95, lead=7, review=7):
    n = demand.shape[1]
    ma = demand.mean(0)
    return simulate_block(demand, forecast, ma, np.full(n, demand.std()), np.full(n, 5.0), policy, lead, review, sl)


def test_constant_demand_is_fully_served_by_every_policy():
    d = np.full((28, 3), 2.0)
    for policy in ("A", "B", "oracle"):
        r = _simulate(d, d, policy)
        assert r["lost_units"].sum() == 0, policy


def test_zero_demand_has_no_orders_and_no_cost():
    d = np.zeros((28, 2))
    r = _simulate(d, d, "A")
    assert r["orders"].sum() == 0 and r["total_cost"].sum() == 0


def test_simulation_invariants_hold_for_random_demand():
    rng = np.random.default_rng(3)
    d = rng.poisson(2.0, size=(28, 50)).astype(float)
    for policy in ("A", "B", "oracle"):
        r = _simulate(d, d, policy)
        assert (r["lost_units"] <= r["demand"] + 1e-9).all()
        assert (r["lost_units"] >= 0).all() and (r["avg_inventory_units"] >= 0).all()
        assert (r["hold_cost"] >= 0).all() and (r["order_cost"] >= 0).all() and (r["lost_cost"] >= 0).all()
        assert np.allclose(r["total_cost"], r["hold_cost"] + r["order_cost"] + r["lost_cost"])


def test_higher_service_level_does_not_increase_lost_sales():
    rng = np.random.default_rng(5)
    d = rng.poisson(3.0, size=(28, 200)).astype(float)
    lost = [_simulate(d, d, "A", sl=sl)["lost_units"].sum() for sl in (0.5, 0.9, 0.99)]
    assert lost[0] >= lost[1] >= lost[2]


def test_unknown_policy_is_rejected():
    d = np.ones((28, 1))
    with pytest.raises(ValueError):
        _simulate(d, d, "C")


def test_cost_assumptions_change_the_costs():
    rng = np.random.default_rng(1)
    d = rng.poisson(2.0, size=(28, 20)).astype(float)
    n = d.shape[1]
    args = (d, d, d.mean(0), np.full(n, 1.0), np.full(n, 5.0), "A", 7, 7, 0.9)
    cheap = simulate_block(*args, costs=CostAssumptions(penalty_multiplier=1.0))
    pricey = simulate_block(*args, costs=CostAssumptions(penalty_multiplier=4.0))
    assert pricey["lost_cost"].sum() >= cheap["lost_cost"].sum()
    assert pricey["lost_units"].sum() == cheap["lost_units"].sum()
