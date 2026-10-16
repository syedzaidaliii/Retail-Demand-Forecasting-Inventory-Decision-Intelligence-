"""Inventory decision logic: an order-up-to (base-stock) policy and a small simulator.

All operational numbers (lead time, review period, service level, costs) are *assumptions* that the
caller chooses; M5 contains no inventory data. The formulas are identical to the ones used in
Notebook 6.

    target level S      = expected demand over (L + R) days + safety stock
    safety stock        = z * sigma_daily * sqrt(L + R)
    order quantity      = ceil(max(0, S - inventory position)),   position = on-hand + on-order
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

__all__ = ["CostAssumptions", "OrderRecommendation", "z_value", "protection_demand", "safety_stock",
           "recommend_order", "simulate_block"]


@dataclass(frozen=True)
class CostAssumptions:
    cost_fraction: float = 0.70       # unit cost as a share of the selling price
    hold_rate_annual: float = 0.25    # holding cost per year as a share of unit cost
    penalty_multiplier: float = 1.0   # lost-sale cost = margin x multiplier
    order_cost: float = 2.0           # fixed cost per order


@dataclass(frozen=True)
class OrderRecommendation:
    order_quantity: int
    target_level: float
    expected_demand: float
    safety_stock: float
    inventory_position: float
    z_value: float
    protection_period_days: int
    sigma_daily: float

    def as_dict(self) -> dict:
        return asdict(self)


def z_value(service_level: float) -> float:
    """Normal quantile for a target service level in [0.5, 1)."""
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1)")
    return NormalDist().inv_cdf(service_level)


def protection_demand(forecast, days: int) -> float:
    """Expected demand over ``days`` days. Beyond the forecast, the mean of the last 7 forecast days is used."""
    f = np.asarray(forecast, dtype=float)
    if days < 1:
        raise ValueError("days must be >= 1")
    if days <= len(f):
        return float(f[:days].sum())
    tail = float(f[-7:].mean())
    return float(f.sum() + tail * (days - len(f)))


def safety_stock(z: float, sigma_daily: float, days: int) -> float:
    return float(z * sigma_daily * math.sqrt(days))


def recommend_order(forecast, on_hand: float, on_order: float, lead_time_days: int, review_period_days: int,
                    service_level: float, sigma_daily: float) -> OrderRecommendation:
    """Order quantity to place today under an order-up-to policy."""
    if on_hand < 0 or on_order < 0:
        raise ValueError("on_hand and on_order must be >= 0")
    if sigma_daily < 0:
        raise ValueError("sigma_daily must be >= 0")
    if lead_time_days < 0 or review_period_days < 1:
        raise ValueError("lead_time_days must be >= 0 and review_period_days >= 1")

    days = lead_time_days + review_period_days
    expected = protection_demand(forecast, days)
    z = z_value(service_level)
    ss = safety_stock(z, sigma_daily, days)
    target = expected + ss
    position = on_hand + on_order
    qty = int(math.ceil(round(max(0.0, target - position), 9)))
    return OrderRecommendation(order_quantity=qty, target_level=target, expected_demand=expected, safety_stock=ss,
                               inventory_position=position, z_value=z, protection_period_days=days,
                               sigma_daily=sigma_daily)


def simulate_block(demand, forecast, ma_rate, sigma, price, policy: str, lead_time: int, review_period: int,
                   service_level: float, costs: CostAssumptions | None = None) -> dict[str, np.ndarray]:
    """Day-by-day simulation of one block for many series at once.

    Parameters are arrays of shape ``(T, n)`` for ``demand``/``forecast`` and ``(n,)`` for the rest.
    ``policy`` is ``"A"`` (flat moving-average rate ``ma_rate``), ``"B"`` (the supplied forecast) or
    ``"oracle"`` (actual demand, no safety stock). Unmet demand is lost. Orders that would arrive after
    the block are not placed.
    """
    costs = costs or CostAssumptions()
    D = np.asarray(demand, dtype=float)
    T, n = D.shape
    ma = np.asarray(ma_rate, dtype=float)
    sg = np.asarray(sigma, dtype=float)
    v = np.asarray(price, dtype=float)
    L, R = int(lead_time), int(review_period)
    hz = L + R
    z = z_value(service_level)

    unit_cost = costs.cost_fraction * v
    hold_u = costs.hold_rate_annual / 365.0 * unit_cost
    pen_u = (v - unit_cost) * costs.penalty_multiplier

    if policy == "A":
        fx = np.tile(ma, (T + hz, 1))
        ss = z * sg * math.sqrt(hz)
    elif policy == "B":
        f = np.asarray(forecast, dtype=float)
        fx = np.vstack([f, np.tile(f[-7:].mean(0), (hz, 1))])
        ss = z * sg * math.sqrt(hz)
    elif policy == "oracle":
        fx = np.vstack([D, np.tile(D[-7:].mean(0), (hz, 1))])
        ss = np.zeros(n)
    else:
        raise ValueError(f"unknown policy '{policy}'")
    cum = np.vstack([np.zeros((1, n)), np.cumsum(fx, axis=0)])

    onhand = np.ceil(ma * hz)
    pipe = np.zeros((T + L + 1, n))
    hold_cost = np.zeros(n)
    lost_units = np.zeros(n)
    stockout_days = np.zeros(n)
    inv_sum = np.zeros(n)
    orders = np.zeros(n)
    for d in range(T):
        onhand = onhand + pipe[d]
        if d % R == 0 and d + L < T:
            target = (cum[d + hz] - cum[d]) + ss
            position = onhand + pipe[d + 1:].sum(axis=0)
            qty = np.ceil(np.maximum(0.0, target - position))
            pipe[d + L] += qty
            orders += qty > 0
        sold = np.minimum(onhand, D[d])
        lost = D[d] - sold
        onhand = onhand - sold
        hold_cost += hold_u * onhand
        lost_units += lost
        stockout_days += lost > 0
        inv_sum += onhand

    order_cost = orders * costs.order_cost
    lost_cost = lost_units * pen_u
    return {
        "demand": D.sum(0), "lost_units": lost_units, "stockout_days": stockout_days, "avg_inventory_units": inv_sum / T,
        "orders": orders, "hold_cost": hold_cost, "order_cost": order_cost, "lost_cost": lost_cost,
        "total_cost": hold_cost + order_cost + lost_cost,
    }
