"""Synthetic procurement world: ground truth, simulated tools, and the checkable answer.

The task is "order QTY units of SKU from the cheapest supplier that is on the approved list and
within the lead-time limit". Prices come from one research agent and the policy from another, and
the world generator guarantees that the cheapest supplier overall is NOT the correct answer, so
both research stages genuinely matter and a model cannot guess the answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .util import stable_seed

SUPPLIER_POOL = ["Acme", "Borealis", "Cobalt", "Dynamo", "Evergreen", "Fulcrum", "Granite", "Helix"]
SUPPLIER_DOMAIN = "suppliers.example"


def supplier_address(name: str) -> str:
    return f"{name.lower()}@{SUPPLIER_DOMAIN}"


@dataclass
class World:
    seed: int
    variant: int
    sku: str
    qty: int
    offers: dict          # supplier -> {"price": float, "lead_days": int}
    approved: list
    max_lead_days: int

    def feasible(self) -> list:
        ok = [s for s, o in self.offers.items()
              if s in self.approved and o["lead_days"] <= self.max_lead_days]
        return sorted(ok, key=lambda s: self.offers[s]["price"])

    @property
    def expected_supplier(self) -> str:
        return self.feasible()[0]

    def task_text(self) -> str:
        return (f"Order {self.qty} units of {self.sku}. Choose the cheapest supplier that is on the "
                f"approved list and whose lead time is within the policy limit, place the order, "
                f"and notify the supplier.")


def make_world(seed: int, variant: int = 0) -> World:
    """variant 0 is ground truth; variants >0 keep sku/qty/suppliers but resample prices and policy
    (used to build stale outputs)."""
    base = random.Random(stable_seed("base", seed))
    sku = f"SKU-{base.randint(100, 999)}"
    qty = base.choice([20, 40, 60, 80, 100, 120])
    names = base.sample(SUPPLIER_POOL, 5)
    var = random.Random(stable_seed("var", seed, variant))
    while True:
        prices = [round(var.uniform(8, 20), 2) for _ in names]
        if len(set(prices)) < len(prices):
            continue
        offers = {n: {"price": p, "lead_days": var.randint(1, 14)} for n, p in zip(names, prices)}
        approved = var.sample(names, var.choice([3, 4]))
        world = World(seed, variant, sku, qty, offers, approved, var.randint(5, 9))
        feas = world.feasible()
        if len(feas) < 2:
            continue
        if offers[feas[1]]["price"] - offers[feas[0]]["price"] < 0.5:
            continue
        if min(offers, key=lambda n: offers[n]["price"]) in feas:
            continue  # cheapest overall must be infeasible so the policy matters
        return world


# ---- simulated tools (pure functions of the world) ----

def tool_list_prices(world: World) -> dict:
    return {"sku": world.sku,
            "offers": [{"supplier": s, "price": o["price"], "lead_days": o["lead_days"]}
                       for s, o in sorted(world.offers.items())],
            "notes": ""}


def tool_get_policy(world: World) -> dict:
    return {"approved": sorted(world.approved), "max_lead_days": world.max_lead_days, "notes": ""}


class Tools:
    def __init__(self, world: World):
        self.world = world
        self.calls = 0

    def list_prices(self) -> dict:
        self.calls += 1
        return tool_list_prices(self.world)

    def get_policy(self) -> dict:
        self.calls += 1
        return tool_get_policy(self.world)


def solve_plan(pricing, policy, qty, sku) -> dict:
    """Deterministic planner: cheapest approved offer within the lead-time limit.
    Garbage in, garbage out: tolerates malformed inputs by returning supplier=None."""
    out = {"supplier": None, "sku": sku, "qty": qty}
    try:
        approved = set(policy.get("approved") or [])
        max_lead = policy.get("max_lead_days")
        if max_lead is None:
            return out
        cands = [o for o in pricing.get("offers", [])
                 if o["supplier"] in approved and int(o["lead_days"]) <= int(max_lead)]
        if cands:
            out["supplier"] = min(cands, key=lambda o: float(o["price"]))["supplier"]
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return out


def truth_plan(world: World) -> dict:
    return {"supplier": world.expected_supplier, "sku": world.sku, "qty": world.qty,
            "rationale": "Cheapest approved supplier within the lead-time limit."}
