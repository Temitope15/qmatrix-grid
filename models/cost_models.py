"""Thermal generator cost models for the dispatch study.

Two cost models are evaluated on the same generator fleet so that the effect of
non-convexity is isolated.

Model A - dataset cubic (control):
    Cubic curve fitted from the piecewise-linear cost tables in `case.rop`. The
    ARPA-E GO data is near-convex (fitted cubic coefficient close to zero), so on
    this model the classical optimum is already attainable.

Model B - non-convex valve-point cubic:
    Thermal units with multiple steam-admission valves exhibit valve-point
    loading: a non-monotonic marginal cost. We represent this with a cubic
    C(P) = a P^3 + b P^2 + c P + d whose marginal cost m(P) = 3aP^2 + 2bP + c
    attains its minimum inside the operating range, making each unit's cost
    non-convex and the coupled dispatch multimodal.

Calibration is per unit from the dataset's real [Pmin, Pmax] ranges; only the
cost shape follows the valve-point dispatch benchmarks of Walters & Sheble
(IEEE T-PWRS 1993) and Sinha, Chakrabarti & Chattopadhyay (IEEE T-EC 2003).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from models.dispatch_dirac3 import Generator


# ----------------------------------------------------------------------- Model A
def dataset_cubic_generators(grid, cluster_buses: List[int]) -> List[Generator]:
    """Model A: cubic curves fitted from `case.rop` for generators in a cluster."""
    cost_map = grid.build_generator_cost_map()
    gens: List[Generator] = []
    for (bus, gen_id), gdata in grid.generators.items():
        if gdata["stat"] != 1 or bus not in cluster_buses:
            continue
        info = cost_map.get((bus, gen_id))
        if info is None:
            continue
        a, b, c, d = info["cubic_coeffs"]
        p_min, p_max = info["p_min"], info["p_max"]
        if p_max <= p_min:
            continue
        gens.append(Generator(bus, gen_id, float(p_min), float(p_max),
                              float(a), float(b), float(c), float(d)))
    return gens


# ----------------------------------------------------------------------- Model B
def calibrate_nonconvex_cubic(
    p_min: float,
    p_max: float,
    base_marginal: float,
    valve_frac: float = 0.35,
    ripple: float = 0.55,
    fixed_cost: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return (a, b, c, d) of a non-convex cubic with a valve-point marginal curve.

    Design goals (all verifiable):
      * Marginal cost m(P) = 3aP^2 + 2bP + c has its MINIMUM at
        P* = p_min + valve_frac * (p_max - p_min)  -> the valve operating point.
      * a > 0 (cost eventually convex/superlinear, physically correct at high load).
      * ``ripple`` sets how deep the marginal-cost dip is relative to ``base_marginal``
        at P*, i.e. how strong the non-convexity is (0 -> convex, larger -> more).
      * ``base_marginal`` (~$/MWh) anchors the marginal cost level at P*.

    With the vertex of the (upward) marginal-cost parabola pinned at P*:
        b = -3 a P*                          (vertex condition m'(P*)=0)
        m(P*) = c - 3 a P*^2 = base_marginal (dip level at the operating point)
    We pick ``a`` from the desired dip depth over the range so the curve is clearly
    non-monotonic but stays positive and increasing beyond P*.
    """
    rng = max(p_max - p_min, 1e-6)
    p_star = p_min + valve_frac * rng
    # Choose curvature 'a' so marginal cost rises by ~ripple*base_marginal from the
    # operating point P* to Pmax: m(Pmax) - m(P*) = 3a (Pmax - P*)^2 = ripple*base_marginal
    span_hi = max(p_max - p_star, 1e-6)
    a = ripple * base_marginal / (3.0 * span_hi**2)
    b = -3.0 * a * p_star
    c = base_marginal + 3.0 * a * p_star**2  # so m(P*) = base_marginal
    d = fixed_cost
    return float(a), float(b), float(c), float(d)


def literature_nonconvex_generators(
    grid,
    cluster_buses: List[int],
    valve_frac: float = 0.35,
    ripple: float = 0.55,
    seed: int = 7,
) -> List[Generator]:
    """Model B: non-convex cubic fleet built on the cluster's REAL generators.

    Each unit keeps its real [Pmin, Pmax] from the dataset. Cost *shape* is
    calibrated to a valve-point signature. Base marginal cost is spread across a
    realistic thermal band ($14-$38/MWh), with cheaper units given larger
    capacity-normalized efficiency, so the merit order is non-trivial.
    """
    rng = np.random.default_rng(seed)
    metas = []
    for (bus, gen_id), gdata in grid.generators.items():
        if gdata["stat"] != 1 or bus not in cluster_buses:
            continue
        p_min = float(gdata["pb"])
        p_max = float(gdata["pt"])
        if p_max <= p_min or p_max <= 0:
            continue
        metas.append((bus, gen_id, max(p_min, 0.0), p_max))
    # Assign base marginal cost inversely-ish to size (larger units cheaper), with
    # a little spread so units are distinguishable. Deterministic given seed.
    metas.sort(key=lambda m: m[3], reverse=True)
    gens: List[Generator] = []
    n = len(metas)
    for i, (bus, gen_id, p_min, p_max) in enumerate(metas):
        base_marginal = 14.0 + 24.0 * (i / max(n - 1, 1)) + float(rng.uniform(-1.5, 1.5))
        fixed = 100.0 + 3.0 * p_max
        a, b, c, d = calibrate_nonconvex_cubic(p_min, p_max, base_marginal,
                                               valve_frac=valve_frac, ripple=ripple,
                                               fixed_cost=fixed)
        gens.append(Generator(bus, gen_id, p_min, p_max, a, b, c, d))
    return gens


# ------------------------------------------------------------------ diagnostics
def characterize_convexity(gen: Generator, n: int = 400) -> Dict:
    """Report whether a unit's cubic cost is non-convex over its operating range.

    Returns the marginal-cost minimum location, whether it lies strictly inside
    [Pmin, Pmax] (the valve operating point => non-convex), and the second-derivative
    sign range. Used to *prove* the model is genuinely non-convex, not asserted.
    """
    P = np.linspace(gen.p_min, gen.p_max, n)
    second = 6 * gen.a * P + 2 * gen.b          # C''(P)
    marg = 3 * gen.a * P**2 + 2 * gen.b * P + gen.c  # C'(P)
    p_star = -gen.b / (3 * gen.a) if gen.a != 0 else None
    inside = (p_star is not None) and (gen.p_min < p_star < gen.p_max)
    return {
        "bus": gen.bus,
        "gen_id": gen.gen_id,
        "p_min": gen.p_min,
        "p_max": gen.p_max,
        "marginal_min_at_MW": round(float(p_star), 2) if p_star is not None else None,
        "marginal_min_inside_range": bool(inside),
        "is_nonconvex": bool(np.any(second < 0)),
        "marginal_cost_range": [round(float(marg.min()), 3), round(float(marg.max()), 3)],
        "min_second_derivative": round(float(second.min()), 6),
    }


if __name__ == "__main__":
    # Show the calibration produces a genuinely non-convex, valve-point-like unit.
    a, b, c, d = calibrate_nonconvex_cubic(50, 300, base_marginal=20.0, ripple=0.55)
    g = Generator(1, "demo", 50, 300, a, b, c, d)
    info = characterize_convexity(g)
    print("Calibrated cubic (a,b,c,d) =", [round(x, 6) for x in (a, b, c, d)])
    for k, v in info.items():
        print(f"  {k}: {v}")
