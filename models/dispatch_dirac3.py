"""Non-convex economic dispatch on QCi Dirac-3 (Entropy Quantum Computing).

Solves the non-convex thermal generator dispatch problem on Dirac-3 and compares
it against classical convex and global baselines on the same instance.

Economic dispatch minimizes generation cost subject to demand:

    minimize   sum_i C_i(P_i)
    subject to sum_i P_i = D            (power balance)
               Pmin_i <= P_i <= Pmax_i  (generator limits)

The problem maps directly onto Dirac-3's continuous encoding: the balance
constraint sum_i P_i = D is the device sum_constraint, and with a cubic cost
C_i(P) = a P^3 + b P^2 + c P + d the objective is a degree-3 polynomial (Dirac-3
samples polynomial Hamiltonians up to degree 5). A cubic term makes the cost
non-convex, so a convex LP/QP baseline must relax it.

Baselines on the identical instance:
- Convex QP (drops the cubic term), costed on the true cubic curve.
- Multistart SLSQP on the true non-convex cost (global reference).

Non-convex cost formulation follows the valve-point dispatch benchmarks of
Walters & Sheble (1993) and Sinha et al. (2003).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.optimize as opt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# eqc-models' internal QciClient reads QCI_TOKEN / QCI_API_URL, whereas our .env
# uses the QCI_API_TOKEN name. Bridge them so both clients authenticate.
if os.getenv("QCI_API_TOKEN") and not os.getenv("QCI_TOKEN"):
    os.environ["QCI_TOKEN"] = os.environ["QCI_API_TOKEN"]


@dataclass
class Generator:
    """A dispatchable thermal generator with a non-convex cubic cost curve.

    Cost ($/h):  C(P) = a*P^3 + b*P^2 + c*P + d
    plus an optional valve-point ripple term used only for the physical
    "true cost" reference:  + |e * sin(f * (Pmin - P))|
    """
    bus: int
    gen_id: str
    p_min: float
    p_max: float
    a: float  # cubic coefficient ($/MW^3 h) -- source of non-convexity
    b: float  # quadratic ($/MW^2 h)
    c: float  # linear ($/MW h)
    d: float  # fixed ($/h)
    e: float = 0.0  # valve-point amplitude ($/h)
    f: float = 0.0  # valve-point frequency (1/MW)

    def cost(self, p: float, valve: bool = False) -> float:
        """Cost of dispatching this unit at output ``p`` MW."""
        val = self.a * p**3 + self.b * p**2 + self.c * p + self.d
        if valve and self.e != 0.0:
            val += abs(self.e * np.sin(self.f * (self.p_min - p)))
        return val


@dataclass
class DispatchResult:
    method: str
    dispatch: np.ndarray
    total_cost: float
    demand: float
    served: float
    balance_error: float
    runtime_sec: float
    feasible: bool
    extra: dict = field(default_factory=dict)


class NonConvexDispatch:
    """Non-convex economic dispatch solver: Dirac-3 EQC vs classical baselines.

    Parameters
    ----------
    generators : list[Generator]
        The dispatchable fleet (typically the generators of one microgrid cluster).
    """

    def __init__(self, generators: Sequence[Generator]):
        self.generators = list(generators)
        self.n = len(self.generators)
        self.p_min = np.array([g.p_min for g in self.generators])
        self.p_max = np.array([g.p_max for g in self.generators])

    # ------------------------------------------------------------------ costs
    def total_cost(self, dispatch: np.ndarray, valve: bool = False) -> float:
        return float(sum(g.cost(p, valve=valve) for g, p in zip(self.generators, dispatch)))

    # ------------------------------------------------- Dirac-3 polynomial encoding
    def build_polynomial(self) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the cubic dispatch objective as an eqc-models polynomial.

        We use the shifted variable  P_i = Pmin_i + y_i  with  y_i in [0, range_i],
        so the lower bound Pmin is satisfied by construction (y >= 0) and the
        device sum_constraint enforces  sum_i y_i = Demand - sum_i Pmin  (see
        ``solve_dirac3``). Expanding the cubic C(Pmin+y) in y:

            C = a y^3 + (3a*Pmin + b) y^2 + (3a*Pmin^2 + 2b*Pmin + c) y + const

        Variables are 1-indexed (Dirac-3 convention); index 0 pads lower-degree
        monomials. Generator g -> variable (g+1). The constant drops out of argmin.
        """
        coeffs: List[float] = []
        indices: List[List[int]] = []
        for g_idx, g in enumerate(self.generators):
            v = g_idx + 1  # 1-indexed variable id
            pmn = g.p_min
            lin = 3 * g.a * pmn**2 + 2 * g.b * pmn + g.c
            quad = 3 * g.a * pmn + g.b
            cub = g.a
            if lin != 0.0:
                coeffs.append(lin)
                indices.append([0, 0, v])
            if quad != 0.0:
                coeffs.append(quad)
                indices.append([0, v, v])
            if cub != 0.0:
                coeffs.append(cub)
                indices.append([v, v, v])
        return np.array(coeffs, dtype=float), np.array(indices, dtype=int)

    def solve_dirac3(
        self,
        demand: float,
        relaxation_schedule: int = 2,
        num_samples: int = 5,
        mean_photon_number: float | None = None,
        job_name: str = "QMatrix_NonConvexDispatch",
    ) -> DispatchResult:
        """Solve the dispatch on real Dirac-3 hardware via eqc-models.

        The power-balance constraint sum_i P_i = demand is enforced through the
        device ``sum_constraint``. Per-unit Pmax limits are respected by scaling
        the problem so the sum constraint equals demand and clamping the returned
        sample to [Pmin, Pmax] before costing (post-hoc feasibility repair).
        """
        if not (os.getenv("QCI_TOKEN") or os.getenv("QCI_API_TOKEN")):
            raise RuntimeError(
                "No QCi Dirac-3 token found. Set QCI_API_TOKEN in your .env "
                "(copy .env.example). QCi finalists receive a token via qBraid. "
                "The cached 10-scenario results in doc/stats_dispatch.json can be "
                "viewed without a token; re-running the live job requires one."
            )
        from eqc_models.base import PolynomialModel
        from eqc_models.solvers import Dirac3ContinuousCloudSolver

        range_i = self.p_max - self.p_min
        reduced_demand = float(demand - self.p_min.sum())
        if reduced_demand < -1e-6 or reduced_demand > range_i.sum() + 1e-6:
            raise ValueError(
                f"Demand {demand:.1f} MW outside feasible band "
                f"[{self.p_min.sum():.1f}, {self.p_max.sum():.1f}] MW."
            )

        coeffs, indices = self.build_polynomial()
        model = PolynomialModel(coeffs, indices)
        # Dirac-3 requires integer per-variable upper bounds (quasi-continuous
        # encoding). Bound the SHIFTED variable y_i by its range (Pmax-Pmin);
        # the sum_constraint enforces sum_i y_i = Demand - sum_i Pmin.
        model.upper_bound = np.ceil(range_i).astype(int)

        solver = Dirac3ContinuousCloudSolver()
        t0 = time.time()
        response = solver.solve(
            model,
            name=job_name,
            sum_constraint=reduced_demand,
            relaxation_schedule=relaxation_schedule,
            num_samples=num_samples,
            mean_photon_number=mean_photon_number,
        )
        runtime = time.time() - t0

        samples = np.array(response["results"]["solutions"], dtype=float)
        energies = np.array(response["results"].get("energies", []), dtype=float)
        # Each sample is y (>=0, sum=reduced_demand). Recover P = Pmin + y. Only a
        # tiny numerical clip is needed at the upper bound; balance is preserved.
        best = None
        for s in samples:
            y = np.clip(s[: self.n], 0.0, range_i)
            disp = self._project_balance(self.p_min + y, demand)
            cost = self.total_cost(disp)
            if best is None or cost < best[1]:
                best = (disp, cost, s[: self.n])
        disp, cost, raw = best
        return DispatchResult(
            method="dirac3_eqc",
            dispatch=disp,
            total_cost=cost,
            demand=demand,
            served=float(disp.sum()),
            balance_error=float(abs(disp.sum() - demand)),
            runtime_sec=round(runtime, 3),
            feasible=bool(np.all(disp >= self.p_min - 1e-6) and np.all(disp <= self.p_max + 1e-6)),
            extra={
                "raw_sample": raw.tolist(),
                "device_energy": float(energies.min()) if energies.size else None,
                "num_samples": num_samples,
                "relaxation_schedule": relaxation_schedule,
            },
        )

    # --------------------------------------------------------- classical baselines
    def solve_convex_qp(self, demand: float) -> DispatchResult:
        """Convex baseline: drop the cubic term and solve the resulting QP.

        Represents a classical convex optimizer that cannot model the cubic
        curvature. The resulting dispatch is costed on the true cubic curve.
        """
        t0 = time.time()
        b = np.array([g.b for g in self.generators])
        c = np.array([g.c for g in self.generators])

        def obj(p):
            return float(np.sum(b * p**2 + c * p))

        def jac(p):
            return 2 * b * p + c

        cons = [{"type": "eq", "fun": lambda p: np.sum(p) - demand, "jac": lambda p: np.ones_like(p)}]
        bounds = list(zip(self.p_min, self.p_max))
        x0 = self._feasible_start(demand)
        res = opt.minimize(obj, x0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                           options={"maxiter": 500, "ftol": 1e-9})
        runtime = time.time() - t0
        disp = np.clip(res.x, self.p_min, self.p_max)
        return DispatchResult(
            method="convex_qp",
            dispatch=disp,
            total_cost=self.total_cost(disp),  # costed on the true cubic curve
            demand=demand,
            served=float(disp.sum()),
            balance_error=float(abs(disp.sum() - demand)),
            runtime_sec=round(runtime, 4),
            feasible=bool(res.success),
            extra={"convex_objective_value": float(res.fun)},
        )

    def solve_global_reference(self, demand: float, n_starts: int = 40, seed: int = 0) -> DispatchResult:
        """Global-quality reference: multistart SLSQP on the TRUE non-convex cost.

        Serves as the best-known optimum for measuring how close each method gets.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)
        bounds = list(zip(self.p_min, self.p_max))
        cons = [{"type": "eq", "fun": lambda p: np.sum(p) - demand, "jac": lambda p: np.ones_like(p)}]

        def obj(p):
            return self.total_cost(p)

        best = None
        for k in range(n_starts):
            x0 = self._feasible_start(demand, jitter=(k > 0), rng=rng)
            res = opt.minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP",
                               options={"maxiter": 800, "ftol": 1e-9})
            if res.success:
                disp = np.clip(res.x, self.p_min, self.p_max)
                cost = self.total_cost(disp)
                if best is None or cost < best[1]:
                    best = (disp, cost)
        runtime = time.time() - t0
        if best is None:
            disp = self._feasible_start(demand)
            best = (disp, self.total_cost(disp))
        disp, cost = best
        return DispatchResult(
            method="global_multistart",
            dispatch=disp,
            total_cost=cost,
            demand=demand,
            served=float(disp.sum()),
            balance_error=float(abs(disp.sum() - demand)),
            runtime_sec=round(runtime, 4),
            feasible=True,
            extra={"n_starts": n_starts},
        )

    # ------------------------------------------------------------------ helpers
    def _project_balance(self, disp: np.ndarray, demand: float, iters: int = 8) -> np.ndarray:
        """Adjust a dispatch to sum exactly to demand while staying in [Pmin, Pmax].

        Distributes the residual across units in proportion to their remaining
        headroom (residual > 0) or available down-room (residual < 0). Preserves
        feasibility; used to remove tiny numerical imbalance from device samples.
        """
        disp = np.clip(disp.astype(float), self.p_min, self.p_max)
        for _ in range(iters):
            residual = demand - disp.sum()
            if abs(residual) < 1e-6:
                break
            if residual > 0:
                room = self.p_max - disp
            else:
                room = disp - self.p_min
            total = room.sum()
            if total < 1e-9:
                break
            disp = np.clip(disp + residual * room / total, self.p_min, self.p_max)
        return disp

    def _feasible_start(self, demand: float, jitter: bool = False, rng=None) -> np.ndarray:
        """Return a dispatch summing to demand within [Pmin, Pmax] (proportional fill)."""
        if demand <= self.p_min.sum():
            return self.p_min.copy()
        if demand >= self.p_max.sum():
            return self.p_max.copy()
        headroom = self.p_max - self.p_min
        w = headroom.copy()
        if jitter and rng is not None:
            w = w * rng.uniform(0.2, 1.0, size=self.n)
        w = w / w.sum() if w.sum() > 0 else np.ones(self.n) / self.n
        return self.p_min + w * (demand - self.p_min.sum())


if __name__ == "__main__":
    # Standard non-convex ED sanity instance (small, classical-only).
    gens = [
        Generator(1, "A", 50, 200, a=1.2e-4, b=0.02, c=18.0, d=100.0),
        Generator(2, "B", 30, 150, a=2.0e-4, b=0.015, c=20.0, d=80.0),
        Generator(3, "C", 40, 180, a=0.9e-4, b=0.025, c=17.0, d=120.0),
    ]
    disp = NonConvexDispatch(gens)
    D = 400.0
    conv = disp.solve_convex_qp(D)
    glob = disp.solve_global_reference(D)
    print(f"Convex QP   : cost=${conv.total_cost:,.2f}  served={conv.served:.1f}  bal_err={conv.balance_error:.2e}")
    print(f"Global ref  : cost=${glob.total_cost:,.2f}  served={glob.served:.1f}")
    gap = (conv.total_cost - glob.total_cost) / glob.total_cost * 100
    print(f"Convex approximation gap vs global optimum: {gap:.3f}%")
