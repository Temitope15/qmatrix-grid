"""Non-convex economic dispatch benchmark: Dirac-3 EQC vs classical baselines.

Runs the QMatrix Phase 3 quantum centerpiece and writes reproducible results to
doc/stats_dispatch.json. Two experiments on Network_03O-10's largest microgrid:

  Experiment A (CONTROL): dataset-faithful cubic cost (near-convex). Confirms
      Dirac-3 matches the classical optimum when the problem is convex.
  Experiment B (ADVANTAGE): literature-calibrated non-convex cubic cost, swept
      across all 10 operational scenarios at each scenario's real cluster demand.
      Shows Dirac-3 consistently beating the convex LP/QP baseline.

Usage:  python experiments/dispatch_benchmark.py [--quick]
        --quick  : scenario 1 only (fast smoke test)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grid_graph import GridGraphModel
from models.cost_models import (
    dataset_cubic_generators,
    literature_nonconvex_generators,
    characterize_convexity,
)
from models.dispatch_dirac3 import NonConvexDispatch

DATA = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10"
N_DISPATCH_UNITS = 10          # largest K units of the target cluster
LOAD_FRACTION = 0.60           # fallback demand if a scenario load can't be read
RELAX_SCHEDULE = 3
NUM_SAMPLES = 20


def scenario_cluster_demand(scenario: int, cluster_buses: list[int], fallback: float) -> float:
    """Real cluster demand for a scenario, read from that scenario's case.raw."""
    raw_f = DATA / f"scenario_{scenario}" / "case.raw"
    if not raw_f.exists():
        return fallback
    try:
        g = GridGraphModel(raw_f)
        return float(sum(g.graph.nodes[b].get("p_load", 0.0)
                         for b in cluster_buses if b in g.graph))
    except Exception:
        return fallback


def run_triplet(disp: NonConvexDispatch, demand: float, label: str) -> dict:
    """Convex QP + Dirac-3 + global reference on one instance."""
    conv = disp.solve_convex_qp(demand)
    glob = disp.solve_global_reference(demand, n_starts=80)
    dirac = disp.solve_dirac3(demand, relaxation_schedule=RELAX_SCHEDULE,
                              num_samples=NUM_SAMPLES, job_name=f"QMatrix_{label}")
    g = glob.total_cost

    def gap(x):
        return round((x - g) / g * 100, 3) if g else 0.0

    conv_save = conv.total_cost - dirac.total_cost
    return {
        "label": label,
        "demand_mw": round(demand, 2),
        "convex_qp_cost": round(conv.total_cost, 2),
        "dirac3_cost": round(dirac.total_cost, 2),
        "global_cost": round(glob.total_cost, 2),
        "convex_qp_gap_pct": gap(conv.total_cost),
        "dirac3_gap_pct": gap(dirac.total_cost),
        "dirac3_saving_vs_convex_usd_h": round(conv_save, 2),
        "dirac3_saving_vs_convex_pct": round(conv_save / conv.total_cost * 100, 3) if conv.total_cost else 0.0,
        "dirac3_balance_error_mw": round(dirac.balance_error, 4),
        "dirac3_feasible": bool(dirac.feasible),
        "dirac3_runtime_sec": dirac.runtime_sec,
        "global_runtime_sec": glob.runtime_sec,
        "device_energy": dirac.extra.get("device_energy"),
    }


def main(quick: bool = False) -> dict:
    print("=" * 74)
    print("   QMatrix Phase 3 — Non-Convex Dispatch: Dirac-3 EQC vs Classical")
    print("=" * 74)

    grid = GridGraphModel(DATA / "scenario_1" / "case.raw", DATA / "case.rop",
                          DATA / "scenario_1" / "case.con")
    clusters = grid.identify_microgrids_spectral(n_clusters=5)
    cid = max(clusters, key=lambda k: len(clusters[k]))
    buses = clusters[cid]
    print(f"Target microgrid: cluster {cid} ({len(buses)} buses)")

    # ---- Experiment A: dataset-faithful cubic (control) --------------------
    gensA = dataset_cubic_generators(grid, buses)
    gensA.sort(key=lambda x: x.p_max, reverse=True)
    gensA = gensA[:N_DISPATCH_UNITS]
    dispA = NonConvexDispatch(gensA)
    nA = sum(1 for g in gensA if characterize_convexity(g)["is_nonconvex"])
    demandA = LOAD_FRACTION * dispA.p_max.sum()
    print(f"\n[A] Dataset cubic control: {len(gensA)} units, {nA} non-convex, demand {demandA:.0f} MW")
    resA = run_triplet(dispA, demandA, "A_dataset_control")
    print(f"    convexQP=${resA['convex_qp_cost']:,} dirac3=${resA['dirac3_cost']:,} "
          f"global=${resA['global_cost']:,} | dirac gap {resA['dirac3_gap_pct']}%")

    # ---- Experiment B: non-convex cubic, swept over scenarios --------------
    gensB = literature_nonconvex_generators(grid, buses, ripple=0.9)
    gensB.sort(key=lambda x: x.p_max, reverse=True)
    gensB = gensB[:N_DISPATCH_UNITS]
    dispB = NonConvexDispatch(gensB)
    nB = sum(1 for g in gensB if characterize_convexity(g)["is_nonconvex"])
    convexity = [characterize_convexity(g) for g in gensB]
    fallbackB = LOAD_FRACTION * dispB.p_max.sum()
    band = (float(dispB.p_min.sum()), float(dispB.p_max.sum()))
    print(f"\n[B] Non-convex cubic: {len(gensB)} units, {nB} non-convex, "
          f"feasible band [{band[0]:.0f}, {band[1]:.0f}] MW")

    scenarios = [1] if quick else list(range(1, 11))
    sweep = []
    for sc in scenarios:
        dem = scenario_cluster_demand(sc, buses, fallbackB)
        dem = float(np.clip(dem, band[0] + 1, band[1] - 1))
        r = run_triplet(dispB, dem, f"B_scenario_{sc}")
        r["scenario"] = sc
        sweep.append(r)
        print(f"    scenario {sc:2d}: demand {r['demand_mw']:7.0f} MW | "
              f"convexQP ${r['convex_qp_cost']:>10,.0f} | dirac3 ${r['dirac3_cost']:>10,.0f} "
              f"(gap {r['dirac3_gap_pct']:+.2f}%) | saves {r['dirac3_saving_vs_convex_pct']:+.2f}%")

    mean_save = round(float(np.mean([s["dirac3_saving_vs_convex_pct"] for s in sweep])), 3)
    mean_gap = round(float(np.mean([s["dirac3_gap_pct"] for s in sweep])), 3)
    mean_convex_gap = round(float(np.mean([s["convex_qp_gap_pct"] for s in sweep])), 3)
    max_balerr = round(float(max(s["dirac3_balance_error_mw"] for s in sweep)), 4)

    out = {
        "dataset": "ARPA-E GO Challenge 1 (Network_03O-10)",
        "target_cluster": int(cid),
        "target_cluster_buses": len(buses),
        "n_dispatch_units": N_DISPATCH_UNITS,
        "relaxation_schedule": RELAX_SCHEDULE,
        "num_samples": NUM_SAMPLES,
        "experiment_A_dataset_control": resA,
        "experiment_A_nonconvex_units": nA,
        "experiment_B_feasible_band_mw": band,
        "experiment_B_nonconvex_units": nB,
        "experiment_B_convexity": convexity,
        "experiment_B_sweep": sweep,
        "summary": {
            "dirac3_mean_saving_vs_convex_pct": mean_save,
            "dirac3_mean_gap_vs_global_pct": mean_gap,
            "convex_qp_mean_gap_vs_global_pct": mean_convex_gap,
            "dirac3_max_balance_error_mw": max_balerr,
            "scenarios_evaluated": len(sweep),
            "dirac3_all_feasible": all(s["dirac3_feasible"] for s in sweep),
        },
    }
    out_file = ROOT / "doc" / "stats_dispatch.json"
    out_file.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 74)
    print(f"  MEAN: Dirac-3 saves {mean_save}% vs convex QP, "
          f"gap to global {mean_gap}% (convex QP gap {mean_convex_gap}%)")
    print(f"  Max balance error {max_balerr} MW | all feasible: {out['summary']['dirac3_all_feasible']}")
    print(f"  Saved -> {out_file.relative_to(ROOT)}")
    print("=" * 74)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="scenario 1 only")
    args = ap.parse_args()
    main(quick=args.quick)
