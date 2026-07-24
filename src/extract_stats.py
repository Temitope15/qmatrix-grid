"""Network 03 stats extractor for QCi Phase 2 (QMatrix team).

Parses case.raw (PSS/E v33), case.rop (generator cost curves), and case.con
(N-1 contingencies) for `Network_03O-10/scenario_1`, runs a sanity AC power
flow with pandapower, and writes:

  doc/stats.json
  doc/figures/network03_topology.png
  doc/figures/cost_curve_example.png
  doc/figures/contingency_breakdown.png

All numbers cited in the Phase 2 submission come from stats.json.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10"
SCEN = DATA / "scenario_1"
OUT_JSON = ROOT / "doc" / "stats.json"
FIG_DIR = ROOT / "doc" / "figures"


# ---------------------------------------------------------------------------
# PSS/E v33 RAW parser (only the sections we need for the doc)
# ---------------------------------------------------------------------------

@dataclass
class RawNetwork:
    sbase: float
    buses: list[dict] = field(default_factory=list)
    loads: list[dict] = field(default_factory=list)
    fixed_shunts: list[dict] = field(default_factory=list)
    generators: list[dict] = field(default_factory=list)
    branches: list[dict] = field(default_factory=list)
    transformers: list[dict] = field(default_factory=list)
    areas: list[dict] = field(default_factory=list)
    switched_shunts: list[dict] = field(default_factory=list)


def _split(line: str) -> list[str]:
    """Split a PSS/E CSV row, stripping quoted strings."""
    out, buf, in_quote = [], [], False
    for ch in line:
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif ch == "," and not in_quote:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return out


def _is_end(line: str) -> bool:
    return line.lstrip().startswith("0 ") or line.lstrip().startswith("0,") or line.strip() == "0"


def parse_raw(path: Path) -> RawNetwork:
    lines = path.read_text().splitlines()
    # Header: 3 lines (case ID, comment, date)
    header = _split(lines[0])
    sbase = float(header[1])
    net = RawNetwork(sbase=sbase)
    i = 3

    def read_section() -> list[list[str]]:
        nonlocal i
        rows = []
        while i < len(lines) and not _is_end(lines[i]):
            rows.append(_split(lines[i]))
            i += 1
        i += 1  # skip the "0 / end ..." delimiter
        return rows

    # Bus: I, NAME, BASKV, IDE, AREA, ZONE, OWNER, VM, VA, NVHI, NVLO, EVHI, EVLO
    for r in read_section():
        net.buses.append({
            "bus": int(r[0]), "name": r[1].strip("'").strip(),
            "base_kv": float(r[2]), "ide": int(r[3]),
            "area": int(r[4]), "zone": int(r[5]), "owner": int(r[6]),
            "vm": float(r[7]), "va": float(r[8]),
            "nvhi": float(r[9]), "nvlo": float(r[10]),
            "evhi": float(r[11]), "evlo": float(r[12]),
        })
    # Load: I, ID, STATUS, AREA, ZONE, PL, QL, IP, IQ, YP, YQ, OWNER, SCALE, INTRPT
    for r in read_section():
        net.loads.append({
            "bus": int(r[0]), "id": r[1].strip("'").strip(),
            "status": int(r[2]), "p_mw": float(r[5]), "q_mvar": float(r[6]),
        })
    # Fixed Shunt: I, ID, STATUS, GL, BL
    for r in read_section():
        net.fixed_shunts.append({
            "bus": int(r[0]), "id": r[1].strip("'").strip(),
            "status": int(r[2]), "gl": float(r[3]), "bl": float(r[4]),
        })
    # Generator: I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT, RMPCT, PT, PB, ...
    for r in read_section():
        net.generators.append({
            "bus": int(r[0]), "id": r[1].strip("'").strip(),
            "p_mw": float(r[2]), "q_mvar": float(r[3]),
            "qt": float(r[4]), "qb": float(r[5]),
            "vs": float(r[6]), "stat": int(r[14]),
            "pt": float(r[16]), "pb": float(r[17]),
        })
    # Non-transformer branch: I, J, CKT, R, X, B, RATEA, RATEB, RATEC, ..., ST, ...
    for r in read_section():
        net.branches.append({
            "from": int(r[0]), "to": int(r[1]), "ckt": r[2].strip("'").strip(),
            "r": float(r[3]), "x": float(r[4]), "b": float(r[5]),
            "rate_a": float(r[6]), "rate_b": float(r[7]), "rate_c": float(r[8]),
            "status": int(r[13]),
        })
    # Transformer (2-winding): 4 lines per record
    while i < len(lines) and not _is_end(lines[i]):
        l1 = _split(lines[i]); l2 = _split(lines[i + 1])
        l3 = _split(lines[i + 2]); l4 = _split(lines[i + 3])
        net.transformers.append({
            "from": int(l1[0]), "to": int(l1[1]),
            "ckt": l1[3].strip("'").strip(), "status": int(l1[11]),
            "r": float(l2[0]), "x": float(l2[1]), "sbase": float(l2[2]),
            "rate_a": float(l3[3]), "rate_b": float(l3[4]), "rate_c": float(l3[5]),
            "windv1": float(l3[0]), "windv2": float(l4[0]),
        })
        i += 4
    i += 1  # skip "0 / end transformer"
    # Area: I, ISW, PDES, PTOL, ARNAME
    for r in read_section():
        net.areas.append({"area": int(r[0]), "isw": int(r[1]),
                          "pdes": float(r[2]), "ptol": float(r[3])})
    # Skip 9 empty/unused sections to reach switched shunt: two-terminal DC,
    # VSC DC, transformer impedance, multi-terminal DC, multi-section line,
    # zone, interarea, owner, FACTS.
    for _ in range(9):
        read_section()
    # Switched shunt: I, MODSW, ADJM, STAT, VSWHI, VSWLO, ...
    for r in read_section():
        net.switched_shunts.append({
            "bus": int(r[0]), "modsw": int(r[1]), "stat": int(r[3]),
            "vswhi": float(r[4]), "vswlo": float(r[5]),
        })
    return net


# ---------------------------------------------------------------------------
# ROP (generator cost) parser
# ---------------------------------------------------------------------------

def parse_rop(path: Path) -> dict:
    lines = path.read_text().splitlines()
    i = 0
    # Skip the 5 leading "0 / END ... BEGIN ..." sections until "Generator Dispatch"
    while i < len(lines) and "Generator Dispatch" not in lines[i]:
        i += 1
    i += 1  # past the BEGIN line

    # Generator Dispatch: bus, gen_id, dspf, disptbl
    dispatch = []
    while i < len(lines) and not _is_end(lines[i]):
        r = _split(lines[i])
        dispatch.append({"bus": int(r[0]), "id": r[1].strip("'").strip(),
                         "dspf": float(r[2]), "disptbl": int(r[3])})
        i += 1
    i += 1

    # Active Power Dispatch Tables: tbl_id, pmax, pmin, fuelcost, costmodel, ..., ctbl
    dispatch_tables = {}
    while i < len(lines) and not _is_end(lines[i]):
        r = _split(lines[i])
        dispatch_tables[int(r[0])] = {
            "pmax": float(r[1]), "pmin": float(r[2]),
            "fuel_cost": float(r[3]), "cost_model": int(r[4]),
            "ctbl": int(r[6]),
        }
        i += 1
    i += 1

    # Skip 4 sections (linear cost, quadratic cost, polynomial cost) to piecewise linear
    while i < len(lines) and "Piece-wise Linear" not in lines[i]:
        i += 1
    i += 1

    # Piecewise linear cost tables
    pwl = {}
    while i < len(lines) and not _is_end(lines[i]):
        head = _split(lines[i])
        tbl_id = int(head[0])
        n_pts = int(head[2])
        pts = []
        for k in range(1, n_pts + 1):
            r = _split(lines[i + k])
            pts.append((float(r[0]), float(r[1])))
        pwl[tbl_id] = pts
        i += n_pts + 1
    return {"dispatch": dispatch, "dispatch_tables": dispatch_tables, "pwl": pwl}


# ---------------------------------------------------------------------------
# CON (contingency) parser
# ---------------------------------------------------------------------------

def parse_con(path: Path) -> list[dict]:
    out = []
    for stanza in path.read_text().split("CONTINGENCY"):
        stanza = stanza.strip()
        if not stanza:
            continue
        first_line, *rest = stanza.splitlines()
        name = first_line.strip()
        for ln in rest:
            ln = ln.strip()
            if ln.startswith("OPEN BRANCH"):
                # OPEN BRANCH FROM BUS i TO BUS j CIRCUIT c
                toks = ln.split()
                out.append({"name": name, "type": "branch_out",
                            "from": int(toks[4]), "to": int(toks[7]),
                            "ckt": toks[9].strip("'")})
            elif ln.startswith("REMOVE UNIT"):
                toks = ln.split()
                out.append({"name": name, "type": "gen_out",
                            "unit_id": toks[2].strip("'"),
                            "bus": int(toks[5])})
            elif ln.startswith("END"):
                break
    return out


# ---------------------------------------------------------------------------
# Build pandapower net for AC power flow validation
# ---------------------------------------------------------------------------

def build_pandapower(raw: RawNetwork) -> pp.pandapowerNet:
    net = pp.create_empty_network(sn_mva=raw.sbase, f_hz=60.0)
    bus_lookup = {}
    for b in raw.buses:
        idx = pp.create_bus(net, vn_kv=b["base_kv"], name=str(b["bus"]),
                            index=b["bus"], in_service=(b["ide"] != 4),
                            max_vm_pu=b["nvhi"], min_vm_pu=b["nvlo"],
                            zone=b["zone"])
        bus_lookup[b["bus"]] = idx
    # External grid: bus where ide == 3 (slack)
    slack = next((b for b in raw.buses if b["ide"] == 3), None)
    if slack is not None:
        pp.create_ext_grid(net, bus=slack["bus"], vm_pu=slack["vm"], name="slack")
    for l in raw.loads:
        if l["status"] == 1:
            pp.create_load(net, bus=l["bus"], p_mw=l["p_mw"],
                           q_mvar=l["q_mvar"], name=f"L{l['bus']}_{l['id']}")
    for g in raw.generators:
        if g["stat"] == 1 and not (slack and g["bus"] == slack["bus"]):
            pp.create_gen(net, bus=g["bus"], p_mw=g["p_mw"], vm_pu=g["vs"],
                          min_p_mw=g["pb"], max_p_mw=g["pt"],
                          min_q_mvar=g["qb"], max_q_mvar=g["qt"],
                          name=f"G{g['bus']}_{g['id']}", slack=False)
    for ln in raw.branches:
        if ln["status"] == 1 and ln["from"] in bus_lookup and ln["to"] in bus_lookup:
            f_bus = next(b for b in raw.buses if b["bus"] == ln["from"])
            base_kv = f_bus["base_kv"]
            z_base = (base_kv ** 2) / raw.sbase  # ohms per p.u.
            r_ohm = ln["r"] * z_base
            x_ohm = ln["x"] * z_base
            c_nf = ln["b"] / (2 * np.pi * 60 * z_base) * 1e9
            try:
                pp.create_line_from_parameters(
                    net, from_bus=ln["from"], to_bus=ln["to"], length_km=1.0,
                    r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
                    c_nf_per_km=max(c_nf, 0), max_i_ka=ln["rate_a"] / (np.sqrt(3) * base_kv) if base_kv > 0 else 1.0,
                    name=f"L{ln['from']}-{ln['to']}-{ln['ckt']}")
            except Exception:
                pass
    return net


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_topology(raw: RawNetwork, out: Path) -> None:
    G = nx.MultiGraph()
    for b in raw.buses:
        G.add_node(b["bus"], base_kv=b["base_kv"])
    for ln in raw.branches:
        if ln["status"] == 1:
            G.add_edge(ln["from"], ln["to"], weight=ln["rate_a"])
    for t in raw.transformers:
        if t["status"] == 1:
            G.add_edge(t["from"], t["to"], weight=t["rate_a"])
    kvs = sorted({b["base_kv"] for b in raw.buses})
    cmap = plt.colormaps["viridis"].resampled(max(len(kvs), 1))
    color = {kv: cmap(i) for i, kv in enumerate(kvs)}
    np.random.seed(7)
    pos = nx.spring_layout(G, seed=7, iterations=80, k=0.15)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    node_colors = [color[G.nodes[n]["base_kv"]] for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.4, edge_color="#666")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=8, linewidths=0)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          color=color[kv], label=f"{kv:g} kV")
               for kv in kvs]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False,
              title="Bus base voltage", title_fontsize=7)
    ax.set_axis_off()
    ax.set_title(f"GO Competition Network 03 — {G.number_of_nodes()} buses, "
                 f"{G.number_of_edges()} edges (branches + transformers)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cost_curve(rop: dict, out: Path) -> dict:
    # Pick the PWL curve with the largest deviation from a straight line — that's
    # where a cubic actually buys something over the trivial linear fit.
    def curvature(pts):
        p = np.array([x for x, _ in pts])
        c = np.array([y for _, y in pts])
        if c.max() < 1.0:
            return -1.0
        a, b = np.polyfit(p, c, 1)
        return float(np.max(np.abs(a * p + b - c)) / c.max())

    candidates = [(tid, pts) for tid, pts in rop["pwl"].items() if len(pts) >= 4]
    candidates.sort(key=lambda kv: curvature(kv[1]), reverse=True)
    tbl_id, pts = candidates[0]
    p = np.array([pt[0] for pt in pts])
    c = np.array([pt[1] for pt in pts])
    cubic = np.polyfit(p, c, 3)
    linear = np.polyfit(p, c, 1)
    p_dense = np.linspace(p.min(), p.max(), 200)
    c_cubic = np.polyval(cubic, p_dense)
    c_linear = np.polyval(linear, p_dense)
    rel_residual_cubic = float(np.max(np.abs(np.polyval(cubic, p) - c)) / max(c.max(), 1e-9))
    rel_residual_linear = float(np.max(np.abs(np.polyval(linear, p) - c)) / max(c.max(), 1e-9))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(p, c, "o", color="C0", ms=6, label="Piecewise-linear (case.rop)", zorder=3)
    ax.plot(p_dense, c_linear, "--", color="gray", lw=1.0,
            label=f"Linear fit (max error {rel_residual_linear:.1%})")
    ax.plot(p_dense, c_cubic, "-", color="C3", lw=1.4,
            label=f"Cubic fit  $C(P)=aP^3+bP^2+cP+d$ (max error {rel_residual_cubic:.2%})")
    ax.set_xlabel("Generator active power $P_G$ (MW)")
    ax.set_ylabel("Total fuel cost (\\$/h)")
    ax.set_title(f"Cost curve, dispatch table {tbl_id} — cubic captures "
                 "the non-linear heat-rate of a thermal unit", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"table_id": tbl_id, "n_points": len(pts),
            "cubic_coeffs": cubic.tolist(),
            "max_relative_residual_cubic": rel_residual_cubic,
            "max_relative_residual_linear": rel_residual_linear,
            "p_max_mw": float(p.max()), "cost_at_pmax": float(c.max())}


def fig_contingencies(cons: list[dict], out: Path) -> None:
    counts = Counter(c["type"] for c in cons)
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    ax.bar(labels, values, color=["#1f77b4", "#d62728"])
    ax.set_ylabel("Number of N-1 contingencies")
    ax.set_title(f"case.con: {sum(values)} N-1 contingencies", fontsize=10)
    for k, v in zip(labels, values):
        ax.text(k, v, f" {v}", va="bottom", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    raw = parse_raw(SCEN / "case.raw")
    rop = parse_rop(DATA / "case.rop")
    con = parse_con(SCEN / "case.con")

    # Sanity: counts must match readme.txt
    expected = {"buses": 793, "loads": 568, "generators": 210,
                "branches": 769, "transformers": 143, "switched_shunts": 50}
    actual = {"buses": len(raw.buses), "loads": len(raw.loads),
              "generators": len(raw.generators), "branches": len(raw.branches),
              "transformers": len(raw.transformers),
              "switched_shunts": len(raw.switched_shunts)}
    mismatches = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    if mismatches:
        raise SystemExit(f"Count mismatch vs readme.txt: {mismatches}")

    base_kvs = Counter(b["base_kv"] for b in raw.buses)
    branch_rates = np.array([b["rate_a"] for b in raw.branches if b["status"] == 1])
    gen_pmax = np.array([g["pt"] for g in raw.generators if g["stat"] == 1])

    # Cost curves: flat vs non-flat
    flat_tables = [tid for tid, pts in rop["pwl"].items()
                   if len(pts) <= 2 and max(c for _, c in pts) < 1e-3]
    flat_gens = sum(1 for d in rop["dispatch"]
                    if rop["dispatch_tables"].get(d["disptbl"], {}).get("ctbl") in flat_tables)

    # AC power flow on base case
    pp_net = build_pandapower(raw)
    pf_converged = False
    pf_msg = ""
    try:
        pp.runpp(pp_net, algorithm="nr", numba=False)
        pf_converged = pp_net.converged
        pf_msg = "converged" if pf_converged else "did not converge"
    except Exception as exc:
        pf_msg = f"runpp raised {type(exc).__name__}: {exc}"

    cost_info = fig_cost_curve(rop, FIG_DIR / "cost_curve_example.png")
    fig_topology(raw, FIG_DIR / "network03_topology.png")
    fig_contingencies(con, FIG_DIR / "contingency_breakdown.png")

    stats = {
        "dataset": {
            "network": "Network_03O-10",
            "scenario": "scenario_1",
            "source": "ARPA-E GO Competition Challenge 1",
            "sbase_mva": raw.sbase,
        },
        "counts": actual,
        "demand": {
            "total_active_load_mw": float(sum(l["p_mw"] for l in raw.loads if l["status"] == 1)),
            "total_reactive_load_mvar": float(sum(l["q_mvar"] for l in raw.loads if l["status"] == 1)),
            "n_active_loads": sum(1 for l in raw.loads if l["status"] == 1),
        },
        "generation": {
            "total_pmax_mw": float(gen_pmax.sum()),
            "pmax_median_mw": float(np.median(gen_pmax)),
            "pmax_p10_mw": float(np.percentile(gen_pmax, 10)),
            "pmax_p90_mw": float(np.percentile(gen_pmax, 90)),
            "n_committed_in_basecase": int(sum(1 for g in raw.generators if g["stat"] == 1 and g["p_mw"] > 0)),
            "n_zero_cost_generators": flat_gens,
        },
        "topology": {
            "base_kv_levels": sorted(base_kvs.keys()),
            "buses_per_kv": {f"{k:g}": v for k, v in sorted(base_kvs.items())},
            "n_areas": len(raw.areas),
            "n_active_branches": int(sum(1 for b in raw.branches if b["status"] == 1)),
            "branch_rate_a_min_mva": float(branch_rates.min()),
            "branch_rate_a_median_mva": float(np.median(branch_rates)),
            "branch_rate_a_max_mva": float(branch_rates.max()),
        },
        "voltage_limits": {
            "v_min_pu_min": float(min(b["nvlo"] for b in raw.buses)),
            "v_min_pu_max": float(max(b["nvlo"] for b in raw.buses)),
            "v_max_pu_min": float(min(b["nvhi"] for b in raw.buses)),
            "v_max_pu_max": float(max(b["nvhi"] for b in raw.buses)),
        },
        "cost_curves": {
            "n_piecewise_tables": len(rop["pwl"]),
            "n_flat_tables_zero_cost": len(flat_tables),
            "example_cubic_fit": cost_info,
        },
        "contingencies": {
            "total": len(con),
            "branch_out": sum(1 for c in con if c["type"] == "branch_out"),
            "gen_out": sum(1 for c in con if c["type"] == "gen_out"),
        },
        "power_flow_validation": {
            "tool": f"pandapower {pp.__version__}",
            "algorithm": "Newton-Raphson",
            "converged": bool(pf_converged),
            "note": pf_msg,
        },
    }

    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"[ok] wrote {OUT_JSON.relative_to(ROOT)}")
    for name in ["network03_topology.png", "cost_curve_example.png", "contingency_breakdown.png"]:
        print(f"[ok] wrote doc/figures/{name}")
    print(f"[info] AC power flow: {pf_msg}")


if __name__ == "__main__":
    main()
