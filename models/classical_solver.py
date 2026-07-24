"""Classical Power Flow & ACOPF Baseline Solver for QCi Challenge Phase 3.

Provides classical benchmarks using Pandapower & SciPy for:
- Base case AC/DC power flow with real cost curves.
- Linear Programming (LP) based DC-OPF for exact load shedding and cost-optimal dispatch.
- N-1 Contingency evaluation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.optimize as opt
import pandapower as pp
import pandapower.topology as top
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grid_graph import GridGraphModel


class ClassicalMicrogridSolver:
    """Classical benchmark solver for power grid optimization & contingency handling."""

    def __init__(self, grid_model: GridGraphModel):
        self.grid_model = grid_model
        # Build real cost mapping: (bus, gen_id) -> cost info
        self.gen_cost_map = grid_model.build_generator_cost_map()
        self.pp_net = self._build_pandapower_net()
        # Map pandapower gen index -> (bus, gen_id) for cost lookup
        self._pp_gen_to_key: Dict[int, Tuple[int, str]] = {}
        self._build_gen_index_map()

    def _build_pandapower_net(self) -> pp.pandapowerNet:
        """Construct Pandapower network from raw parsed data with correct ohms scaling and transformers."""
        sbase = self.grid_model.raw_net.sbase
        net = pp.create_empty_network(sn_mva=sbase)
        
        bus_lookup = {}
        for b in self.grid_model.raw_net.buses:
            b_idx = pp.create_bus(net, vn_kv=b["base_kv"], name=b["name"], zone=b["zone"], index=b["bus"])
            bus_lookup[b["bus"]] = b_idx
            
        for ld in self.grid_model.raw_net.loads:
            if ld["status"] == 1 and ld["bus"] in bus_lookup:
                pp.create_load(net, bus=bus_lookup[ld["bus"]], p_mw=ld["p_mw"], q_mvar=ld["q_mvar"])
                
        # Set slack bus (external grid reference) on swing bus
        slack = next((b for b in self.grid_model.raw_net.buses if b["ide"] == 3), None)
        if slack is not None:
            pp.create_ext_grid(net, bus=bus_lookup[slack["bus"]], vm_pu=slack["vm"], va_degree=0.0)
        else:
            first_bus = list(bus_lookup.values())[0]
            pp.create_ext_grid(net, bus=first_bus, vm_pu=1.0, va_degree=0.0)

        for g in self.grid_model.raw_net.generators:
            if g["stat"] == 1 and g["bus"] in bus_lookup:
                pp.create_gen(
                    net,
                    bus=bus_lookup[g["bus"]],
                    p_mw=g["p_mw"],
                    vm_pu=g["vs"],
                    max_p_mw=g["pt"],
                    min_p_mw=g["pb"],
                    max_q_mvar=g["qt"],
                    min_q_mvar=g["qb"],
                    name=f"G{g['bus']}_{g['id']}"
                )
                
        # Add transmission lines with correct Z_base scaling to ohms
        for br in self.grid_model.raw_net.branches:
            if br["status"] == 1 and br["from"] in bus_lookup and br["to"] in bus_lookup:
                vn_kv = net.bus.at[bus_lookup[br["from"]], "vn_kv"]
                z_base = (vn_kv ** 2) / sbase
                r_ohm = br["r"] * z_base
                x_ohm = br["x"] * z_base
                max_i = br["rate_a"] / (np.sqrt(3) * vn_kv) if vn_kv > 0 else 1.0
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_lookup[br["from"]],
                    to_bus=bus_lookup[br["to"]],
                    length_km=1.0,
                    r_ohm_per_km=r_ohm,
                    x_ohm_per_km=x_ohm,
                    c_nf_per_km=max(0.0, br["b"] / z_base * 1e9 / (2 * np.pi * 60)),
                    max_i_ka=max(0.1, max_i),
                    name=f"L{br['from']}-{br['to']}-{br['ckt']}"
                )

        # Add transformers as lines with correct Z_base scaling to ohms
        for xf in self.grid_model.raw_net.transformers:
            if xf["status"] == 1 and xf["from"] in bus_lookup and xf["to"] in bus_lookup:
                vn_kv = net.bus.at[bus_lookup[xf["from"]], "vn_kv"]
                z_base = (vn_kv ** 2) / sbase
                r_ohm = xf["r"] * z_base
                x_ohm = xf["x"] * z_base
                max_i = xf["rate_a"] / (np.sqrt(3) * vn_kv) if vn_kv > 0 else 1.0
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_lookup[xf["from"]],
                    to_bus=bus_lookup[xf["to"]],
                    length_km=1.0,
                    r_ohm_per_km=r_ohm,
                    x_ohm_per_km=x_ohm,
                    c_nf_per_km=0.0,
                    max_i_ka=max(0.1, max_i),
                    name=f"T{xf['from']}-{xf['to']}-{xf['ckt']}"
                )
                
        return net

    def _build_gen_index_map(self) -> None:
        """Map pandapower generator indices to (bus, gen_id) tuples for cost lookup."""
        pp_gen_idx = 0
        for g in self.grid_model.raw_net.generators:
            if g["stat"] == 1:
                self._pp_gen_to_key[pp_gen_idx] = (g["bus"], g["id"])
                pp_gen_idx += 1

    def _evaluate_cost_cubic(self, bus: int, gen_id: str, p_mw: float) -> float:
        """Evaluate generation cost for a single generator using real cubic cost curve."""
        cost_info = self.gen_cost_map.get((bus, gen_id))
        if cost_info is None:
            return max(0.0, 20.0 * p_mw)
        
        a, b, c, d = cost_info["cubic_coeffs"]
        p_clamped = max(cost_info["p_min"], min(abs(p_mw), cost_info["p_max"]))
        cost = a * p_clamped**3 + b * p_clamped**2 + c * p_clamped + d
        return max(0.0, cost)

    def compute_total_generation_cost(self, gen_powers: np.ndarray | None = None) -> float:
        """Compute total generation cost using real cubic cost curves from case.rop."""
        if gen_powers is None:
            if self.pp_net.res_gen.empty:
                return 0.0
            gen_powers = self.pp_net.res_gen.p_mw.values

        total_cost = 0.0
        for pp_idx, p_mw in enumerate(gen_powers):
            key = self._pp_gen_to_key.get(pp_idx)
            if key is not None:
                total_cost += self._evaluate_cost_cubic(key[0], key[1], p_mw)
        return total_cost

    def run_power_flow(self, mode: str = "dc") -> Dict[str, Any]:
        """Run AC or DC power flow and return metrics with real cost evaluation."""
        start_t = time.time()
        try:
            if mode == "dc":
                pp.rundcpp(self.pp_net, numba=False)
            else:
                pp.runpp(self.pp_net, algorithm="nr", max_iteration=50, numba=False)
                
            elapsed = time.time() - start_t
            
            total_load_mw = float(self.pp_net.load.p_mw.sum())
            total_gen_mw = float(self.pp_net.res_gen.p_mw.sum()) + (float(self.pp_net.res_ext_grid.p_mw.sum()) if not self.pp_net.res_ext_grid.empty else 0.0)
            total_loss_mw = max(0.0, total_gen_mw - total_load_mw)
            
            # REAL cost evaluation
            generation_cost = self.compute_total_generation_cost()
            
            return {
                "mode": mode,
                "converged": bool(self.pp_net.converged),
                "runtime_sec": round(elapsed, 4),
                "total_load_mw": round(total_load_mw, 2),
                "total_gen_mw": round(total_gen_mw, 2),
                "total_loss_mw": round(total_loss_mw, 2),
                "generation_cost_usd_h": round(generation_cost, 2),
                "unserved_load_mw": 0.0
            }
        except Exception as e:
            elapsed = time.time() - start_t
            return {
                "mode": mode,
                "converged": False,
                "runtime_sec": round(elapsed, 4),
                "error": str(e),
                "total_load_mw": float(self.pp_net.load.p_mw.sum()),
                "total_gen_mw": 0.0,
                "generation_cost_usd_h": 0.0,
                "unserved_load_mw": float(self.pp_net.load.p_mw.sum())
            }

    def solve_dc_opf(self, tripped_branch: dict | None = None, tripped_gen: dict | None = None) -> Dict[str, Any]:
        """Solve exact DC-OPF with load shedding using Linear Programming for optimal baseline.
        
        Args:
            tripped_branch: Optional contingency dict representing branch/transformer outage.
            tripped_gen: Optional contingency dict representing generator outage.
        """
        start_t = time.time()
        sbase = self.grid_model.raw_net.sbase
        buses = list(self.grid_model.graph.nodes())
        n_buses = len(buses)
        bus_to_idx = {b: i for i, b in enumerate(buses)}

        # 1. Filter active generators
        active_gens = []
        for (bus, gen_id), gen_data in self.grid_model.generators.items():
            if gen_data["stat"] == 1:
                if tripped_gen and tripped_gen["bus"] == bus and tripped_gen["unit_id"] == gen_id:
                    continue
                cost_info = self.gen_cost_map.get((bus, gen_id), {})
                coeffs = cost_info.get("cubic_coeffs", [0.0, 0.0, 20.0, 0.0])
                active_gens.append({
                    "bus": bus,
                    "gen_id": gen_id,
                    "p_min": gen_data["pb"],
                    "p_max": gen_data["pt"],
                    "c_linear": coeffs[2]
                })
        n_gens = len(active_gens)

        # 2. Filter active branches
        def is_branch_tripped(br):
            if not tripped_branch:
                return False
            f, t, c = tripped_branch["from"], tripped_branch["to"], tripped_branch["ckt"]
            return (br["from"] == f and br["to"] == t and br["ckt"] == c) or \
                   (br["from"] == t and br["to"] == f and br["ckt"] == c)

        active_branches = []
        for br in self.grid_model.raw_net.branches:
            if br["status"] == 1 and not is_branch_tripped(br):
                active_branches.append({
                    "from": br["from"],
                    "to": br["to"],
                    "x_pu": max(br["x"], 1e-4),
                    "rate_a": br["rate_a"] if br["rate_a"] > 0 else 9999.0
                })
                
        for tf in self.grid_model.raw_net.transformers:
            if tf["status"] == 1 and not is_branch_tripped(tf):
                active_branches.append({
                    "from": tf["from"],
                    "to": tf["to"],
                    "x_pu": max(tf["x"], 1e-4),
                    "rate_a": tf["rate_a"] if tf["rate_a"] > 0 else 9999.0
                })

        # 3. Setup LP Formulation
        # Variables: [P_g (n_gens), P_shed (n_buses), theta (n_buses)]
        c = np.zeros(n_gens + 2 * n_buses)
        for g_idx, g in enumerate(active_gens):
            c[g_idx] = g["c_linear"]
        for b_idx in range(n_buses):
            c[n_gens + b_idx] = 1e6  # Huge penalty for load shedding

        bounds = []
        for g in active_gens:
            bounds.append((g["p_min"], g["p_max"]))
        for b in buses:
            p_load = self.grid_model.graph.nodes[b].get("p_load", 0.0)
            bounds.append((0.0, max(0.0, p_load)))
        for b in buses:
            bounds.append((-np.pi, np.pi))

        # Equal constraints: power balance
        A_eq = []
        b_eq = []

        # Find components to pin one theta per component
        G_temp = nx.Graph()
        G_temp.add_nodes_from(buses)
        for br in active_branches:
            G_temp.add_edge(br["from"], br["to"])
        components = list(nx.connected_components(G_temp))
        pinned_buses = {list(comp)[0] for comp in components}

        for b_idx, b in enumerate(buses):
            row = np.zeros(n_gens + 2 * n_buses)
            for g_idx, g in enumerate(active_gens):
                if g["bus"] == b:
                    row[g_idx] = 1.0
            row[n_gens + b_idx] = 1.0
            
            for br in active_branches:
                if br["from"] == b:
                    j = br["to"]
                    x = br["x_pu"]
                    row[n_gens + n_buses + b_idx] -= sbase / x
                    row[n_gens + n_buses + bus_to_idx[j]] += sbase / x
                elif br["to"] == b:
                    j = br["from"]
                    x = br["x_pu"]
                    row[n_gens + n_buses + b_idx] -= sbase / x
                    row[n_gens + n_buses + bus_to_idx[j]] += sbase / x
            A_eq.append(row)
            b_eq.append(self.grid_model.graph.nodes[b].get("p_load", 0.0))

        for b in pinned_buses:
            row = np.zeros(n_gens + 2 * n_buses)
            row[n_gens + n_buses + bus_to_idx[b]] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)

        # Unequal constraints: branch limits
        A_ub = []
        b_ub = []
        for br in active_branches:
            if br["rate_a"] < 9999.0:
                x = br["x_pu"]
                limit = br["rate_a"]
                u_idx = bus_to_idx[br["from"]]
                v_idx = bus_to_idx[br["to"]]
                
                row1 = np.zeros(n_gens + 2 * n_buses)
                row1[n_gens + n_buses + u_idx] = sbase / x
                row1[n_gens + n_buses + v_idx] = -sbase / x
                A_ub.append(row1)
                b_ub.append(limit)
                
                row2 = np.zeros(n_gens + 2 * n_buses)
                row2[n_gens + n_buses + u_idx] = -sbase / x
                row2[n_gens + n_buses + v_idx] = sbase / x
                A_ub.append(row2)
                b_ub.append(limit)

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        elapsed = time.time() - start_t

        if res.success:
            P_g = res.x[:n_gens]
            P_shed = res.x[n_gens:n_gens+n_buses]
            total_shed = float(sum(P_shed))
            total_load = sum(self.grid_model.graph.nodes[b].get("p_load", 0.0) for b in buses)
            
            # Evaluate cost using exact cubic cost curves
            cost = 0.0
            for g_idx, g in enumerate(active_gens):
                cost += self._evaluate_cost_cubic(g["bus"], g["gen_id"], P_g[g_idx])

            return {
                "converged": True,
                "runtime_sec": round(elapsed, 4),
                "unserved_load_mw": round(total_shed, 2),
                "total_load_mw": round(total_load, 2),
                "generation_cost_usd_h": round(cost, 2),
                "unserved_pct": round(total_shed / total_load * 100.0, 2) if total_load > 0 else 0.0
            }
        else:
            total_load = sum(self.grid_model.graph.nodes[b].get("p_load", 0.0) for b in buses)
            return {
                "converged": False,
                "runtime_sec": round(elapsed, 4),
                "unserved_load_mw": round(total_load, 2),
                "total_load_mw": round(total_load, 2),
                "generation_cost_usd_h": 0.0,
                "unserved_pct": 100.0
            }

    def solve_dc_opf_with_tripped_pccs(
        self,
        tripped_pcc_edges: list[tuple[int, int]],
        extra_gen_mw: dict[int, float] | None = None,
        tripped_gen: dict | None = None,
    ) -> Dict[str, Any]:
        """Solve DC-OPF after disconnecting a set of PCC tie-lines.

        This models quantum-assisted islanding: the quantum optimizer decides
        which PCC lines to disconnect, then we evaluate the resulting
        multi-island network via LP DC-OPF to compute actual load shedding.

        Args:
            tripped_pcc_edges: List of (from_bus, to_bus) PCC edges to trip.
            extra_gen_mw: Optional dict mapping bus_id -> added DER capacity (MW)
                          for upgrade scenario evaluation.
            tripped_gen: Optional generator contingency to also apply.
        """
        start_t = time.time()
        sbase = self.grid_model.raw_net.sbase
        buses = list(self.grid_model.graph.nodes())
        n_buses = len(buses)
        bus_to_idx = {b: i for i, b in enumerate(buses)}

        tripped_set = set()
        for u, v in tripped_pcc_edges:
            tripped_set.add((min(u, v), max(u, v)))

        def _is_pcc_tripped(fr: int, to: int) -> bool:
            key = (min(fr, to), max(fr, to))
            return key in tripped_set

        # 1. Active generators
        active_gens = []
        for (bus, gen_id), gen_data in self.grid_model.generators.items():
            if gen_data["stat"] == 1:
                if tripped_gen and tripped_gen["bus"] == bus and tripped_gen["unit_id"] == gen_id:
                    continue
                cost_info = self.gen_cost_map.get((bus, gen_id), {})
                coeffs = cost_info.get("cubic_coeffs", [0.0, 0.0, 20.0, 0.0])
                active_gens.append({
                    "bus": bus, "gen_id": gen_id,
                    "p_min": gen_data["pb"],
                    "p_max": gen_data["pt"],
                    "c_linear": coeffs[2],
                })

        # Add DER upgrades as virtual generators
        if extra_gen_mw:
            for bus, cap_mw in extra_gen_mw.items():
                if cap_mw > 0 and bus in bus_to_idx:
                    active_gens.append({
                        "bus": bus, "gen_id": "__DER__",
                        "p_min": 0.0, "p_max": cap_mw,
                        "c_linear": 50.0,  # DER marginal cost
                    })
        n_gens = len(active_gens)

        # 2. Active branches (excluding tripped PCC lines)
        active_branches = []
        for br in self.grid_model.raw_net.branches:
            if br["status"] == 1 and not _is_pcc_tripped(br["from"], br["to"]):
                active_branches.append({
                    "from": br["from"], "to": br["to"],
                    "x_pu": max(br["x"], 1e-4),
                    "rate_a": br["rate_a"] if br["rate_a"] > 0 else 9999.0,
                })
        for tf in self.grid_model.raw_net.transformers:
            if tf["status"] == 1 and not _is_pcc_tripped(tf["from"], tf["to"]):
                active_branches.append({
                    "from": tf["from"], "to": tf["to"],
                    "x_pu": max(tf["x"], 1e-4),
                    "rate_a": tf["rate_a"] if tf["rate_a"] > 0 else 9999.0,
                })

        # 3. LP formulation (identical structure to solve_dc_opf)
        c_vec = np.zeros(n_gens + 2 * n_buses)
        for g_idx, g in enumerate(active_gens):
            c_vec[g_idx] = g["c_linear"]
        for b_idx in range(n_buses):
            c_vec[n_gens + b_idx] = 1e6  # load-shedding penalty

        bounds = []
        for g in active_gens:
            bounds.append((g["p_min"], g["p_max"]))
        for b in buses:
            p_load = self.grid_model.graph.nodes[b].get("p_load", 0.0)
            bounds.append((0.0, max(0.0, p_load)))
        for _ in buses:
            bounds.append((-np.pi, np.pi))

        # Equality: power balance + reference angles
        G_temp = nx.Graph()
        G_temp.add_nodes_from(buses)
        for br in active_branches:
            G_temp.add_edge(br["from"], br["to"])
        components = list(nx.connected_components(G_temp))
        pinned_buses = {list(comp)[0] for comp in components}

        A_eq, b_eq = [], []
        for b_idx, b in enumerate(buses):
            row = np.zeros(n_gens + 2 * n_buses)
            for g_idx, g in enumerate(active_gens):
                if g["bus"] == b:
                    row[g_idx] = 1.0
            row[n_gens + b_idx] = 1.0
            for br in active_branches:
                if br["from"] == b:
                    j = br["to"]
                    row[n_gens + n_buses + b_idx] -= sbase / br["x_pu"]
                    row[n_gens + n_buses + bus_to_idx[j]] += sbase / br["x_pu"]
                elif br["to"] == b:
                    j = br["from"]
                    row[n_gens + n_buses + b_idx] -= sbase / br["x_pu"]
                    row[n_gens + n_buses + bus_to_idx[j]] += sbase / br["x_pu"]
            A_eq.append(row)
            b_eq.append(self.grid_model.graph.nodes[b].get("p_load", 0.0))

        for b in pinned_buses:
            row = np.zeros(n_gens + 2 * n_buses)
            row[n_gens + n_buses + bus_to_idx[b]] = 1.0
            A_eq.append(row)
            b_eq.append(0.0)

        # Inequality: branch thermal limits
        A_ub, b_ub = [], []
        for br in active_branches:
            if br["rate_a"] < 9999.0:
                u_idx = bus_to_idx[br["from"]]
                v_idx = bus_to_idx[br["to"]]
                row1 = np.zeros(n_gens + 2 * n_buses)
                row1[n_gens + n_buses + u_idx] = sbase / br["x_pu"]
                row1[n_gens + n_buses + v_idx] = -sbase / br["x_pu"]
                A_ub.append(row1)
                b_ub.append(br["rate_a"])
                row2 = np.zeros(n_gens + 2 * n_buses)
                row2[n_gens + n_buses + u_idx] = -sbase / br["x_pu"]
                row2[n_gens + n_buses + v_idx] = sbase / br["x_pu"]
                A_ub.append(row2)
                b_ub.append(br["rate_a"])

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        res = opt.linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
        elapsed = time.time() - start_t

        total_load = sum(self.grid_model.graph.nodes[b].get("p_load", 0.0) for b in buses)
        if res.success:
            P_shed = res.x[n_gens:n_gens + n_buses]
            total_shed = float(sum(P_shed))
            P_g = res.x[:n_gens]
            cost = 0.0
            for g_idx, g in enumerate(active_gens):
                if g["gen_id"] != "__DER__":
                    cost += self._evaluate_cost_cubic(g["bus"], g["gen_id"], P_g[g_idx])
            return {
                "converged": True,
                "runtime_sec": round(elapsed, 4),
                "unserved_load_mw": round(total_shed, 2),
                "total_load_mw": round(total_load, 2),
                "generation_cost_usd_h": round(cost, 2),
                "unserved_pct": round(total_shed / total_load * 100, 2) if total_load > 0 else 0.0,
                "num_islands": len(components),
            }
        else:
            return {
                "converged": False,
                "runtime_sec": round(elapsed, 4),
                "unserved_load_mw": round(total_load, 2),
                "total_load_mw": round(total_load, 2),
                "generation_cost_usd_h": 0.0,
                "unserved_pct": 100.0,
                "num_islands": len(components),
            }

    def solve_classical_load_shedding_contingency(self, tripped_branch_idx: int = 0) -> Dict[str, Any]:
        """Backward compatible wrapper. Solves DC-OPF with tripped branch by index."""
        if 0 <= tripped_branch_idx < len(self.pp_net.line):
            line_name = self.pp_net.line.loc[tripped_branch_idx, "name"]
            # Parse line name back to from/to/ckt
            # Format: Lfrom-to-ckt or Tfrom-to-ckt
            parts = line_name[1:].split("-")
            tripped_branch = {
                "from": int(parts[0]),
                "to": int(parts[1]),
                "ckt": parts[2]
            }
            res = self.solve_dc_opf(tripped_branch=tripped_branch)
            res["contingency_branch_idx"] = tripped_branch_idx
            res["contingency_runtime_sec"] = res["runtime_sec"]
            return res
        return self.run_power_flow(mode="dc")


if __name__ == "__main__":
    print("Testing Updated ClassicalMicrogridSolver with LP DC-OPF...")
    raw_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "scenario_1" / "case.raw"
    rop_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "case.rop"
    con_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "scenario_1" / "case.con"
    
    grid = GridGraphModel(raw_f, rop_f, con_f)
    solver = ClassicalMicrogridSolver(grid)
    
    base_res = solver.solve_dc_opf()
    print("Base Case DC-OPF Result:")
    print(f"  Converged: {base_res['converged']}")
    print(f"  Runtime: {base_res['runtime_sec']}s")
    print(f"  Total Load: {base_res['total_load_mw']} MW")
    print(f"  Unserved Load: {base_res['unserved_load_mw']} MW ({base_res['unserved_pct']}%)")
    print(f"  Generation Cost: ${base_res['generation_cost_usd_h']:,.2f}/h")
