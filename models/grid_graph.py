"""Grid Graph & Microgrid Topology Modeling for QCi Challenge Phase 3.

Provides NetworkX graph representation of power transmission networks (ARPA-E GO Challenge datasets)
and spectral graph partition tools for identifying candidate microgrids and PCC tie-lines.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extract_stats import parse_raw, parse_rop, parse_con, RawNetwork


class GridGraphModel:
    """NetworkX Graph representation of power grid topology with microgrid candidate extraction."""

    # Loads at/above this level are treated as aggregated critical infrastructure.
    CRITICAL_LOAD_THRESHOLD_MW = 25.0

    def __init__(self, raw_path: Path, rop_path: Path | None = None, con_path: Path | None = None):
        self.raw_path = raw_path
        self.rop_path = rop_path
        self.con_path = con_path
        
        self.raw_net: RawNetwork = parse_raw(raw_path)
        self.rop_data: dict = parse_rop(rop_path) if rop_path and rop_path.exists() else {}
        self.con_data: dict = parse_con(con_path) if con_path and con_path.exists() else {}
        
        self.graph: nx.Graph = nx.Graph()
        self.buses: dict = {}
        self.generators: dict = {}
        self.loads: dict = {}
        self.branches: dict = {}
        
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct NetworkX graph with node and edge attributes."""
        # 1. Add Bus Nodes
        for b in self.raw_net.buses:
            bus_id = b["bus"]
            self.buses[bus_id] = b
            self.graph.add_node(
                bus_id,
                name=b["name"],
                base_kv=b["base_kv"],
                area=b["area"],
                zone=b["zone"],
                vm=b["vm"],
                va=b["va"],
                p_load=0.0,
                q_load=0.0,
                p_gen_max=0.0,
                p_gen_min=0.0,
                is_critical=False,
                cluster_id=0
            )

        # 2. Accumulate Loads per Bus
        for ld in self.raw_net.loads:
            if ld["status"] == 1:
                b_id = ld["bus"]
                if b_id in self.graph:
                    self.graph.nodes[b_id]["p_load"] += ld["p_mw"]
                    self.graph.nodes[b_id]["q_load"] += ld["q_mvar"]
                    # Critical-infrastructure proxy: large concentrated loads (>= 25 MW)
                    # represent aggregated critical facilities (hospitals, water
                    # treatment, data centers, transit) that must be prioritized for
                    # restoration. This is a documented modeling assumption (the ARPA-E
                    # dataset does not tag facility type); the threshold is applied
                    # uniformly and is the ONLY basis for the critical designation.
                    if self.graph.nodes[b_id]["p_load"] >= self.CRITICAL_LOAD_THRESHOLD_MW:
                        self.graph.nodes[b_id]["is_critical"] = True
                self.loads[(b_id, ld["id"])] = ld

        # 3. Accumulate Generator Capacity per Bus
        for g in self.raw_net.generators:
            if g["stat"] == 1:
                b_id = g["bus"]
                if b_id in self.graph:
                    self.graph.nodes[b_id]["p_gen_max"] += g["pt"]
                    self.graph.nodes[b_id]["p_gen_min"] += g["pb"]
                self.generators[(b_id, g["id"])] = g

        # 4. Add Transmission Lines and Transformers as Edges
        for branch in self.raw_net.branches:
            if branch["status"] == 1:
                u, v = branch["from"], branch["to"]
                r, x = branch["r"], branch["x"]
                impedance = max(np.sqrt(r**2 + x**2), 1e-6)
                admittance = 1.0 / impedance
                rate_a = branch["rate_a"] if branch["rate_a"] > 0 else 100.0
                
                self.graph.add_edge(
                    u, v,
                    key=f"line_{branch['ckt']}",
                    type="line",
                    r=r, x=x,
                    weight=admittance,
                    rate_a=rate_a,
                    is_pcc=False
                )

        for xfmr in self.raw_net.transformers:
            if xfmr["status"] == 1:
                u, v = xfmr["from"], xfmr["to"]
                r, x = xfmr["r"], xfmr["x"]
                impedance = max(np.sqrt(r**2 + x**2), 1e-6)
                admittance = 1.0 / impedance
                rate_a = xfmr["rate_a"] if xfmr["rate_a"] > 0 else 100.0
                
                self.graph.add_edge(
                    u, v,
                    key=f"xfmr_{xfmr['ckt']}",
                    type="transformer",
                    r=r, x=x,
                    weight=admittance,
                    rate_a=rate_a,
                    is_pcc=False
                )

    def identify_microgrids_spectral(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """Partition grid into microgrid clusters using spectral graph clustering on line admittances.
        
        Returns:
            Dict mapping cluster_id (0..n_clusters-1) to list of bus IDs in that microgrid.
        """
        node_list = list(self.graph.nodes())
        adj_matrix = nx.to_numpy_array(self.graph, nodelist=node_list, weight="weight")
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42
        )
        labels = spectral.fit_predict(adj_matrix)
        
        clusters: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
        for idx, cluster_id in enumerate(labels):
            bus_id = node_list[idx]
            self.graph.nodes[bus_id]["cluster_id"] = int(cluster_id)
            clusters[int(cluster_id)].append(bus_id)
            
        # Identify Point of Common Coupling (PCC) tie-lines connecting different clusters
        for u, v, data in self.graph.edges(data=True):
            c_u = self.graph.nodes[u]["cluster_id"]
            c_v = self.graph.nodes[v]["cluster_id"]
            data["is_pcc"] = (c_u != c_v)

        return clusters

    def build_generator_cost_map(self) -> Dict[Tuple[int, str], Dict]:
        """Build mapping from (bus, gen_id) to real cubic cost coefficients from case.rop.
        
        Returns:
            Dict mapping (bus_id, gen_id_str) to:
                cubic_coeffs: [a, b, c, d] for C(P) = aP³ + bP² + cP + d ($/h)
                p_min, p_max: operating range (MW) from PWL cost table
                pwl_points: original piecewise-linear breakpoints [(P, Cost), ...]
        """
        if not self.rop_data:
            return {}

        cost_map: Dict[Tuple[int, str], Dict] = {}
        dispatch = self.rop_data.get("dispatch", [])
        tables = self.rop_data.get("dispatch_tables", {})
        pwl = self.rop_data.get("pwl", {})

        for d in dispatch:
            bus = d["bus"]
            gen_id = d["id"]
            tbl = tables.get(d["disptbl"], {})
            ctbl_id = tbl.get("ctbl", -1)
            pts = pwl.get(ctbl_id, [])

            if len(pts) < 2:
                continue

            p_arr = np.array([x for x, _ in pts])
            c_arr = np.array([y for _, y in pts])

            # Fit cubic (degree 3) when we have enough points, degrade gracefully
            if len(pts) >= 4:
                coeffs = np.polyfit(p_arr, c_arr, 3)
            elif len(pts) >= 3:
                quad = np.polyfit(p_arr, c_arr, 2)
                coeffs = np.array([0.0, quad[0], quad[1], quad[2]])
            else:
                lin = np.polyfit(p_arr, c_arr, 1)
                coeffs = np.array([0.0, 0.0, lin[0], lin[1]])

            cost_map[(bus, gen_id)] = {
                "cubic_coeffs": coeffs.tolist(),
                "p_min": float(p_arr.min()),
                "p_max": float(p_arr.max()),
                "pwl_points": pts,
            }

        return cost_map

    def evaluate_generation_cost(self, cost_map: Dict[Tuple[int, str], Dict],
                                  dispatch: Dict[Tuple[int, str], float] | None = None) -> float:
        """Evaluate total generation cost ($/h) using real cubic cost curves.
        
        Args:
            cost_map: Output of build_generator_cost_map().
            dispatch: Optional dict of (bus, gen_id) -> P_MW. If None, uses raw case dispatch points.
            
        Returns:
            Total generation cost in $/h.
        """
        total = 0.0
        for (bus, gen_id), info in cost_map.items():
            a, b, c, d = info["cubic_coeffs"]
            if dispatch is not None:
                p_g = dispatch.get((bus, gen_id), 0.0)
            else:
                # Use the dispatch point from the raw case file
                raw_gen = self.generators.get((bus, gen_id))
                p_g = raw_gen["p_mw"] if raw_gen else 0.0

            # Clamp to operating range
            p_g = max(info["p_min"], min(p_g, info["p_max"]))
            cost_g = a * p_g**3 + b * p_g**2 + c * p_g + d
            total += max(0.0, cost_g)
        return total

    def get_cluster_generators(self, clusters: Dict[int, List[int]]) -> Dict[int, List[Dict]]:
        """Get generators belonging to each cluster with their cost curves.
        
        Returns:
            Dict mapping cluster_id -> list of generator dicts with cost info.
        """
        cost_map = self.build_generator_cost_map()
        cluster_gens: Dict[int, List[Dict]] = {cid: [] for cid in clusters}

        for (bus, gen_id), gen_data in self.generators.items():
            # Find which cluster this bus belongs to
            cluster_id = self.graph.nodes[bus].get("cluster_id", -1)
            if cluster_id in cluster_gens:
                cost_info = cost_map.get((bus, gen_id), {})
                cluster_gens[cluster_id].append({
                    "bus": bus,
                    "gen_id": gen_id,
                    "p_mw": gen_data["p_mw"],
                    "p_min": gen_data["pb"],
                    "p_max": gen_data["pt"],
                    "cubic_coeffs": cost_info.get("cubic_coeffs", [0, 0, 20.0, 0]),
                    "cost_p_min": cost_info.get("p_min", gen_data["pb"]),
                    "cost_p_max": cost_info.get("p_max", gen_data["pt"]),
                })

        return cluster_gens

    def compute_microgrid_upgrade_plan(self, clusters: Dict[int, List[int]]) -> List[Dict]:
        """Compute minimum cost resource upgrades (DERs, BESS) required for self-sustaining islanded mode.
        
        Returns:
            List of upgrade requirements per microgrid cluster.
        """
        upgrades = []
        for cid, bus_list in clusters.items():
            total_load = sum(self.graph.nodes[b]["p_load"] for b in bus_list)
            critical_load = sum(self.graph.nodes[b]["p_load"] for b in bus_list if self.graph.nodes[b]["is_critical"])
            existing_gen = sum(self.graph.nodes[b]["p_gen_max"] for b in bus_list)
            
            # Contingency-aware N-1 sizing: target must cover load + 15% margin even after losing the largest generator
            cluster_gens = [g["pt"] for g_key, g in self.generators.items() if g["bus"] in bus_list and g["stat"] == 1]
            max_gen = max(cluster_gens) if cluster_gens else 0.0
            
            target_capacity = 1.15 * total_load + max_gen
            gen_deficit = max(0.0, target_capacity - existing_gen)
            
            # Sizing strategy: 60% dispatchable microturbine, 40% BESS + Solar
            required_microturbine = 0.6 * gen_deficit
            required_bess_mw = 0.4 * gen_deficit
            required_bess_mwh = 4.0 * required_bess_mw  # 4-hour storage
            
            # Cost model ($/kW, $/kWh)
            cost_microturbine = required_microturbine * 1000 * 950  # $950/kW
            cost_bess = (required_bess_mw * 1000 * 300) + (required_bess_mwh * 1000 * 350) # $300/kW + $350/kWh
            total_upgrade_cost = cost_microturbine + cost_bess
            
            upgrades.append({
                "cluster_id": cid,
                "num_buses": len(bus_list),
                "total_load_mw": round(total_load, 2),
                "critical_load_mw": round(critical_load, 2),
                "existing_gen_mw": round(existing_gen, 2),
                "gen_deficit_mw": round(gen_deficit, 2),
                "add_microturbine_mw": round(required_microturbine, 2),
                "add_bess_mw": round(required_bess_mw, 2),
                "add_bess_mwh": round(required_bess_mwh, 2),
                "upgrade_cost_usd": round(total_upgrade_cost, 2),
                "is_feasible": existing_gen + gen_deficit >= total_load
            })
            
        return upgrades


if __name__ == "__main__":
    # Test script execution on Network 03
    raw_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "scenario_1" / "case.raw"
    rop_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "scenario_1" / "case.rop"
    con_f = ROOT / "Original_Dataset_Offline_Edition_1" / "Network_03O-10" / "scenario_1" / "case.con"
    
    print(f"Loading grid graph model from {raw_f.name}...")
    grid = GridGraphModel(raw_f, rop_f, con_f)
    print(f"Buses: {grid.graph.number_of_nodes()}, Edges: {grid.graph.number_of_edges()}")
    
    clusters = grid.identify_microgrids_spectral(n_clusters=5)
    print(f"Identified {len(clusters)} microgrid clusters.")
    
    upgrades = grid.compute_microgrid_upgrade_plan(clusters)
    print("\nMicrogrid Upgrade Plan:")
    for u in upgrades:
        print(f"  Cluster {u['cluster_id']}: Load={u['total_load_mw']}MW, Critical={u['critical_load_mw']}MW, Deficit={u['gen_deficit_mw']}MW, Cost=${u['upgrade_cost_usd']:,.2f}")
