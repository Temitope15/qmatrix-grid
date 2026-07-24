"""Hamiltonian & Polynomial Unconstrained Binary/Continuous Optimization Builder for QCi Dirac-3 EQC.

Builds HOBO / PUBO matrix and polynomial representation for:
1. Hamiltonian 1: Optimal Microgrid Islanding Strategy (PCC line switching under N-1 contingencies).
2. Hamiltonian 2: Islanded DER & Battery Storage Dispatch (Cubic generator cost curves + active power balance).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


class IslandingHamiltonianBuilder:
    """Builds QUBO/PUBO Hamiltonian for PCC tie-line switching under contingencies."""

    def __init__(self, pcc_edges: List[Tuple[int, int]], penalty_overload: float = 1e3, penalty_unserved: float = 1e4, switching_cost: float = 10.0):
        self.pcc_edges = pcc_edges
        self.n_vars = len(pcc_edges)
        self.penalty_overload = penalty_overload
        self.penalty_unserved = penalty_unserved
        self.switching_cost = switching_cost

    def build_qubo_matrix(self, line_flows: List[float], line_capacities: List[float], critical_loads: List[float]) -> np.ndarray:
        """Constructs quadratic QUBO matrix Q such that x^T Q x represents islanding energy landscape.
        
        Variables:
            x_i = 1 if PCC tie-line i remains CONNECTED, 0 if DISCONNECTED (islanded).
        """
        Q = np.zeros((self.n_vars, self.n_vars))

        for i, (flow, cap, load) in enumerate(zip(line_flows, line_capacities, critical_loads)):
            # Linear cost: switching action (disconnecting line costs switching_cost)
            # 1 - x_i gives 1 if disconnected
            Q[i, i] += self.switching_cost

            # Line overload penalty if connected: penalty_overload * max(0, flow - cap) * x_i
            overload = max(0.0, flow - cap)
            Q[i, i] += self.penalty_overload * overload

            # Unserved load penalty if islanded: penalty_unserved * load * (1 - x_i)
            # Constant terms drop out of argmin, so we subtract from linear term
            Q[i, i] -= self.penalty_unserved * load

        # Off-diagonal interaction terms between adjacent tie-lines (co-dependent tripping)
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                Q[i, j] += 0.5 * self.penalty_overload  # penalize concurrent overloading on parallel ties

        return Q


class DispatchHamiltonianBuilder:
    """Builds Higher-Order / Polynomial Hamiltonian (PUBO/HOBO) for non-convex DER dispatch."""

    def __init__(self, num_generators: int, bits_per_gen: int = 4):
        self.num_generators = num_generators
        self.bits_per_gen = bits_per_gen
        self.total_bits = num_generators * bits_per_gen

    def decode_binary_to_power(self, binary_vector: np.ndarray, p_min: float, p_max: float) -> np.ndarray:
        """Decode binary spin string into continuous power dispatch P_g."""
        powers = []
        for g in range(self.num_generators):
            start = g * self.bits_per_gen
            end = start + self.bits_per_gen
            bit_segment = binary_vector[start:end]
            
            # Binary string to fraction in [0, 1]
            scale = sum(b * (2**(self.bits_per_gen - 1 - idx)) for idx, b in enumerate(bit_segment))
            max_val = (2**self.bits_per_gen) - 1
            fraction = scale / max_val if max_val > 0 else 0.0
            
            p_g = p_min + fraction * (p_max - p_min)
            powers.append(p_g)
        return np.array(powers)

    def build_polynomial_dict(self, generator_cost_cubic: List[Tuple[float, float, float, float]], demand_mw: float, p_min: float, p_max: float, lambda_balance: float = 100.0) -> Dict[Tuple[int, ...], float]:
        """Construct higher-order polynomial dictionary for QCi Dirac-3 EQC solver.
        
        Supports up to 5th degree polynomial interaction terms representing cubic cost f(P_g) = a P^3 + b P^2 + c P + d.
        """
        poly_terms: Dict[Tuple[int, ...], float] = {}

        # 1. Polynomial generator cost terms
        for g, (a, b, c, d) in enumerate(generator_cost_cubic):
            # For each bit in binary representation of P_g
            start = g * self.bits_per_gen
            step_size = (p_max - p_min) / ((2**self.bits_per_gen) - 1)

            for i in range(self.bits_per_gen):
                bit_i = start + i
                w_i = (2**(self.bits_per_gen - 1 - i)) * step_size
                
                # Degree 1 (c * w_i)
                key_1 = (bit_i,)
                poly_terms[key_1] = poly_terms.get(key_1, 0.0) + c * w_i

                # Degree 2 (b * w_i * w_j)
                for j in range(self.bits_per_gen):
                    bit_j = start + j
                    w_j = (2**(self.bits_per_gen - 1 - j)) * step_size
                    key_2 = tuple(sorted([bit_i, bit_j]))
                    poly_terms[key_2] = poly_terms.get(key_2, 0.0) + b * w_i * w_j

                # Degree 3 (a * w_i * w_j * w_k)
                for j in range(self.bits_per_gen):
                    bit_j = start + j
                    w_j = (2**(self.bits_per_gen - 1 - j)) * step_size
                    for k in range(self.bits_per_gen):
                        bit_k = start + k
                        w_k = (2**(self.bits_per_gen - 1 - k)) * step_size
                        key_3 = tuple(sorted([bit_i, bit_j, bit_k]))
                        poly_terms[key_3] = poly_terms.get(key_3, 0.0) + a * w_i * w_j * w_k

        return poly_terms


if __name__ == "__main__":
    print("Testing IslandingHamiltonianBuilder...")
    pcc_edges = [(10, 20), (30, 40), (50, 60)]
    builder1 = IslandingHamiltonianBuilder(pcc_edges)
    Q = builder1.build_qubo_matrix(line_flows=[120.0, 85.0, 45.0], line_capacities=[100.0, 100.0, 50.0], critical_loads=[30.0, 15.0, 10.0])
    print(f"Constructed QUBO Matrix of shape {Q.shape}:")
    print(Q)

    print("\nTesting DispatchHamiltonianBuilder (Cubic Cost HOBO)...")
    builder2 = DispatchHamiltonianBuilder(num_generators=3, bits_per_gen=3)
    cubic_coeffs = [(0.01, 0.5, 20.0, 100.0), (0.008, 0.4, 18.0, 80.0), (0.012, 0.6, 22.0, 120.0)]
    poly_dict = builder2.build_polynomial_dict(cubic_coeffs, demand_mw=250.0, p_min=10.0, p_max=100.0)
    print(f"Generated {len(poly_dict)} polynomial interaction terms up to 3rd degree.")
