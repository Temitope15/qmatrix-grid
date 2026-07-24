"""QCi Dirac-3 API Adapter for EQC Quantum Optimization.

Provides robust interface for submitting QUBO/HOBO/Hamiltonian optimization jobs to QCi's Dirac-3 hardware
using `qci-client` and `eqc-models`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv

# Load environment variables (.env)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class QciDirac3Adapter:
    """Adapter for interacting with QCi Dirac-3 Quantum Computer API."""

    def __init__(self, api_token: str | None = None, api_url: str | None = None):
        self.api_token = api_token or os.getenv("QCI_API_TOKEN")
        self.api_url = api_url or os.getenv("QCI_API_URL", "https://api.qci-prod.com")
        self.client = None
        self.is_connected = False
        
        self._init_client()

    def _init_client(self) -> None:
        """Initialize QciClient with token credentials."""
        if not self.api_token:
            print("WARNING: QCI_API_TOKEN not found in environment.")
            return

        try:
            from qci_client import QciClient
            self.client = QciClient(api_token=self.api_token, url=self.api_url)
            allocations = self.client.get_allocations()
            dirac_alloc = allocations.get("allocations", {}).get("dirac", {})
            print(f"[QCi Dirac-3] Successfully authenticated! Active Dirac allocation: {dirac_alloc}")
            self.is_connected = True
        except Exception as e:
            print(f"[QCi Dirac-3] Connection notice: {e}")
            self.is_connected = False

    def solve_qubo_local_sim(self, Q: np.ndarray, num_samples: int = 2000) -> Tuple[np.ndarray, float]:
        """Local classical simulated annealing fallback for testing QUBO ground state."""
        n = Q.shape[0]
        best_x = None
        best_energy = float("inf")
        
        for _ in range(num_samples):
            x = np.random.randint(0, 2, size=n)
            energy = float(x.T @ Q @ x)
            if energy < best_energy:
                best_energy = energy
                best_x = x
                
        return best_x if best_x is not None else np.zeros(n, dtype=int), best_energy

    def solve_hamiltonian_dirac3(self, H_matrix: np.ndarray, nsamples: int = 5) -> Tuple[np.ndarray, float]:
        """Submit Hamiltonian matrix directly to QCi Dirac-3 quantum hardware using `sample-hamiltonian`.
        
        Args:
            H_matrix: Symmetric Hamiltonian matrix.
            nsamples: Number of quantum measurement samples.
            
        Returns:
            Tuple of (optimal_solution_vector, energy_value).
        """
        if not self.is_connected or self.client is None:
            print("[QCi Adapter] Running local EQC simulated solver...")
            return self.solve_qubo_local_sim(H_matrix)

        try:
            h_file = {
                "file_name": "hamiltonian.json",
                "file_config": {
                    "hamiltonian": {
                        "data": H_matrix
                    }
                }
            }
            print("[QCi Dirac-3] Uploading Hamiltonian matrix payload...")
            file_response = self.client.upload_file(file=h_file)
            file_id = file_response["file_id"]

            job_body = self.client.build_job_body(
                job_type="sample-hamiltonian",
                job_params={
                    "device_type": "dirac-3",
                    "nsamples": nsamples
                },
                hamiltonian_file_id=file_id,
                job_name="QMatrix_Phase3_Islanding_Dirac3"
            )
            print("[QCi Dirac-3] Submitting Hamiltonian job to Dirac-3 backend...")
            job_response = self.client.submit_job(job_body=job_body)
            job_id = job_response["job_id"]
            print(f"[QCi Dirac-3] Job submitted successfully (ID: {job_id}). Waiting for quantum execution...")

            # Poll for results
            for attempt in range(15):
                time.sleep(2)
                res = self.client.get_job_results(job_id=job_id)
                status = res.get("status", "UNKNOWN")
                if status == "COMPLETED":
                    solution_data = res.get("results", {})
                    solutions = solution_data.get("solutions", [])
                    energies = solution_data.get("energies", [])
                    if solutions:
                        best_idx = int(np.argmin(energies)) if energies else 0
                        best_sol = np.array(solutions[best_idx], dtype=int)
                        best_energy = float(energies[best_idx]) if energies else float(best_sol.T @ H_matrix @ best_sol)
                        print(f"[QCi Dirac-3] Quantum Execution Completed! Energy: {best_energy:.4f}")
                        return best_sol, best_energy
                elif status in ["ERROR", "CANCELLED", "ERRORED"]:
                    print(f"[QCi Dirac-3] Job ended with status: {status}. Using simulator fallback.")
                    break

            return self.solve_qubo_local_sim(H_matrix)

        except Exception as e:
            print(f"[QCi Adapter] Exception during Dirac-3 execution ({e}). Using local simulator fallback.")
            return self.solve_qubo_local_sim(H_matrix)

    def solve_islanding_qubo(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Wrapper to solve QUBO using Dirac-3 Hamiltonian sampler or local EQC simulator."""
        return self.solve_hamiltonian_dirac3(Q)


if __name__ == "__main__":
    print("Testing QciDirac3Adapter...")
    adapter = QciDirac3Adapter()
    
    H_test = np.array([
        [ 1.0, -2.0],
        [-2.0,  1.0]
    ])
    sol, energy = adapter.solve_hamiltonian_dirac3(H_test)
    print(f"Optimal Solution: {sol}, Energy: {energy}")
