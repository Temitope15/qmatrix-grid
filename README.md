# QMatrix — Phase 3: Cost Optimization in Resilient Power Grids

**Team QMatrix** | 2026 Global Industry Challenge | QCi Energy Infrastructure Track

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com/?gitHubUrl=https://github.com/YOUR_USERNAME/YOUR_REPO.git)

| Role | Member |
|---|---|
| Project Lead & Quantum Strategy Architect | Sharmila L |
| Lead Coder | Temitope Akinsunmade |
| Lead Quantum & Energy Systems Specialist | Abdullahi Tajudeen O. |
| Lead Data & Simulation Analyst | Joseph Falade |
| Lead Computational Modeling | Udochukwu Okorie |

---

## Project Overview

We model the ARPA-E GO Competition Challenge 1 benchmark grid (`Network_03O-10`: 793 buses, 904 branches/transformers, 210 generators, 7,801 MW total load) and solve two coupled optimization problems using QCi's Dirac-3 Entropy Quantum Computer:

1. **Islanding QUBO** — which of the 23 PCC tie-lines to open under a transmission blackout, minimizing unserved load across 5 spectral-graph microgrid clusters.
2. **Dispatch HOBO** — non-convex cubic generator cost minimization across 8 generators (24 qubits), using real cost curves from `case.rop`.

We compare all quantum results against a classical LP-based DC-OPF baseline on the same problem instances.

---

## Clean Repository Structure

```
.
├── models/                       # Core optimization modules
│   ├── __init__.py               # Package marker
│   ├── grid_graph.py             # Spectral clustering, topology, upgrade siting
│   ├── hamiltonian_builder.py    # QUBO (islanding) + HOBO (dispatch) construction
│   ├── classical_solver.py       # LP DC-OPF, N-1 sweep, multi-island OPF
│   └── qci_adapter.py           # Dirac-3 API client + local simulator fallback
├── src/
│   ├── __init__.py               # Package marker
│   └── extract_stats.py         # PSS/E v33 parser (case.raw, case.rop, case.con)
├── experiments/
│   ├── sprint678_complete.py    # End-to-end reproducibility script (Sprints 6, 7, 8)
│   ├── sprint6_contingency_analysis.py  # Sprint 6 standalone contingency script
│   └── run_quantum_dirac3.py    # Dirac-3 orchestration module
├── doc/
│   ├── QMatrix__Phase3_Final_Report.md   # Written proposal report
│   ├── stats_phase3_final.json           # Final machine-readable metrics
│   └── figures/                          # Topology and cost curve plots
├── Original_Dataset_Offline_Edition_1/   # ARPA-E GO dataset (included)
│   └── Network_03O-10/
│       ├── case.rop                      # Generator cost curves
│       └── scenario_1/
│           ├── case.raw                  # Bus/branch/generator data
│           └── case.con                  # 91 contingency definitions
├── phase3_final_submission.ipynb         # Main executable submission notebook
├── requirements.txt                      # Dependencies for qBraid & local environments
├── pyproject.toml                        # Project configuration
├── .env.example                          # Credential template for QCi API token
├── .gitignore                            # Clean repository hygiene rules
└── README.md                             # Project instructions & documentation
```

---

## Setup & Execution Instructions

### Option 1: On qBraid

1. Launch this repository on qBraid by clicking the **Launch on qBraid** button above or importing the repository URL.
2. Open a terminal on qBraid and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your QCi credentials by creating `.env` from the provided `.env.example`:
   ```bash
   cp .env.example .env
   # Open .env and insert your QCI_API_TOKEN
   ```
4. Open `phase3_final_submission.ipynb` using Jupyter Notebook or JupyterLab, select Python 3 kernel, and run all cells.

### Option 2: Local Python Environment

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```
2. Create and activate a Python virtual environment (Python 3.10 to 3.12 recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure API credentials:
   ```bash
   cp .env.example .env
   # Edit .env and set QCI_API_TOKEN=your_qci_api_token
   ```
5. Run the submission notebook using Jupyter Notebook:
   ```bash
   jupyter notebook phase3_final_submission.ipynb
   ```
6. Alternatively, execute the complete pipeline script directly from the terminal:
   ```bash
   python experiments/sprint678_complete.py
   ```

---

## Expected Inputs and Outputs

### Inputs
- `Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_1/case.raw` — 793-bus power system model
- `Original_Dataset_Offline_Edition_1/Network_03O-10/case.rop` — 210 generator piecewise-linear cost curves
- `Original_Dataset_Offline_Edition_1/Network_03O-10/scenario_1/case.con` — 91 contingency definitions
- `QCI_API_TOKEN` environment variable — QCi Dirac-3 API authentication key

### Outputs
- `doc/stats_phase3_final.json` — Machine-readable summary of all benchmark metrics
- Notebook cell outputs containing:
  - 5-cluster partition with per-cluster load/generation breakdown
  - 91-contingency sweep results (0.00 MW unserved on intact grid)
  - QCi Dirac-3 QUBO execution results with job IDs and ground state energies
  - Brute-force ground state comparison ($2^{23} = 8,388,608$ configurations)
  - Transmission blackout resilience: 76.38 MW shed (unupgraded) vs 0.00 MW shed (with $123.3M DER upgrades)
  - HOBO dispatch Hamiltonian statistics (152 terms, 24 qubits)

---

## Key Results

| Metric | Classical Baseline | QCi Dirac-3 EQC |
|---|---|---|
| N-1 Contingencies (intact grid) | 0.00 MW shed (91/91 secure) | — |
| Blackout + N-1 gen trip (no upgrades) | 76.38 MW (0.98%) shed | 76.38 MW shed |
| Blackout + N-1 gen trip (with quantum-sited DER) | — | **0.00 MW shed** |
| DER Upgrade Cost (Cluster 3 only) | — | **$123,308,875** |
| QUBO Variables | — | 23 (PCC tie-lines) |
| HOBO Qubits | — | 24 (8 generators × 3 bits) |
| Brute-force ground state energy | −6,045,330 | — |
| Dirac-3 QUBO energy | — | −3,138,017 |
| Brute-force evaluation time | 107 s (8.4M states) | — |
| Dirac-3 wall-clock time | — | 9.3 s |

---

## Quantum Advantage & Limitations

The quantum approach provides a concrete benefit in the **microgrid upgrade siting** pipeline: the Dirac-3 islanding solution identifies which PCC tie-lines to open, and the resulting island topology determines where generation deficits emerge under N-1 conditions. This drives the targeted $123.3M DER investment in Cluster 3 — a decision that classical load flow alone does not surface because it treats the grid as a single connected network.

### Known Limitations & Observations
1. **Continuous / Integer Outputs**: QCi Dirac-3 is a continuous Hamiltonian sampler, so returned vector values are integer/continuous rather than strictly binary. We threshold the vector outputs to $\{0, 1\}$ for tie-line switching decisions.
2. **Problem Scale**: At 23 binary variables ($2^{23} \approx 8.4\text{M}$ states), exact brute-force search is feasible and finds a deeper binary energy minimum (−6,045,330 vs −3,138,017). The primary value of quantum optimization lies in scaling to larger networks (>40 tie-lines) where brute-force search becomes intractable.
3. **DC Linearization**: Power flow calculations use DC linearization (B-theta), which is standard for contingency screening. AC power flow verification was performed on the base case via Pandapower.
4. **Dirac-3 Hardware Connectivity**: In cases of network API timeouts, the solution seamlessly uses a local eigensolver simulation fallback to ensure pipeline robustness.

---

## Reproducing Pipeline Artifacts

To regenerate the metric reports and figures directly from source:

```bash
# Run the complete Phase 3 pipeline verification
python experiments/sprint678_complete.py

# Extract raw network statistics and generate topology plots
python src/extract_stats.py
```

---

## Dependencies

Listed in `requirements.txt`. Core packages:
- `pandapower` — power system modeling and power flow verification
- `scipy` — LP solver (HiGHS) for DC-OPF
- `networkx` & `scikit-learn` — graph topology and spectral clustering
- `qci-client` — QCi Dirac-3 EQC API client
- `python-dotenv` — credential management
- `numpy`, `pandas`, `matplotlib` — numerics and plotting
