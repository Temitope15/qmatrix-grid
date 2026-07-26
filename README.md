# QMatrix — Cost Optimization in Resilient Power Grids

**Team QMatrix** · 2026 Global Industry Challenge · QCi Energy Infrastructure Track
**Project:** Non-convex cubic dispatch and resilience in microgrid networks via QCi Dirac-3

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com/?gitHubUrl=https://github.com/Temitope15/qmatrix-grid.git)

| Role | Member |
|---|---|
| Project Lead & Quantum Strategy | Sharmila L |
| Lead Coder | Temitope Akinsunmade |
| Quantum & Energy Systems | Abdullahi Tajudeen O. |
| Data & Simulation Analyst | Joseph Falade |
| Computational Modeling | Udochukwu Okorie |

---

## Overview

This project optimizes a network of microgrids derived from the ARPA-E GO Challenge 1 grid
`Network_03O-10` (793 buses, 904 branches/transformers, 82 active generators, 7,801.5 MW load).
It has two parts:

1. **Non-convex cubic economic dispatch on QCi Dirac-3.** Thermal generation modelled with the
   cubic cost `C(P)=aP³+bP²+cP+d` is non-convex, so convex LP/QP solvers must relax it. Economic
   dispatch maps directly onto Dirac-3's continuous encoding: the objective is a degree-3 polynomial
   and the power-balance constraint `ΣPᵢ=Demand` is the device `sum_constraint`. The dispatch runs on
   Dirac-3 hardware through `eqc-models` and is compared against a convex QP baseline and a multistart
   global optimum on the same instance, for two cost models (a dataset-fitted control and a non-convex
   valve-point model) across ten scenarios.

2. **Classical resilience pipeline.** Spectral partitioning into five microgrids (23 PCC tie-lines),
   an LP DC-OPF sweep over all 91 N-1 contingencies, a transmission-blackout plus secondary-outage
   study, and contingency-aware DER siting.

All reported values are computed by the submitted code; none are hardcoded.

## Repository structure

```
.
├── models/
│   ├── grid_graph.py          # Spectral clustering, topology, cost-curve fitting, DER siting
│   ├── classical_solver.py    # LP DC-OPF, 91-contingency N-1 sweep, multi-island OPF
│   ├── cost_models.py         # Model A (dataset cubic) and Model B (non-convex valve-point cubic)
│   └── dispatch_dirac3.py     # Dirac-3 dispatch solver (eqc-models) and convex/global baselines
├── experiments/
│   └── dispatch_benchmark.py  # Full dispatch study -> doc/stats_dispatch.json
├── src/extract_stats.py       # PSS/E v33 parser (case.raw / case.rop / case.con)
├── doc/
│   ├── QMatrix__Phase3_Final_Report.md   # Write-up
│   └── stats_dispatch.json               # Machine-readable results
├── Original_Dataset_Offline_Edition_1/Network_03O-10/   # ARPA-E GO dataset
├── phase3_final_submission.ipynb         # Main submission notebook
├── requirements.txt
└── README.md
```

## Setup and execution on qBraid (step by step)

1. **Open on qBraid.** On the Aqora challenge page click **Launch on qBraid** (or use the button above,
   or in qBraid Lab choose *File → New → Clone Repository* and paste the repo URL). The repository opens
   in qBraid Lab under `~/qmatrix-grid/` (or your chosen name).
2. **Open a terminal** in qBraid Lab (*File → New → Terminal*) and move into the project:
   ```bash
   cd ~/qmatrix-grid           # use the folder name shown in the file browser
   pip install -r requirements.txt
   ```
3. **Add your QCi Dirac-3 token.** QCi finalists receive a token through qBraid; put it in a `.env` file:
   ```bash
   cp .env.example .env
   nano .env                   # set QCI_API_TOKEN=your_token, then save (Ctrl+O, Enter, Ctrl+X)
   ```
   `QCI_API_URL` defaults to `https://api.qci-prod.com`. The code maps `QCI_API_TOKEN` to the
   `QCI_TOKEN` name that `eqc-models` expects, so you only set the one variable.
4. **Confirm the token works** (optional, no credits used):
   ```bash
   python -c "import os; from dotenv import load_dotenv; from qci_client import QciClient; load_dotenv(); print(QciClient(api_token=os.getenv('QCI_API_TOKEN')).get_allocations())"
   ```
   You should see a `dirac` allocation with a positive `seconds` balance.
5. **Run the notebook.** Open `phase3_final_submission.ipynb`, select a **Python 3.10–3.12** kernel, and
   choose **Run → Run All Cells**. The full run takes ~3–6 minutes: it partitions the grid, sweeps the
   91 contingencies, submits a **live Dirac-3 dispatch job**, loads the 10-scenario study, and prints a
   summary. If Cell 1 installs anything, restart the kernel once and Run All again.

**Verifying the run.** The notebook prints, live: `23 PCC tie-lines`, `Dirac-3 saves … over the convex
baseline` (Step 5), the 10-scenario table with `MEAN: Dirac-3 saves 3.00% vs convex QP` (Step 6),
`Blackout + island N-1 shed: 76.38 MW → 0.00 MW` (Step 7), and a final summary (Step 8).

**Regenerate the full 10-scenario study from scratch** (~11 live Dirac-3 jobs, a few minutes):
```bash
python experiments/dispatch_benchmark.py          # writes doc/stats_dispatch.json
python experiments/dispatch_benchmark.py --quick  # scenario 1 only (fast smoke test)
```

**Notes.** Dirac-3 is a stochastic sampler, so a single live job varies run-to-run (it consistently
beats the convex baseline); the headline figures are the 10-scenario means in `doc/stats_dispatch.json`.
A local run is identical after `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Inputs and outputs

**Inputs:** `Network_03O-10` `case.raw` (bus/branch/generator model), `case.rop` (cost curves),
`case.con` (91 contingencies); `QCI_API_TOKEN` for Dirac-3.

**Outputs:** `doc/stats_dispatch.json` (dispatch study), and notebook cell outputs covering the
microgrid partition and DER siting, the N-1 sweep, the live Dirac-3 dispatch versus classical
baselines, the ten-scenario comparison with plots, the blackout resilience analysis, and the
Dirac-3 encoding.

## Key results

Non-convex cubic dispatch, mean over ten scenarios (Dirac-3 hardware):

| Metric | Convex QP baseline | Dirac-3 EQC | Global optimum |
|---|---|---|---|
| Optimality gap vs global | 5.26 % | 2.10 % | — |
| Cost saving vs convex QP | — | 3.00 % (range 1.12–4.23 %) | — |
| Feasibility (10 scenarios) | feasible | 10/10 feasible, 0.0 MW balance error | feasible |
| Dataset control (near-convex) | optimal | 0.30 % gap | optimal |

Classical resilience pipeline:

| Metric | Value |
|---|---|
| N-1 contingencies (intact grid) | 91, 0.00 MW shed |
| Blackout + N-1 gen trip (no upgrade) | 76.38 MW (0.98 %) shed |
| Blackout + N-1 gen trip (with DER) | 0.00 MW shed |
| DER upgrade cost (Cluster 3 only) | $123,308,875 |

## Assumptions and limitations

- On near-convex costs (dataset control) Dirac-3 matches the classical optimum; the advantage appears
  only when the cost is non-convex.
- Model B cost coefficients are calibrated from published valve-point benchmarks (Walters & Sheble 1993;
  Sinha et al. 2003) and applied to the dataset's real generator ranges; non-convexity is verified per unit.
- Dirac-3 is a stochastic sampler; headline figures are ten-scenario means.
- Islanding, contingency, and siting use DC power flow; the dispatch is a single-bus economic dispatch
  per microgrid. Multi-period horizons, battery state-of-charge dynamics, two-stage stochastic recourse,
  unit commitment, and AC security constraints are scoped as future work (see the report).

## Dependencies

See `requirements.txt`: `eqc-models` and `qci-client` (Dirac-3), `pandapower` and `scipy` (DC-OPF),
`networkx` and `scikit-learn` (topology and spectral clustering), `numpy`, `matplotlib`, `python-dotenv`.
