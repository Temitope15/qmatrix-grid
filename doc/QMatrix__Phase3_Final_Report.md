# Cost Optimization and Contingency-Aware Resilience in Power Grids via Quantum Entropy Computers

**Team QMatrix** — Sharmila L, Temitope Akinsunmade, Abdullahi Tajudeen O., Joseph Falade, Udochukwu Okorie  
2026 Global Industry Challenge — Energy Infrastructure Track (QCi)

---

## 1. Problem Statement and Approach

We address the problem of maintaining power grid resilience under transmission outages and generator failures, using the ARPA-E GO Competition Challenge 1 benchmark dataset (`Network_03O-10`). The network has 793 buses, 904 transmission lines and transformers, 210 generators, and serves 7,801 MW of load.

Our approach has three parts:

1. Partition the grid into 5 self-balancing microgrid clusters using spectral graph clustering on line admittances. The clusters are connected by 23 PCC (Point of Common Coupling) tie-lines.
2. Formulate the tie-line switching decision as a 23-variable QUBO and solve it on QCi's Dirac-3 EQC. This determines which tie-lines to open during a transmission blackout so that each island can self-balance.
3. Evaluate the resulting islanded networks using LP-based DC Optimal Power Flow, identify generation deficits under N-1 conditions, and size DER upgrades to close the gap.

All classical baselines use the same grid model and LP solver (SciPy HiGHS) on the same contingency set.

---

## 2. Classical Baseline: N-1 Contingency Analysis

We parsed all 91 contingencies from `case.con` (62 branch outages, 29 generator outages) and ran each through an LP DC-OPF that minimizes load shedding subject to generation limits and power balance.

On the intact centralized grid, the result is straightforward: **0.00 MW unserved load across all 91 contingencies**. The network has sufficient transmission redundancy and generation headroom to handle any single outage.

The interesting case is the transmission blackout scenario — loss of all 23 PCC tie-lines, forcing the grid to operate as 5 isolated microgrids. Under base conditions, each island has enough generation to serve its own load. But when we apply a secondary N-1 generator trip (the largest generator in Cluster 3, 502 MW at bus 268), Cluster 3 sheds **76.38 MW (0.98% of system load)**.

---

## 3. Quantum Optimization: Dirac-3 QUBO Execution

We built a 23×23 QUBO matrix where each binary variable $x_i$ represents a PCC tie-line. The objective encodes three competing terms: minimizing power flow interruption on opened lines, respecting line capacity limits, and protecting critical load. The matrix was submitted to the Dirac-3 `sample-hamiltonian` endpoint.

**Dirac-3 results:**
- Job completed in **9.3 seconds** wall-clock time
- Returned energy: **−3,138,017**
- Solution vector: `[38, 0, 0, ..., 0, 9961]` (23 elements)

**Brute-force validation:**
- Enumerated all $2^{23}$ = 8,388,608 binary configurations in **107 seconds**
- True binary ground state energy: **−6,045,330**

The brute-force solver finds a deeper minimum. This is expected: at 23 variables, the problem is classically tractable, and the Dirac-3 EQC is a continuous Hamiltonian sampler rather than a binary optimizer. Its solution vector contains non-binary integer values, which means the energy is evaluated on a different feasible set. We threshold the Dirac-3 output to binary for the islanding decision.

---

## 4. Per-Contingency Quantum Islanding (15 Jobs on Dirac-3)

We selected 15 representative contingencies (10 largest generator outages, 5 critical branch outages) and submitted a separate QUBO for each to Dirac-3. For every returned solution, we evaluated the actual load shed by running per-island DC-OPF on the resulting sub-networks.

- 11 of 15 jobs executed on Dirac-3 hardware
- 4 fell back to the local simulator due to API timeouts
- All 15 quantum islanding solutions produced **0.00 MW unserved load** under single-contingency conditions

This confirms that the grid's base-case generation surplus is large enough that any reasonable islanding partition — quantum or classical — avoids shedding under single outages. The shedding only appears under the compound scenario (blackout + secondary gen trip in Cluster 3).

---

## 5. Contingency-Aware DER Upgrade Siting

The quantum islanding analysis revealed that Cluster 3 (33 buses, 388 MW load, 849 MW generation) has a **98.65 MW N-1 generation deficit** when its largest generator trips during islanded operation.

We sized an upgrade to close the deficit:

| Component | Capacity | Unit Cost | Total |
|---|---|---|---|
| Microturbine | 59.19 MW | $950/kW | $56,230,500 |
| BESS (Power) | 39.46 MW | $300/kW | $11,838,000 |
| BESS (Energy) | 157.84 MWh | $350/kWh | $55,244,000 |
| **Total** | | | **$123,308,875** |

With this upgrade in place, the blackout + N-1 gen trip scenario produces **0.00 MW unserved load**.

The other four clusters (0, 1, 2, 4) have no generation deficit and require no upgrades.

---

## 6. Non-Convex Generator Dispatch (HOBO)

We fitted cubic cost curves $C(P) = aP^3 + bP^2 + cP + d$ from the 210 piecewise-linear tables in `case.rop`. The base-case generation cost is **$72,943/hour**.

For the largest cluster (Cluster 1, 209 buses), we encoded 8 generators at 3 bits per generator into a 24-qubit HOBO (Higher-Order Binary Optimization) polynomial with **152 non-zero terms** up to degree 3. This payload was submitted to Dirac-3. The HOBO formulation captures the non-convexity in generator costs that LP or QP solvers must linearize.

---

## 7. Summary of Results

| Metric | Value |
|---|---|
| Grid size | 793 buses, 904 branches, 210 generators |
| System load | 7,801 MW (5,212 MW critical) |
| Microgrid clusters | 5 (23 PCC tie-lines) |
| N-1 contingencies evaluated | 91 (0.00 MW shed on intact grid) |
| Blackout + N-1 gen trip shed (no upgrade) | 76.38 MW (0.98%) |
| Blackout + N-1 gen trip shed (with DER) | 0.00 MW |
| DER upgrade cost | $123,308,875 (Cluster 3 only) |
| QUBO size | 23 variables |
| HOBO qubits | 24 (8 generators × 3 bits) |
| Dirac-3 QUBO energy | −3,138,017 |
| Brute-force ground state energy | −6,045,330 |
| Dirac-3 wall-clock time | 9.3 s |
| Brute-force time | 107 s |
| Dirac-3 job success rate | 11/15 (73%) |
| Base-case generation cost | $72,943/h |

---

## 8. Discussion and Limitations

The central finding is that quantum-optimized microgrid islanding, combined with targeted DER upgrades, eliminates load shedding under compound failure scenarios (transmission blackout + secondary generator trip) that classical centralized operation cannot handle without similar infrastructure investment.

However, several limitations should be noted:

**On quantum advantage at this scale:** The 23-variable QUBO is classically tractable. Brute-force finds a better binary solution in 107 seconds. The scaling argument — that Dirac-3 execution time stays roughly constant as variable count grows while brute-force grows as $2^n$ — is valid in principle, but we have not demonstrated it on a larger instance.

**On the Dirac-3 solution format:** The EQC returns continuous/integer-valued vectors, not binary. Direct energy comparison with the binary brute-force is therefore not straightforward. The practical evaluation (per-island DC-OPF) uses the thresholded binary decision, which is valid regardless.

**On the power flow model:** We use DC power flow (B-theta linearization), which is standard for contingency screening but omits reactive power and voltage constraints. Pandapower AC Newton-Raphson was used for base-case convergence verification.

**On reliability of Dirac-3 API:** 4 of 15 job submissions failed due to result retrieval errors and fell back to local simulation. The fallback uses the same Hamiltonian and a classical eigensolver, so the islanding decision is valid but not quantum-computed.

**On dataset scope:** All results are from a single scenario of a single network. The methodology should generalize, but we have not tested it on the other 9 scenarios or different network sizes.

---

*Submitted for the 2026 QCi Global Industry Challenge, Phase 3 Final.*
