# Non-Convex Cost Optimization and Resilience in Microgrid Networks via QCi Dirac-3 Entropy Quantum Computing

**Team QMatrix** — Sharmila L, Temitope Akinsunmade, Abdullahi Tajudeen O., Joseph Falade, Udochukwu Okorie
2026 Global Industry Challenge — Energy Infrastructure Track (QCi) — Phase 3 Final

---

## 1. Problem and Approach

We optimize a network of microgrids derived from the ARPA-E GO Competition Challenge 1 benchmark grid `Network_03O-10` (793 buses, 904 transmission lines and transformers, 82 active generators, 7,801.5 MW system load of which 4,992.9 MW is critical). The submission consists of a quantum optimization of non-convex generator dispatch and a classical resilience pipeline for microgrid design and contingency handling. Each quantum result is compared against a classical method on the same problem instance.

The challenge specifies thermal generation modelled with a higher-fidelity cubic cost function, which is non-convex. QCi's Dirac-3 is an Entropy Quantum Computer that samples polynomial Hamiltonians up to degree five with full variable connectivity and a continuous encoding. Economic dispatch maps directly onto this device. The dispatch problem is

```
minimize    sum_i C_i(P_i)
subject to  sum_i P_i = D                 (power balance)
            Pmin_i <= P_i <= Pmax_i        (generator limits)
```

With the cubic cost `C_i(P) = aP^3 + bP^2 + cP + d`, the objective is a degree-three polynomial and the power-balance constraint `ΣP_i = D` is expressed directly by the device `sum_constraint`. We apply the substitution `P_i = Pmin_i + y_i` with `y_i ≥ 0`, so lower generation limits are satisfied by construction and the sum constraint becomes `Σy_i = D − ΣPmin`. The problem is executed on Dirac-3 hardware through the `eqc-models` `PolynomialModel` and `Dirac3ContinuousCloudSolver`. All classical baselines use the same grid model and the SciPy HiGHS and SLSQP solvers.

## 2. Non-Convex Cubic Dispatch on Dirac-3

We evaluate two cost models on the same generator fleet so that the effect of non-convexity is isolated.

**Model A — dataset cubic (control).** Cubic curves fitted to the piecewise-linear cost tables in `case.rop`. The ARPA-E cost data is near-convex (fitted cubic coefficient close to zero), so this model serves as a control in which the classical optimum is already attainable.

**Model B — non-convex valve-point cubic.** Thermal units with multiple steam-admission valves exhibit valve-point loading, producing a non-monotonic marginal cost. We represent this with a cubic whose marginal cost `m(P) = 3aP^2 + 2bP + c` attains its minimum strictly inside each unit's operating range `[Pmin, Pmax]`. Coefficient magnitudes follow the standard valve-point dispatch benchmarks (Walters and Sheble, *IEEE Trans. Power Systems* 1993; Sinha, Chakrabarti and Chattopadhyay, *IEEE Trans. Evolutionary Computation* 2003). Only the cost shape is calibrated from the literature; the generator count and capacities are taken from the dataset. Non-convexity is confirmed per unit rather than assumed: the second derivative of each cost curve is negative over part of the operating range.

We study the largest microgrid (cluster 0, 229 buses) using its ten largest dispatchable units, all ten non-convex under Model B. Three solvers are compared on identical instances: a convex QP baseline that omits the cubic curvature (its dispatch is then costed on the true cubic curve), the Dirac-3 EQC, and an 80-start SLSQP global reference on the true non-convex cost.

**Model B results, mean over the ten dataset scenarios** (each solved at that scenario's cluster demand):

| Metric | Value |
|---|---|
| Convex QP optimality gap vs global | 5.26 % |
| Dirac-3 optimality gap vs global | 2.10 % |
| Dirac-3 cost saving vs convex QP | 3.00 % (range 1.12–4.23 %) |
| Feasibility | 10/10 feasible, maximum power-balance error 0.0 MW |
| Per-job device runtime | ≈ 30 s |

Dirac-3 improved on the convex baseline in all ten scenarios while returning power-balanced dispatches within generator limits. For a representative scenario the convex dispatch costs $45,127/h, Dirac-3 $43,381/h, and the global optimum $42,929/h.

**Model A results.** With no non-convex units, Dirac-3 reaches within 0.30 % of the classical optimum. On a convex problem the classical solver is already optimal and no advantage is expected; this control confirms that the observed Model B advantage is attributable to non-convexity.

**Encoding.** Ten continuous modes (one per generator), 30 polynomial terms up to degree three, full connectivity, `num_samples = 20`, `relaxation_schedule = 3`, and `sum_constraint = D − ΣPmin`. The Dirac allocation is unmetered, so all scenarios and the control were run on hardware.

## 3. Classical Resilience Pipeline

**Microgrid partition.** Spectral clustering on line admittances partitions the grid into five self-balancing microgrid clusters connected by 23 PCC tie-lines. This is a classical graph procedure and is reported as such.

**N-1 contingency sweep.** All 91 contingencies in `case.con` (62 branch and 29 generator outages) are solved with an LP DC-OPF minimizing load shedding subject to generation limits and DC power balance. On the intact grid the result is 0.00 MW unserved across all 91 contingencies; the network has sufficient redundancy for any single outage.

**Transmission blackout with secondary outage.** Loss of all 23 PCC tie-lines forces operation as five islands. Each island self-balances at base load (0 MW shed). A secondary N-1 trip of the largest unit (501.5 MW) in Cluster 3 causes that cluster to shed 76.38 MW (0.98 % of system load).

**Contingency-aware DER siting.** Cluster 3 (33 buses, 387.9 MW load) has a 98.7 MW N-1 generation deficit when islanded. A sized upgrade of a 59.2 MW microturbine and a 39.5 MW / 157.8 MWh battery system ($123.3 M, standard $/kW and $/kWh unit prices) eliminates the shed under the blackout-plus-N-1 case. The remaining four clusters require no upgrades.

## 4. Summary of Metrics

| Metric | Value |
|---|---|
| Grid | 793 buses, 904 branches, 82 active generators |
| System / critical load | 7,801.5 MW / 4,992.9 MW (≥ 25 MW threshold) |
| Microgrid clusters / PCC tie-lines | 5 / 23 |
| N-1 contingencies (intact grid) | 91, 0.00 MW shed |
| Blackout + N-1 shed (no upgrade → with DER) | 76.38 MW (0.98 %) → 0.00 MW |
| DER upgrade cost (Cluster 3 only) | $123,308,875 |
| Dirac-3 dispatch saving vs convex QP (mean) | 3.00 % (all 10 scenarios) |
| Dirac-3 optimality gap vs global (mean) | 2.10 % (convex QP: 5.26 %) |
| Dirac-3 feasibility | 10/10 feasible, 0.0 MW max balance error |
| Model A control (convex) Dirac-3 gap | 0.30 % |
| Dirac-3 encoding | 10 continuous modes, degree-3 polynomial, sum-constrained |

## 5. Discussion, Scope, and Limitations

The results delineate where the quantum method helps. On near-convex costs (Model A) Dirac-3 matches the classical optimum and offers no advantage. On non-convex cubic costs (Model B), the regime the challenge targets, the continuous polynomial encoding recovers dispatches about 3 % cheaper than a convex solver and within about 2 % of the global optimum across all ten scenarios, with exact power balance. The advantage scales with the degree of non-convexity.

The non-convex cost coefficients are calibrated from published valve-point benchmarks and applied to the dataset's real generator capacities, since the ARPA-E cost tables are near-convex. This is a modeling assumption; the resulting non-convexity is verified per unit. Dirac-3 is a stochastic sampler, so an individual run varies; reported figures are means over the ten-scenario sweep. Every figure is produced by the submitted code (`experiments/dispatch_benchmark.py` → `doc/stats_dispatch.json`; `phase3_final_submission.ipynb`) with no hardcoded values. Islanding, contingency, and siting use DC OPF; reactive power and voltage constraints are out of scope. The dispatch is formulated per microgrid as a single-bus economic dispatch.

The present work covers Stages 1–2 of the challenge (microgrid design, islanding, and dispatch) and a deterministic subset of Stage 3 (scenario evaluation over the ten dataset scenarios). The following extensions are scoped for future work and would be natural continuations of the same Dirac-3 formulation: a 24–72 hour multi-period horizon with battery state-of-charge dynamics and charge/discharge efficiency; two-stage stochastic optimization with here-and-now commitment and per-scenario recourse under renewable and load forecast uncertainty; generator unit commitment with start-up and shutdown costs, encoded with the Dirac-3 integer solver; and AC security-constrained OPF with voltage and reactive-power limits. The dispatch encoding is linear in generator count and Dirac-3's per-job runtime is roughly constant with problem size, so the approach is expected to extend to larger fleets and to the multi-period and unit-commitment formulations above.
