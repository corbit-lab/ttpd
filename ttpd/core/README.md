# `ttpd/core/` — problem definition

A single definition of the TTP-D: the structure of an instance, the cost of a
plan, and the admissibility of an action. Every solver in this repository — the
MILP, both metaheuristics, the learned policies, and LISA — scores its output
through this code, which is what renders the reported numbers comparable across
methods.

This directory replaced seven byte-identical copies present in the earlier
layout. Unifying them changed no result.

| Module | Contents |
|---|---|
| `instance.py` | `TTPDInstance`: coordinates, profits, weights, capacity, speeds, endurance. Loads a280 files and samples subsets. |
| `simulator.py` | `TTPDEnv`, the discrete-event simulator. Executes a composite action, advances the mission clock, and transfers drone payload at rendezvous. |
| `masking.py` | Action admissibility. The masks are bidirectional: every admissible action leads to a state with at least one feasible completion, so neither backtracking nor post-hoc repair is required. |
| `core.py` | The evaluator, the nearest-neighbour construction, the move library, and the VND procedure underlying both metaheuristics. |
| `constants.py` | Benchmark parameters: velocity bounds, the a280 capacity of 637,010 and its per-size scaling, R = 72.70, and the drone speed factor of 2.0. |
| `ttpd_common.py` | Converts a (route, sorties, packing) triple into the MDP action sequence and replays it. This underpins both the construction of behaviour-cloning data from SA solutions and the validation of the simulator against MILP optima. |

## Objective

    G = Σ_i p_i z_i − R · τ_sink

Collected profit less the rental rate times the makespan. Two consequences are
worth stating explicitly.

First, G is negative on every a280 benchmark instance. At R = 72.70 the rental
charge exceeds the maximum profit the fleet can collect at these sizes, so
higher (less negative) values are preferable and performance is driven by
mission time rather than by collected profit.

Second, truck velocity is affine in load, v(W) = v_max − W(v_max − v_min)/W_cap,
and is evaluated at the departing weight of each arc rather than the arrival
weight. An early heavy pickup therefore penalises every subsequent arc.

## Feasibility

The masks constitute the contract between the solvers. A plan is admissible when
every customer is visited exactly once, each sortie comprises exactly two arcs
(outbound to a single target, inbound to a later rendezvous node), capacity holds
at every step, and — on ttd300 — the sortie round trip lies within the endurance
budget. The metaheuristic evaluator prunes violations; the policy's masks render
them undrawable. The same rule is enforced by both mechanisms.
