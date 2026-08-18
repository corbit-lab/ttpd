# `ttpd/exact/` — Gurobi MILP formulation

`bench_solver.py` constructs and solves the exact model. It requires
`gurobipy>=10.0` and a valid licence; no other component of the repository does.

## Tractability

Optimality is certified on every instance only at the smallest sizes. At N = 15
some instances close and the remainder exhaust the 24-hour limit; at N = 20 all
five terminate with an open incumbent; beyond N = 20 the model no longer fits in
memory. Five additional customers multiply the visit-mode assignments almost
eight-thousandfold and the joint route-and-assignment space by some ten orders of
magnitude. Past that point simulated annealing serves as the quality reference.

## Formulation

A flow formulation with three visit modes per customer (truck-only, drone-only,
and rendezvous), with the following components.

**SOS2 piecewise linearisation** of the reciprocal truck velocity ψ(W) = 1/v(W)
over K = 10 breakpoints. Since ψ is convex, the chord interpolant bounds it from
above, so the MILP optimum bounds the exact optimum from below, with the
resulting objective error bounded in closed form.

**McCormick envelopes** linearising the bilinear term z_k · x^D_kj, which tracks
the weight transferred from the drone to the truck at the rendezvous node.

**Three valid inequalities** bounding the makespan from below: aggregate truck
distance divided by v_max, aggregate drone distance divided by v_D, and a
mode-aware round-trip minimum for each customer. These substantially tighten the
root LP bound and are what closes the largest certifiable size at all.

**Tight disjunctive constants**: M_W = w_tot, M_T = d_max(N+1)/v_min, and
M_D = d_NN/v_min, where d_NN is the length of the nearest-neighbour warm-start
tour.

Each solve is warm-started from a greedy nearest-neighbour truck tour with a
knapsack-feasible packing, under `MIPFocus=3`, `Cuts=2`, `Presolve=2`,
`Heuristics=0.2`, and a 0.01% relative gap target.

## Surrogate objective

The MILP optimises the piecewise-linear surrogate, whereas all reported
objectives are evaluated under the exact load–speed law. The K = 10 chord is
conservative, so the surrogate optimum bounds the exact one from below, and
every reported number — the MILP's included — is re-scored through
[../core/](../core/). A search method may therefore report a stronger objective
than the exact solver at a size where the latter still certifies its own
surrogate.

Runners are located in [experiments/a280/milp/](../../experiments/a280/milp/).
