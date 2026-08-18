# `ttpd/heuristics/` — simulated annealing and variable neighbourhood search

Two metaheuristics that search the joint route–pack–sortie space directly. They
share the nearest-neighbour warm start, the move library, the
feasibility-preserving evaluator (all defined in [../core/core.py](../core/core.py)),
and the per-size runtime budget; they differ only in how they traverse the space.

Simulated annealing serves as the quality reference throughout the study: beyond
the sizes the exact solver can reach it supplies the best known solution on the
a280 family and defines the per-instance reference on ttd300. VNS is competitive
at small sizes and deteriorates as N grows, a single shake-and-descent trajectory
covering progressively less of the space within the same budget.

## `sa.py`

Acceptance follows the Metropolis criterion: improving moves are always
accepted, worsening moves with probability exp(ΔG/T).

The initial temperature is calibrated per instance. Worsening moves are sampled
from the warm start and T₀ is set to −mean|ΔG⁻| / ln 0.8, so that approximately
80% of worsening moves are accepted at the outset; no per-benchmark temperature
tuning is involved. Each temperature level proposes max(20, 12(N+1)) random
feasible neighbours, after which the temperature cools geometrically with
α = 0.97. When T falls below 10⁻⁴·T₀ the schedule reheats to T₀ and resumes from
the incumbent best, and a final VND descent polishes the result.

Simulated annealing samples candidate moves globally, reserving VND for the warm
start and the final polish.

## `vns.py`

The incumbent is shaken by η random moves, VND descends to a local optimum, and
the result is accepted only on improvement. On success the search recentres and
resets η = 1; otherwise η is incremented, cycling back to 1 at η_max = 8. A
General VNS variant restarting after stagnation was evaluated and produced
identical results, so the standard form is retained for simplicity.

## Shared components

**Candidate sampling.** Beyond N = 20, each VND pass evaluates at most 30N
randomly sampled candidates per operator, since the neighbourhoods are
O(N²)–O(N³) and exhaustive enumeration is not affordable.

**Move library.** Ten operators, defined in `core.py`: a collection flip; three
route operators (truck-customer swaps, 2-opt reversals, and Or-opt relocations
of segments of length one to three); two sortie operators preserving the truck
route (exact re-scheduling of launch and rejoin anchors by dynamic programming
over subsets of the drone targets, and re-anchoring of a single sortie over all
admissible launch–rejoin pairs); and four transfer operators moving customers
between the truck and the drone. Where a route operator inverts a sortie's
launch and rejoin nodes, an anchor repair restores their order.

**Runtime budgets** are specified per size and shared between the two methods so
that the comparison is conducted at equal time: {5:10, 10:30, 15:120, 20:300,
30:450, 40:600, 50:750} seconds on a280, and {10:120, 20:300, 30:450, 40:600,
50:750, 75:975, 100:1200} on ttd300. The runners in
[experiments/](../../experiments/) set these values; they should not be altered
in one place alone, as a substantial part of the reported comparison is
gap-at-equal-time.
