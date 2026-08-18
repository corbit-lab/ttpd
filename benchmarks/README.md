# `benchmarks/` — instance generators

Two deterministic generators. Both are seeded, so re-execution reproduces the
shipped instances exactly, subject to the documented exception below.

## `generate_bench.py` — the a280 family

Samples customers from the TSPLIB `a280` TTP instance. The depot is fixed at
node 1, each of the 279 remaining cities contributes its single highest-profit
item, and five instances are drawn per size for
N ∈ {5, 10, 15, 20, 30, 40, 50, 100}.

Capacity scales with size at the a280 per-city rate, W = 637,010·N/280. The
renting ratio R = 72.70 exceeds the collectable profit at these sizes, so the
objective is negative on every instance and performance is governed by mission
time. Drone speed is 2·v_max and the payload limit equals the heaviest item in
the sampled instance, rendering every item drone-eligible. Sortie length is
unbounded.

## `generate_ttd300.py` — the endurance family

The benchmark introduced in this paper. Uniform integer coordinates on [0,300]²
under `CEIL_2D` distances, N ∈ {10, 20, 30, 40, 50, 75, 100}, with five layouts
each. Five items per customer with w ~ U[1000,1009] and p ~ U[1,1000], the a280
"uncorrelated, similar weights" class. Capacity retains the a280 per-city rate
at W = round(2275.0357·N), fixed per size and shared across layouts, with
R = 50.

Each layout is instantiated at four endurance fractions,
E_D = round(f · d_max) for f ∈ {0.25, 0.5, 0.75, 1.0}, where d_max is the
instance's own maximum pairwise distance, yielding 7 × 5 × 4 = 140 instances.
Filenames carry the fraction (`_f025`, `_f050`, `_f075`, `_f100`) and the
absolute endurance appears in the header. The bare `ttd300_n<N>_L<L>.txt` file,
which has no endurance header, is the unbounded reference.

The endurance limit is specified relatively because an absolute budget carries a
different physical meaning at different sizes under non-uniform geometry, and
the a280 layout is strongly non-uniform. A fixed-size uniform domain combined
with a fraction of d_max gives the endurance radius a consistent interpretation
across instances, which is what makes it a clean axis to sweep.

`README_ttd300_data.txt` is the machine-readable specification shipped alongside
the data.

## Two caveats before regenerating

**Output path.** Both generators write to `artifacts/bench/instances/`, whereas
the shipped instances read by every runner reside in
`artifacts/data/instances/`. That constant is stale: regeneration creates a new
tree alongside the canonical one rather than overwriting it. This is harmless,
but a regenerate-and-diff check will not work until both constants are
redirected to `artifacts/data/instances/`.

**Thirteen a280 files will not round-trip.** `bench_<n>_<k>.txt` contains the
instance whose internal `PROBLEM NAME` is `bench_<n>_<k+1>`, affecting
`bench_10_{2,3,4}`, `bench_20_{3,4}`, `bench_30_{1,2,3,4}` and
`bench_40_{1,2,3,4}`. The mislabelling predates the repository reorganisation
and invalidates no comparison, since every solver reads the same file for a
given label and all methods therefore solved identical instances, but
regeneration will not reproduce those files under their current names.
