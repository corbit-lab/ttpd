# `experiments/ttd300/` — the endurance study

The second study, on the benchmark introduced in this paper. Uniform coordinates
on a 300×300 box, seven sizes with five layouts each, and drone endurance as the
sole controlled variable: E_D = round(f · d_max) for f ∈ {0.25, 0.5, 0.75, 1.0},
giving 140 instances.

Only the strongest representative of each family from the a280 study is carried
forward: SA, GAT, and LISA. No MILP can be constructed at these scales, so the
reference is the best validated full-budget annealing solution per instance, and
the SA gap is 0.00% by construction.

```
sa/                    reference runs
gat/                   training and evaluation of the endurance-conditioned policy
lisa/                  the budget-fraction sweep
pareto_frontier/       the quality-against-budget frontier
benchmark_sweep/       the full 140-instance driver
sensitivity_analysis/  multi-item, drone speed, and one-at-a-time parameter sweeps
```

## The endurance axis

The endurance fraction acts as a challenge parameter for the learned policy
rather than a difficulty relaxation. A looser radius admits more launch options
and longer sorties, giving a constructive pass more opportunities to commit an
error it cannot subsequently repair; at the largest size the ordering inverts,
since a hundred cities in the same box means even the tightest radius admits
many short sorties and the difficulty shifts to selecting among them.

The axis also bounds what drone speed can buy. Where the radius admits no
feasible sortie at all, the drone-speed sweep is flat: an arbitrarily fast drone
is worthless without range.

## Notes on individual directories

**`sa/`** — `run_sa_ttd300.py` produces the reference solutions;
`exp_ed_normalizer.py` is the calibration check on the endurance normalisation
used by the policy.

**`gat/`** — `train.py` trains per size on the ttd300 distribution with the
endurance fraction drawn per episode; `train_bench_tuned.py` fine-tunes; and
`eval_bench.py` decodes at beam width 256 over eight views, accepting `--fracs`
and `--layouts` to slice the grid.

**`lisa/` and `pareto_frontier/`** — the budget sweep over the L1 layout of every
size at all four endurance fractions, giving 28 cells per β. Each cell is an
independent stochastic run, so individual cells are not always monotone in β even
though the row means are.

**`sensitivity_analysis/`** — `run_sens.py` is the driver, accepting
`--exps multiitem,speed,sens`, and sweeps the renting ratio, capacity, minimum
truck speed, drone speed, and the single- against multi-item collection
protocol. Every configuration is solved at all four endurance fractions, so each
experiment also shows how its conclusions shift as the sortie radius tightens.

**`benchmark_sweep/`** — `run_bench.py` executes the full 140-instance grid. It
appends and skips completed cells, so an interrupted run resumes on reissue, and
`--summary-only` reports on an existing CSV without recomputation. The full
sweep runs five seeds per instance at the per-size budgets and represents
several days of CPU time; `--limit` and `--budget` support smoke tests.
