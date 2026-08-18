# `experiments/` — reproduction of reported results

One runner per reported table. These scripts load instances, invoke a solver
from [ttpd/](../ttpd/), and write a CSV; no algorithm is implemented here.

```
a280/     benchmark_sweep/ gat/ milp/ mlp/ sa/ vns/ lisa/
ttd300/   benchmark_sweep/ gat/ lisa/ pareto_frontier/ sa/ sensitivity_analysis/
```

The split by benchmark family reflects the two studies. The a280 study compares
every method against an exact reference wherever one can be constructed; the
ttd300 study carries forward the strongest representative of each family (SA,
GAT, LISA) and varies drone endurance.

## Invocation

Every entry point resolves the repository root independently, so neither
installation nor `PYTHONPATH` configuration is required and the working
directory is immaterial.

```bash
python3 experiments/a280/sa/run_sa.py --sizes 10 --instances 1 --seeds 1
python3 experiments/ttd300/gat/eval_bench.py --help
```

Most runners default to a full sweep representing hours to days of computation.
The `--sizes`, `--instances`, `--seeds`, `--limit` and `--budget` flags reduce
this and should be used first.

## Experimental protocol

**Seeds.** Every stochastic method (SA, VNS, GAT, MLP, LISA) is run over ten
independent seeds per instance on a280 and five on ttd300, and every reported
value is the mean across those runs. The MILP is deterministic and solved once
per instance.

**Runtime budgets** are specified per size and shared across methods so that
comparisons are conducted at equal time:

| Family | Budget (s) |
|---|---|
| a280 | 5:10, 10:30, 15:120, 20:300, 30:450, 40:600, 50:750 |
| ttd300 | 10:120, 20:300, 30:450, 40:600, 50:750, 75:975, 100:1200 |

These appear as `TIME_BUDGET` or `BUDGET` in the runners. Because a substantial
part of the reported comparison is gap-at-equal-time, a change in one location
must be propagated to the others.

**Scoring.** Every candidate plan produced by every method is re-evaluated
through the exact evaluator in [ttpd/core/](../ttpd/core/) before entering a
table. A solver's self-reported objective is never taken at face value; this is
precisely how the MILP's piecewise-linear surrogate would otherwise propagate
into the results.

**Reference solutions.** On a280, gaps are measured against the best known
solution across all methods and seeds. On ttd300 no MILP can be constructed at
these scales, so the reference is the best validated full-budget annealing
solution per instance, which is why the SA gap is 0.00% there by construction.

**Output** is written to `artifacts/data/results/<variant>/<method>/`, mirroring
the directory of the runner that produced it.

## Resumption and parallelism

The long sweeps (`ttd300/benchmark_sweep/run_bench.py` and
`ttd300/sensitivity_analysis/run_sens.py`) append to their CSV and skip cells
already present, so an interrupted run is resumed by reissuing the same command.
The `run_parallel.py` drivers under `a280/sa/` and `a280/vns/` shard the job
across vCPUs; `--dry-run` prints the shard plan without launching anything.

## Compute environment

Reported results were obtained on an Intel Xeon Platinum 8581C (8 vCPUs, 62 GB
RAM) under Debian GNU/Linux 12, with Gurobi 13 subject to a 24-hour limit and a
0.01% relative gap target. Policy training was performed separately on a single
NVIDIA RTX 4000 Ada; all reported measurements, including policy inference, were
taken on the CPU host. Runtimes will not transfer across hardware; gaps will.
