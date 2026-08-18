# `experiments/a280/` — the a280 study

The first of the two studies. Customers are sampled from the TSPLIB `a280` TTP
instance, so the geometry and item economics are those of the classical
configuration, and drone endurance is unbounded — a sortie may extend
arbitrarily far between launch and rendezvous. All methods are compared here,
with the exact solver providing the reference wherever it can be applied.

Sizes N ∈ {5, 10, 15, 20, 30, 40, 50}, with several runners also carrying
N = 100; five instances per size and ten seeds per stochastic method.

```
milp/              the exact solver
sa/  vns/          the metaheuristics, with sa/sensitivity_analysis/
gat/  mlp/         the learned policies; mlp is the encoder ablation
lisa/              the budget sweep of the hybrid
benchmark_sweep/   the combined driver across methods and sizes
```

## Relation to the ttd300 study

This study selects the method set carried forward. Simulated annealing supplies
the best known solution beyond the sizes the exact solver can reach; VNS
deteriorates as N grows; and the GAT encoder is compared against the MLP
ablation at identical inference cost, the two sharing a decoder, masking scheme,
and training pipeline and decoding at the same rate. The strongest
representative of each family — SA, GAT, and LISA — proceeds to ttd300.

## Structure of a method directory

The `gat/` and `mlp/` directories mirror one another file for file.

| Script | Function |
|---|---|
| `train.py` | PPO training, one policy per size |
| `train_bench_tuned.py` | Fine-tuning on the five benchmark instances, warm-started from `sampled/n20` |
| `eval_bench.py` | The headline evaluation: beam width 256 over eight views, writing the gap table |
| `eval.py` | Evaluation on sampled instances rather than the benchmark set |
| `generalize.py` | Decoding an N = 20 policy at other sizes, for the zero-shot rows |
| `ablation.py`, `ablation_rerun.py` | The comparison of six decoding strategies |
| `replay_milp.py` | Replay of MILP optima through the simulator; the simulator validation |
| `solution_csv.py`, `sample_bench.py` | Per-plan dumps used by the figures |

## Notes on individual directories

**`milp/`** — `run_gurobi15.py` and `run_gurobi20.py` are the per-size drivers
and `run_size.py` the generic one. Gurobi and a valid licence are required. The
larger sizes run against a 24-hour limit per instance and should be budgeted
accordingly; `--seconds` supports smoke tests. Seed 1 ships for reference and is
skipped by default.

**`sa/` and `vns/`** — `run_sa.py` and `run_vns.py` are the sweeps, with
`run_parallel.py` sharding them across vCPUs (`--dry-run` first).
`sa/sensitivity_analysis/` contains the a280 one-at-a-time sweeps over
endurance, multi-item collection, drone speed, and general parameters, with
results versioned under `results/quick/` and `results/full/`.

**`lisa/`** — `gen_dataset_hybrid.py` constructs the behaviour-cloning corpus
from annealing experts, and `run_sa_lisa_pareto.py` sweeps
β ∈ {5, 10, 20, 25, 33.3, 50}%.

## Instance labelling

Thirteen files are labelled one seed offset: `bench_<n>_<k>.txt` contains the
instance whose internal `PROBLEM NAME` is `bench_<n>_<k+1>`, affecting
`bench_10_{2,3,4}`, `bench_20_{3,4}`, `bench_30_{1,2,3,4}` and
`bench_40_{1,2,3,4}`. Every solver reads the same file for a given label, so no
comparison is affected; the issue concerns provenance only. Regenerating from
[benchmarks/generate_bench.py](../../benchmarks/generate_bench.py) will not
reproduce those thirteen files under their current names.
