# `artifacts/` — instances, results, and model weights

```
data/instances/                    benchmark instances      — versioned
data/results/                      solver result tables     — versioned
data/behaviour_cloning_datasets/   expert demonstrations    — Hub (24 MB)
weights/                           model checkpoints        — Hub (499 MB)
```

Instances and result tables are versioned in the repository, so a clone can run
the solvers and verify every reported number without network access. The two
large trees remain on the Hugging Face Hub and are both regenerable —
checkpoints by retraining, behaviour-cloning data by
`ttpd/lisa/*/gen_dataset.py`.

## Correspondence with the Hub

This directory mirrors the two Hub repositories exactly, so the local and remote
layouts cannot diverge:

| Local path | Hub repository | Kind |
|---|---|---|
| `artifacts/weights/<p>` | [`Murjani/ttpd-weights`](https://huggingface.co/Murjani/ttpd-weights) | model |
| `artifacts/data/<p>` | [`Murjani/ttpd-benchmarks`](https://huggingface.co/Murjani/ttpd-benchmarks) | dataset |

The mapping is a prefix substitution in each direction and is implemented in
`ttpd/_internal/hub.py`. Both repositories may be overridden through
`$TTPD_WEIGHTS_REPO` and `$TTPD_BENCH_REPO`.

No prefetching is required. Call sites wrap their path in `ensure_local()`,
which returns an existing local file unchanged and downloads otherwise.

```bash
huggingface-cli login             # both repositories are private
python3 -m ttpd --plan            # display the local-to-Hub mapping
python3 -m ttpd --check           # verify the mapping contains no collisions
TTPD_OFFLINE=1 python3 ...        # raise an error rather than downloading
```

## `data/instances/`

`a280/` contains `bench_<n>_<i>.txt`, five per size, together with the source
`a280_benchmark.txt`. `ttd300/` contains `ttd300_n<N>_L<L>[_f<f>].txt`: 140
endurance-bounded instances and their unbounded references. Both are
reproducible from [benchmarks/](../benchmarks/), subject to the thirteen
mislabelled a280 files documented there.

## `data/results/`

One subtree per benchmark family and method, mirroring the runner that produced
it. The per-size logs under `*/training_logs/` are the PPO training curves, and
`summary/` under ttd300 holds the aggregated tables.

## `weights/`

```
weights/<a280|ttd300>/<gat|mlp|lisa>/<family>/n<N>/best.pt
```

Checkpoint families are named for the distribution the policy was trained on:
`benchmark-tuned/` on the five benchmark instances of that size, supporting the
headline results; `sampled/` on randomly sampled subsets, used for
generalisation and as the warm start for the former; and `specialists/`, early
small-N runs. Within a run, `best.pt` is the deliverable, `last.pt` supports
resumption, and `best_milp.pt` is selected by MILP gap rather than evaluation
return.

The deliverable checkpoint per model and size is tabulated in the
[root README](../README.md#checkpoint-selection). Earlier logs refer to these
directories as `beast`/`beast2` (→ `benchmark-tuned`), `new` (→ `sampled`) and
`final` (→ `specialists`).
