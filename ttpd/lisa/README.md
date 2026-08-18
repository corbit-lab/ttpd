# `ttpd/lisa/` — Learner-Initialised Simulated Annealing

The hybrid solver. Simulated annealing attains high solution quality but incurs
the full search cost on every instance, whereas the learned policy is fast but
degrades with scale. LISA employs the annealer as the trainer, distils it into a
construction policy by behaviour cloning, and returns the policy's plan to a
truncated annealing run for repair.

The single parameter β ∈ (0,1] is the fraction of the full annealing budget
allocated to repair. As β → 0 the method reduces to the cloned policy; at β = 1
it recovers the full-budget annealer.

```
a280/     the fixed-endurance pipeline
ttd300/   the endurance-conditioned pipeline (with ttd300_io.py, run_lisa.py)
```

## Three stages

**Distillation.** The full-budget annealer is run on a corpus of training
instances. `sa_to_actions.py` inverts each expert solution into the composite
action sequence of the MDP, and every sequence is replayed through the simulator
and verified to reproduce the expert objective before being retained. Only
certified state–action pairs enter the dataset. `gen_dataset.py` drives this
stage and writes JSONL.

**Cloning.** `bc_train.py` trains the attention policy by supervised maximum
likelihood over the five factored action heads, under the same feasibility masks
used during reinforcement learning. A distinct checkpoint is trained per
instance size.

**Repair.** `hybrid_solve.py` decodes a beam of candidate plans (width 128 over
eight dihedral views), selects the best feasible plan, and warm-starts an
annealing run under a budget of β·B(N).

## Budget profile

The gap profile against β is steep at the low-budget end and then flattens: most
of the improvement over the unrepaired beam plan is recovered within the first
few percent of the annealing budget, after which each further increment
contributes progressively less. The warm start contributes materially, a cold
annealing run at the same reduced budget trailing LISA at every budget fraction.
The value concentrates at small and intermediate sizes; the largest instances
continue to require the full search.

## Datasets

Behaviour-cloning datasets reside under
`artifacts/data/behaviour_cloning_datasets/` and on the Hub in
`Murjani/ttpd-benchmarks`. They are named for their coverage:
`sampled_n<N>.jsonl` is drawn from randomly sampled instances and
`benchmark_n5_to_n100.jsonl` from the benchmark instances themselves. They are
regenerable at any time via `gen_dataset.py`, which is why they are not
versioned in git.

Runners: [experiments/a280/lisa/](../../experiments/a280/lisa/) and
[experiments/ttd300/lisa/](../../experiments/ttd300/lisa/).
