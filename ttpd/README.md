# `ttpd/` — solver implementations

All code that solves a TTP-D instance. The runners in
[experiments/](../experiments/) import from this package; nothing here depends
on them.

```
core/         problem definition: instance, simulator, masking, exact evaluator
heuristics/   simulated annealing, VNS, and the move library they share
exact/        Gurobi MILP formulation
rl/           three RL stacks (a280/gat, a280/mlp, ttd300/gat)
lisa/         behaviour cloning and the SA-repair hybrid
_internal/    support modules, not part of the method
```

## Import mechanism

The solver code uses flat imports — `from core import evaluate`,
`from env import TTPDEnv`, `from sa import sa` — rather than package-qualified
ones. This is the import style under which the code was written and validated,
and it is the reason `_internal/paths.py` exists: the module is the single place
that knows where each stack resides, and it places the required stacks on
`sys.path` so that a runner declares its dependencies in one statement rather
than traversing relative paths.

```python
from ttpd import _paths
_paths.use("core", "heuristics", "rl/a280/gat")
```

```bash
python3 -m ttpd --stacks     # enumerate the names accepted by use()
```

Rewriting the imports across three forked RL trees would have required
revalidating results that had already been produced and verified, and was
therefore not undertaken.

## Rationale for three RL stacks

The `rl/` directory holds three near-identical trees. These are deliberate
forks rather than redundancy:

- **`a280/gat` against `ttd300/gat`** differ in endurance conditioning: extra
  context scalars, a modified reward shaping, and `ED_frac_power` sampling. That
  conditioning is the subject of the ttd300 study.
- **`a280/gat` against `a280/mlp`** differ only in the encoder, and that
  difference constitutes the ablation reported in the paper.

Merging them would alter published results. By contrast, `core/` is a single
copy that replaced seven byte-identical instances; unifying it changed nothing.

## `_internal/`

Two support modules, neither of which affects any reported result.

`paths.py` implements the stack selection described above.

`hub.py` resolves model weights and benchmark data, downloading from the Hugging
Face Hub only when they are absent from disk. The mapping is a prefix
substitution in each direction — `artifacts/weights/` corresponds to
`Murjani/ttpd-weights` and `artifacts/data/` to `Murjani/ttpd-benchmarks` — so
the local and remote layouts cannot diverge.

```bash
python3 -m ttpd --plan      # display the local-to-Hub mapping
python3 -m ttpd --check     # verify the mapping contains no collisions
```

`ttpd.hub` is re-exported at the package level, so `from ttpd.hub import
ensure_local` is the form used at call sites.
