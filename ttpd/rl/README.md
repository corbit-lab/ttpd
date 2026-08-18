# `ttpd/rl/` — learned construction policies

An encoder–decoder policy that constructs a feasible TTP-D plan in a single
pass: the instance is embedded once, after which a composite action is emitted
at each node the truck reaches. Training is performed offline with Proximal
Policy Optimisation and a POMO group baseline, so solving a new instance
requires one forward pass per decision rather than a search trajectory.

```
a280/gat/     fixed-endurance attention policy      env/ policy/ train/
a280/mlp/     encoder ablation of the above         env/ policy/ train/
ttd300/gat/   endurance-conditioned policy          env/ policy/ train/
```

## Rationale for three stacks

These trees are near-identical and are deliberately not merged.

`a280/gat` and `ttd300/gat` differ in endurance conditioning. The ttd300 policy
reads the normalised endurance E_D/2d_max and the in-flight slack as additional
context scalars, applies a different reward shaping, and samples the endurance
fraction per episode. That conditioning is the subject of the ttd300 study.

`a280/gat` and `a280/mlp` differ in the encoder alone. Each graph-attention
sub-layer is replaced by a node-wise residual feed-forward block of identical
width, so that a node's embedding is computed solely from its own four input
features. Encoder depth, embedding dimension, feed-forward width, decoder,
action heads, masks, PPO procedure, POMO baseline, and training schedule are all
held fixed. That difference constitutes the ablation reported in the paper.

Sharing code between the stacks would alter published results. Each supplies its
own `env`, `policy`, and `train` modules, which is why they collide on module
name and why `ttpd/_internal/paths.py` places exactly one on `sys.path` at a
time.

## Markov decision process

**State.** The truck's current node, elapsed mission time, load on arrival, the
set of visited customers, and the drone's status — idle, or in flight from ℓ
toward k with a recorded launch time. Because the flight tuple fully determines
the drone's arrival at any future rendezvous, the Markov property is preserved.

**Composite action**, decoded autoregressively in the following order:

| Head | Decision |
|---|---|
| ρ | Whether the in-flight drone rejoins the truck at the current node |
| z^D | Whether to collect the item delivered by the landing drone |
| z^T | Whether to collect the item at the current node |
| k | Drone launch target, or ∅ to remain idle |
| j | Next node for the truck |

The launch decision deliberately precedes the truck move, making the latter
conditional on the former. Reversing the order would preclude launching the
drone to the final customer while the truck proceeds directly to the depot to
await its return.

**Reward.** Profit collected at the epoch less the rental cost of elapsed time.
Without discounting the rental terms telescope, so the total episode return
equals the TTP-D objective exactly; no reconciliation between a shaped reward
and the objective is required.

**Feasibility.** Each head is decoded under a mask, and the masks are
bidirectional: every admissible composite action transitions to a state with at
least one feasible completion, eliminating backtracking and post-hoc repair.
Conversely, every feasible MILP solution maps to a valid action sequence, which
is how the simulator was validated — by replaying MILP optima through it.

## `policy/`

`encoder.py` embeds the N+2 nodes (source, N customers, sink) from four features
each — coordinates normalised by the instance bounding box, profit normalised by
the instance maximum, and weight normalised by truck capacity — through a stack
of pre-norm multi-head self-attention blocks at d_model = 128.

`decoder.py` forms the per-step context by concatenating the mean node
embedding, the current node's embedding, the mean embedding of unvisited
customers, and seven state features (load fraction, normalised elapsed time,
fraction of customers remaining, fraction of profit collected, an in-flight
indicator, and two normalised drone timing features). The two node-selection
heads are pointer heads with an eight-head attention glimpse and C·tanh clipping
at C = 10; the three binary heads are two-layer GELU perceptrons. The j head
receives the chosen launch target's embedding, enforcing the decoding order
described above.

`beam.py` implements inference: beam width 256 over all eight dihedral views,
retaining one beam per first-launch option, with the best completed solution
selected by objective. Every reported plan is replayed through the simulator and
verified feasible before scoring. `eas.py` implements efficient active search,
part of the decoding-strategy ablation rather than the headline results.

## `train/`

PPO with a POMO group baseline in place of a learned critic: P = 32 rollouts per
instance, with advantages centred on the group mean and scaled by a per-instance
stationary factor so that reward magnitudes are comparable across sizes.

`ppo.py` implements the clipped surrogate, with the entropy coefficient annealed
on a cosine schedule and an update epoch terminating early when the mean
approximate KL divergence exceeds its target. `rollout.py` advances the group in
lockstep through a single batched forward pass per decoding step. Starts are
stratified jointly over the first truck move and the first drone launch, assigned
by decoupled strides to avoid a rigid product ordering, with each rollout subject
to one of eight dihedral transformations. Pinned initial decisions are excluded
from both behaviour and gradient-time log-probabilities, so the probability ratio
is computed exclusively over policy-selected actions.

`bench_tuned.py` fine-tunes on the five benchmark instances of a given size,
warm-started from `sampled/n20`, since training from scratch does not converge
within the budget beyond N = 50.

## Role in the study

The policies' advantage is inference cost rather than solution quality: a
construction pass costs a forward pass per decision, independent of any search
budget, while the gap to a full metaheuristic run widens with instance size.
Their principal contribution here is as the warm-start generator within
[LISA](../lisa/), where a beam plan followed by a short annealing repair is
stronger than either component alone.

Runners: [experiments/a280/gat/](../../experiments/a280/gat/),
[experiments/a280/mlp/](../../experiments/a280/mlp/),
[experiments/ttd300/gat/](../../experiments/ttd300/gat/).
