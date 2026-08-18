<h1 align="center">Fly, Pack, Drive</h1>
<p align="center"><em>The Travelling Thief Problem with Drone</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.16435"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2608.16435-B31B1B?logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/Murjani/ttpd-weights"><img alt="weights" src="https://img.shields.io/badge/weights-Murjani%2Fttpd--weights-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/datasets/Murjani/ttpd-benchmarks"><img alt="benchmarks" src="https://img.shields.io/badge/data-Murjani%2Fttpd--benchmarks-FFD21E?logo=huggingface&logoColor=black"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="licence" src="https://img.shields.io/badge/licence-MIT-555">
</p>

---

Reference implementation and experimental code for the Travelling Thief Problem
with Drone (TTP-D).

A capacitated truck and a single-package drone operate from a common depot on a
collection route. The truck's velocity decreases affinely with its accumulated
load, so an early pickup penalises every subsequent arc. The drone launches from
a node at which the truck is present, retrieves one item at an outlying
customer, and rejoins the truck at a later rendezvous node; whichever vehicle
arrives first waits, while the rental clock continues to run. The objective is
to maximise collected profit net of a rental cost proportional to the makespan,

    G = Σ_i p_i z_i − R · τ_sink

The problem is NP-hard, generalising both the TSP and the 0-1 knapsack problem,
and the subproblems are strongly coupled: collecting an additional item shifts
all subsequent arrival times and may invalidate a rendezvous scheduled further
downstream.

## Solution methods

| Method | Description |
|---|---|
| MILP | Exact Gurobi formulation with SOS2 piecewise-linear velocity |
| SA | Simulated annealing over the joint route–pack–sortie space |
| VNS | Variable neighbourhood search on the same move library |
| GAT / MLP | Attention construction policy, PPO with a POMO baseline; the MLP is the encoder ablation |
| LISA | Behaviour cloning of the annealer, followed by a truncated SA repair |

The exact solver anchors the comparison where it is tractable, the
metaheuristics search each instance independently, the learned policies amortise
that search into offline training, and LISA interpolates between the last two
under a single budget parameter β. Reported results are in the accompanying
paper.

## Installation

```bash
pip install -r requirements.txt

python3 experiments/a280/sa/run_sa.py --sizes 10 --instances 1 --seeds 1
```

Benchmark instances ship with the repository and every entry point resolves the
repository root independently, so neither installation nor `PYTHONPATH`
configuration is required. Optionally, `pip install 'gurobipy>=10.0'` for the
MILP baselines (a licence is required), and `pip install -e .` to import `ttpd`
from external code.

## Repository structure

```
ttpd/           solver implementations
  core/           instance model, simulator, action masking, exact evaluator
  heuristics/     simulated annealing, VNS, shared move library
  exact/          Gurobi MILP formulation
  rl/             three RL stacks — a280/gat, a280/mlp, ttd300/gat
  lisa/           behaviour cloning and the SA-repair hybrid
experiments/    runners reproducing the reported results
benchmarks/     instance generators for both families
artifacts/      instances and results (versioned); weights and data (Hub)
```

Each directory carries its own README documenting its contents and design
rationale. See [ttpd/](ttpd/) for the algorithms and
[experiments/](experiments/) to reproduce a specific table.

## Benchmark families

**a280.** Customers sampled from the TSPLIB `a280` TTP instance, preserving the
spatial structure and item economics of the classical configuration. Sizes
N ∈ {5, …, 50, 100}, five instances each, with unbounded drone endurance and a
renting ratio of R = 72.70.

**ttd300.** Introduced in this paper. Uniform integer coordinates on a [0,300]²
box under `CEIL_2D` distances, seven sizes with five layouts each, and drone
endurance as the sole controlled variable: E_D = round(f · d_max) for
f ∈ {0.25, 0.5, 0.75, 1.0}, yielding 140 instances. The second family exists
because the `a280` layout is strongly non-uniform, so an absolute endurance
budget carries a different physical meaning at different sizes, whereas a
uniform fixed-size domain gives the endurance radius a consistent
interpretation. Both generators are deterministic and reside in
[benchmarks/](benchmarks/).

## Model weights and datasets

Instances and result tables are versioned here, so a clone reproduces and
verifies results without network access. Two artifacts remain on the Hugging
Face Hub under [Murjani](https://huggingface.co/Murjani):
[`ttpd-weights`](https://huggingface.co/Murjani/ttpd-weights) (checkpoints,
499 MB) and [`ttpd-benchmarks`](https://huggingface.co/Murjani/ttpd-benchmarks)
(behaviour-cloning datasets, 24 MB). Both are regenerable. The `artifacts/` tree
mirrors the two repositories exactly, and call sites invoke `ensure_local()`,
which returns an existing local file unchanged and downloads otherwise.

Overridable through `$TTPD_WEIGHTS_REPO` and `$TTPD_BENCH_REPO`; details in
[artifacts/](artifacts/).

## Checkpoint selection

```
artifacts/weights/<a280|ttd300>/<gat|mlp|lisa>/<family>/n<N>/best.pt
```

Families are named for the distribution the policy was trained on, which
determines their applicability: `benchmark-tuned/` on the five benchmark
instances of that size (the headline results), `sampled/` on randomly sampled
a280 subsets (generalisation, and the warm start for the former), and
`specialists/` for the early small-N runs used in the N ≤ 20 rows. The
`benchmark-tuned` policies warm-start from `sampled/n20`, as training from
scratch does not converge within the budget beyond N = 50. Within a run,
`best.pt` is the deliverable, `last.pt` supports resumption, and `best_milp.pt`
is selected by MILP gap rather than evaluation return.

| Model and size | Checkpoint |
|---|---|
| a280 GAT, N = 5, 10 | `a280/gat/specialists/n5.pt`, `n10.pt` |
| a280 GAT, N = 15, 20 | `a280/gat/sampled/n<N>/best.pt` |
| a280 GAT, N = 30–100 | `a280/gat/benchmark-tuned/n<N>/best.pt` |
| a280 MLP, N = 20 | `a280/mlp/sampled/n20/best.pt` |
| a280 MLP, remaining sizes | `a280/mlp/benchmark-tuned/n<N>/best.pt` |
| ttd300 GAT | `ttd300/gat/benchmark-tuned/n<N>/best.pt` |
| LISA | `<variant>/lisa/behaviour_cloning/n<N>.pt` |

Checkpoints are `torch.save` payloads keyed under `"policy"`:

```python
import torch
from ttpd.hub import weights_dir, ensure_local

p = ensure_local(f"{weights_dir('a280','gat','benchmark-tuned')}/n30/best.pt")
policy.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["policy"])
```

## Notes on the data and the model

**Instance labelling in the a280 family.** Thirteen per-seed instances are
labelled one seed offset: `bench_<n>_<k>.txt` contains the instance whose
internal `PROBLEM NAME` is `bench_<n>_<k+1>`, affecting `bench_10_{2,3,4}`,
`bench_20_{3,4}`, `bench_30_{1,2,3,4}` and `bench_40_{1,2,3,4}`. No comparison
is invalidated, as every solver reads the same file for a given label; the issue
concerns provenance only, and implies these files will not regenerate under
their current names.

**Surrogate objective in the MILP.** The velocity curve is linearised over
K = 10 breakpoints, so the MILP optimises an approximation whose optimum bounds
the exact optimum from below. Every candidate plan, from every method, is
therefore re-evaluated under the exact load–speed law before being reported.

## Requirements and citation

Python 3.9 or later. `requirements.txt` covers all dependencies except Gurobi,
required only by the MILP experiments; all other solvers run without it.

If you use this code or the ttd300 benchmark, please cite the accompanying
paper. Trained checkpoints: <https://huggingface.co/Murjani>.

```bibtex
@inproceedings{murjani2026ttpd,
      title={Drive, Pack, Fly: The Travelling Thief Problem with Drone}, 
      author={Murjani, Kabir and Sobhanan, Abhay},
      year={2026},
      eprint={2608.16435},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.16435}, 
}
```