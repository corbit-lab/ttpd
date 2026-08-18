
from dataclasses import dataclass
import numpy as np
import torch

from env import TTPDEnv
from env.constants import ED_FRACS_BENCH
from env.masking import NO_LAUNCH
from policy.attention_policy import AttentionPolicy


@dataclass
class EvalResult:
    n_instances: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_steps: float
    mean_tau: float
    wall_seconds: float
    # mean return per problem size n -> {n: mean_return}
    per_size: dict | None = None
    # mean return per endurance fraction -> {frac: mean_return}
    per_frac: dict | None = None
    select_score: float = 0.0


def _greedy_nn_baseline(env: TTPDEnv) -> float:
    inst = env.inst
    env.reset(options={"instance": inst})
    total = 0.0
    while True:
        m_j = env.current_masks()["j"]
        cand = np.flatnonzero(m_j)
        custs = [int(j) for j in cand if j not in (inst.source, inst.sink)]
        if custs:
            c = env.c_t
            j = min(custs, key=lambda jj: inst.dist[c, jj])
        else:
            j = inst.sink
        _, r, term, trunc, _ = env.step(
            {"j": int(j), "z_curr": 0, "rejoin": 0, "z_drone": 0, "k": NO_LAUNCH}
        )
        total += r
        if term or trunc:
            return total


def _spec_env(spec: dict) -> TTPDEnv:
    # synthetic instance + relative endurance; same seed -> identical instance
    # (ED_frac consumes no RNG draws, so the coordinates/items match too)
    return TTPDEnv(n=spec["n"], ED_frac=spec["frac"])


def build_eval_set(
    n_instances: int,
    n_lo: int,
    n_hi: int,
    seed: int,
    sizes: list[int] | None = None,
    fracs: tuple = ED_FRACS_BENCH,
    a280_path: str | None = None,      # accepted for call-site compatibility; unused
    scale_capacity: bool = True,       # accepted for call-site compatibility; unused
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if sizes is None:
        graded = [s for s in (10, 20) if n_lo <= s <= n_hi]
        sizes = graded if graded else list(range(n_lo, n_hi + 1))
    specs = []
    for i in range(n_instances):
        n = int(sizes[i % len(sizes)])
        frac = float(fracs[(i // len(sizes)) % len(fracs)])
        specs.append({"n": n, "frac": frac,
                      "seed": int(rng.integers(0, 2**31 - 1))})
    for spec in specs:
        env = _spec_env(spec)
        env.reset(seed=spec["seed"])   # same seed evaluate() uses -> identical instance
        spec["baseline"] = _greedy_nn_baseline(env)
    return specs


def _rollout_once(policy, env, inst, ctx, first_j, deterministic, temperature: float = 1.0):
    """One full episode; pin the first truck node when first_j is given. Returns
    (return, steps, arrival) or (None, ...) if the policy emits an infeasible action."""
    obs, _ = env.reset(options={"instance": inst})
    G = 0.0
    s = 0
    done = False
    while not done:
        if s == 0 and first_j is not None:
            sample = policy.act_force(obs, env, {"j": int(first_j)}, ctx=ctx,
                                      deterministic=deterministic)
        else:
            sample = policy.act(obs, env, deterministic=deterministic, ctx=ctx,
                                temperature=temperature)
        try:
            obs, r, term, trunc, _ = env.step(sample.action)
        except ValueError:
            return None, s, env.tau_t
        G += r
        s += 1
        done = term or trunc
    return G, s, env.tau_t


def _pomo_best(policy, env, inst, *, n_aug: int, n_starts: int, deterministic: bool):
    env.reset(options={"instance": inst})
    feasible = [int(j) for j in np.flatnonzero(env.current_masks()["j"])
                if j not in (inst.source, inst.sink)]
    starts = (feasible[: max(1, n_starts)] or [None])
    best_G, best_s, best_tau = None, 0, 0.0
    for aug in range(max(1, n_aug)):
        ctx = policy.encode(inst, aug_idx=aug)
        for fj in starts:
            G, s, tau = _rollout_once(policy, env, inst, ctx, fj, deterministic)
            if G is not None and (best_G is None or G > best_G):
                best_G, best_s, best_tau = G, s, tau
    return best_G, best_s, best_tau


def evaluate(
    policy: AttentionPolicy,
    eval_set: list[dict],
    *,
    deterministic: bool = True,
    n_aug: int = 8,
    n_starts: int = 8,
    a280_path: str | None = None,      # accepted for call-site compatibility; unused
    scale_capacity: bool = True,       # accepted for call-site compatibility; unused
) -> EvalResult:
    import time
    policy.eval()
    returns = np.full(len(eval_set), np.nan)
    steps = np.zeros(len(eval_set), dtype=np.int64)
    taus = np.zeros(len(eval_set), dtype=np.float64)
    t0 = time.perf_counter()
    for i, spec in enumerate(eval_set):
        env = _spec_env(spec)
        env.reset(seed=spec["seed"])
        inst = env.inst
        G, s, tau = _pomo_best(policy, env, inst, n_aug=n_aug, n_starts=n_starts,
                               deterministic=deterministic)
        if G is not None:
            returns[i], steps[i], taus[i] = G, s, tau

    dt = time.perf_counter() - t0
    valid_mask = ~np.isnan(returns)
    valid = returns[valid_mask]
    if valid.size == 0:  # all rollouts failed with infeasible actions
        raise RuntimeError(
            "All eval rollouts failed with infeasible actions."
            "Policy is producing actions the env refuses."
        )
    sizes = np.array([s["n"] for s in eval_set])
    per_size = {}
    for n in sorted(set(int(x) for x in sizes)):
        sel = (sizes == n) & valid_mask
        if sel.any():
            per_size[int(n)] = float(returns[sel].mean())
    fracs = np.array([s["frac"] for s in eval_set])
    per_frac = {}
    for f in sorted(set(float(x) for x in fracs)):
        sel = (fracs == f) & valid_mask
        if sel.any():
            per_frac[float(f)] = float(returns[sel].mean())

    baselines = np.array([s.get("baseline", np.nan) for s in eval_set], dtype=np.float64)
    if np.isfinite(baselines).all() and np.all(np.abs(baselines) > 1e-9):
        improv = (returns - baselines) / np.abs(baselines)
        select_score = float(improv[valid_mask].mean())
    else:
        select_score = float(valid.mean())

    return EvalResult(
        n_instances=len(eval_set),
        mean_return=float(valid.mean()),
        std_return=float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        min_return=float(valid.min()),
        max_return=float(valid.max()),
        mean_steps=float(steps.mean()),
        mean_tau=float(taus.mean()),
        wall_seconds=dt,
        per_size=per_size,
        per_frac=per_frac,
        select_score=select_score,
    )
