#a
#7; evaluates policy on held-out instances + the exact MILP benchmark instances
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch

from env import TTPDEnv
from env.instance import load_a280, milp_instance
from env.masking import NO_LAUNCH
from policy.attention_policy import AttentionPolicy

MILP_OBJECTIVE = {5: -16725.35, 10: -27493.21, 15: -36397.39, 20: -40099.31}
MILP_PROVEN_OPTIMAL = {5, 10, 15}


def load_milp_refs(milp_dir: str | None = None) -> dict:
    import csv as _csv
    import os as _os
    refs = dict(MILP_OBJECTIVE)
    if milp_dir is None:
        return refs
    for fname in ("milp_results.csv", "ttpd_results.csv", "n15.csv", "n20.csv"):
        path = _os.path.join(milp_dir, fname)
        if not _os.path.exists(path):
            continue
        try:
            with open(path, newline="") as f:
                for row in _csv.DictReader(f):
                    n = int(row["n"])
                    obj = float(row["objective"])
                    w_cap = float(row.get("W_capacity") or 0.0)
                    if w_cap > 600_000:        # unscaled-capacity legacy solve
                        continue
                    if n not in refs or obj > refs[n]:
                        refs[n] = obj
        except Exception:
            continue
    return refs


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
    # mean greedy return per problem size n -> {n: mean_return} on the random held-out set
    per_size: dict | None = None
    select_score: float = 0.0
    # exact-MILP-instance results: {n: {"return": .., "ref": .., "gap_pct": ..}}
    milp_gap: dict | None = None


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


def build_eval_set(
    a280_path: str,
    n_instances: int,
    n_lo: int,
    n_hi: int,
    seed: int,
    scale_capacity: bool = True,
    sizes: list[int] | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if sizes is None:
        graded = [s for s in (5, 10, 15, 20) if n_lo <= s <= n_hi]
        sizes = graded if graded else list(range(n_lo, n_hi + 1))
    specs = []
    for i in range(n_instances):
        n = int(sizes[i % len(sizes)])
        specs.append({"n": n, "seed": int(rng.integers(0, 2**31 - 1))})
    for spec in specs:
        env = TTPDEnv(a280_path=a280_path, n=spec["n"], scale_capacity=scale_capacity)
        env.reset(seed=spec["seed"])           # same seed evaluate() uses -> identical instance
        spec["baseline"] = _greedy_nn_baseline(env)
    return specs


def _rollout_once(policy, env, inst, ctx, first_j, deterministic, temperature: float = 1.0):
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


def evaluate_milp_gap(
    policy: AttentionPolicy,
    a280_path: str,
    *,
    sizes: tuple[int, ...] = (5, 10, 15, 20),
    n_aug: int = 8,
    n_starts: int = 8,
    deterministic: bool = True,
    milp_dir: str | None = None,
) -> dict:
    refs = load_milp_refs(milp_dir)
    a280 = load_a280(a280_path)
    out = {}
    for n in sizes:
        ref = refs.get(n)
        if ref is None:
            continue
        inst = milp_instance(a280, n, seed=n, scale_capacity=True)
        env = TTPDEnv(a280_path=a280_path, n=n, scale_capacity=True)
        G, _, tau = _pomo_best(policy, env, inst, n_aug=n_aug, n_starts=n_starts,
                               deterministic=deterministic)
        gap = float("nan") if G is None else (ref - G) / abs(ref) * 100.0
        out[int(n)] = {"return": G, "ref": ref, "gap_pct": gap,
                       "proven_optimal": n in MILP_PROVEN_OPTIMAL}
    return out


def evaluate(
    policy: AttentionPolicy,
    a280_path: str,
    eval_set: list[dict],
    *,
    deterministic: bool = True,
    scale_capacity: bool = True,
    n_aug: int = 8,
    n_starts: int = 8,
    with_milp_gap: bool = True,
    milp_dir: str | None = None,
    milp_sizes: tuple = (5, 10, 15, 20),
) -> EvalResult:
    import time
    policy.eval()
    returns = np.full(len(eval_set), np.nan)
    steps = np.zeros(len(eval_set), dtype=np.int64)
    taus = np.zeros(len(eval_set), dtype=np.float64)
    t0 = time.perf_counter()
    for i, spec in enumerate(eval_set):
        env = TTPDEnv(a280_path=a280_path, n=spec["n"], scale_capacity=scale_capacity)
        env.reset(seed=spec["seed"])
        inst = env.inst
        G, s, tau = _pomo_best(policy, env, inst, n_aug=n_aug, n_starts=n_starts,
                               deterministic=deterministic)
        if G is not None:
            returns[i], steps[i], taus[i] = G, s, tau

    milp_gap = None
    if with_milp_gap:
        milp_gap = evaluate_milp_gap(policy, a280_path, sizes=milp_sizes,
                                     n_aug=n_aug, n_starts=n_starts,
                                     deterministic=deterministic, milp_dir=milp_dir)

    dt = time.perf_counter() - t0
    valid_mask = ~np.isnan(returns)
    valid = returns[valid_mask]
    if valid.size == 0: # if all rollouts failed with infeasible actions
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
        select_score=select_score,
        milp_gap=milp_gap,
    )
