from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("core", "heuristics")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402
import time
from dataclasses import dataclass, asdict

from core import check_solution, evaluate          # noqa: E402  ttpd/core/core.py
from ttpd_common import _build_actions             # noqa: E402  solution->actions
from sa import sa                                  # noqa: E402  the SA solver
from run_sa import load_bench_instance             # noqa: E402  bench loader
from simulator import TTPDEnv                       # noqa: E402  the real env

ENV_EPS = 1e-4   # env-replay vs evaluate() objective tolerance
SIZES = [5, 10, 15, 20, 30, 40, 50, 100]
INSTANCES = [1, 2, 3, 4, 5]

TIME_BUDGET = {5: 2.0, 10: 3.0, 15: 4.0, 20: 5.0,
               30: 6.0, 40: 8.0, 50: 10.0, 100: 15.0}

@dataclass
class ReplayResult:
    n: int
    instance: int
    seed: int
    sa_objective: float
    structurally_feasible: bool
    replayed: bool
    env_objective: float | None
    obj_match: bool
    n_actions: int
    error: str | None

def invert_and_verify(inst, truck, sorties, z, bench_path):
    actions = _build_actions(inst, truck, sorties, z)
    env = TTPDEnv(a280_path=bench_path, n=inst.n, scale_capacity=False)
    env.reset(options={"instance": inst})

    env_obj = 0.0
    terminated = truncated = False
    try:
        for a in actions:
            _, r, terminated, truncated, _ = env.step(a)
            env_obj += float(r)
            if terminated or truncated:
                break
    except (ValueError, RuntimeError) as e:
        return actions, None, f"env rejected action: {e}"

    if truncated and not terminated:
        return actions, None, "env truncated (unserved customers / drone aloft)"
    if not terminated:
        return actions, None, "actions exhausted without termination"
    return actions, env_obj, None

def run_one(n: int, instance: int, seed: int, time_budget: float,
            dump_fh=None) -> ReplayResult:
    path = hub_instance("a280", f"bench_{n}_{instance}.txt")
    P, inst = load_bench_instance(path)

    res = sa(P, time_limit=time_budget, seed=seed)
    truck, sorties, z = res.truck, res.sorties, res.z
    sa_obj = evaluate(P, truck, sorties, z)

    struct_err = check_solution(P, truck, sorties, z)
    structurally_feasible = struct_err is None

    actions, env_obj, err = invert_and_verify(inst, truck, sorties, z, path)
    replayed = err is None
    obj_match = replayed and abs(env_obj - sa_obj) <= ENV_EPS

    if dump_fh is not None and replayed and obj_match:
        dump_fh.write(json.dumps({
            "n": n, "instance": instance, "seed": seed,
            "bench_file": f"bench_{n}_{instance}.txt",
            "objective": sa_obj,
            "actions": actions,
        }) + "\n")

    return ReplayResult(
        n=n, instance=instance, seed=seed, sa_objective=round(sa_obj, 4),
        structurally_feasible=structurally_feasible, replayed=replayed,
        env_objective=None if env_obj is None else round(env_obj, 4),
        obj_match=obj_match, n_actions=len(actions),
        error=struct_err or err or (None if obj_match else "obj mismatch"),
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--instances", type=int, nargs="+", default=INSTANCES)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of SA seeds per (n, instance)")
    ap.add_argument("--dump", type=str, default=None,
                    help="optional .jsonl path for the BC trajectories")
    args = ap.parse_args()

    dump_fh = open(args.dump, "w") if args.dump else None
    results: list[ReplayResult] = []
    t0 = time.perf_counter()

    for n in args.sizes:
        budget = TIME_BUDGET.get(n, 5.0)
        for instance in args.instances:
            for seed in range(args.seeds):
                r = run_one(n, instance, seed, budget, dump_fh)
                results.append(r)
                flag = "OK " if (r.replayed and r.obj_match) else "XX "
                print(f"  {flag}n={n:<3} inst={instance} seed={seed} "
                      f"obj={r.sa_objective:>12} env={r.env_objective} "
                      f"acts={r.n_actions}"
                      + (f"  <- {r.error}" if r.error else ""))

    if dump_fh:
        dump_fh.close()

    print("\n=== REPLAY COVERAGE (the SA-teaches-GAT feasibility gate) ===")
    print(f"{'n':>4} {'tried':>6} {'struct_ok':>10} {'replayed':>9} "
          f"{'obj_match':>10} {'coverage_%':>11}")
    overall = [0, 0]
    for n in args.sizes:
        rs = [r for r in results if r.n == n]
        tried = len(rs)
        struct = sum(r.structurally_feasible for r in rs)
        rep = sum(r.replayed for r in rs)
        match = sum(r.replayed and r.obj_match for r in rs)
        cov = 100.0 * match / tried if tried else 0.0
        overall[0] += match
        overall[1] += tried
        print(f"{n:>4} {tried:>6} {struct:>10} {rep:>9} {match:>10} {cov:>10.1f}%")
    tot = 100.0 * overall[0] / overall[1] if overall[1] else 0.0
    print(f"{'ALL':>4} {overall[1]:>6} {'':>10} {'':>9} {overall[0]:>10} {tot:>10.1f}%")
    print(f"\n(elapsed {time.perf_counter() - t0:.1f}s)")

    if overall[0] < overall[1]:
        print("\nNOTE: coverage < 100%.")

if __name__ == "__main__":
    main()
