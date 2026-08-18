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

ROOT = _ROOT
import time
from multiprocessing import Pool

import numpy as np

A280 = hub_instance("a280", "a280_benchmark.txt")

from core import check_solution, evaluate, problem_from_instance  # noqa: E402
from ttpd_common import _build_actions                            # noqa: E402
from sa import sa                                                 # noqa: E402
from instance import load_a280, _build_instance, sample_instance  # noqa: E402
from simulator import TTPDEnv                                     # noqa: E402

FULL_BUDGET = {5: 10.0, 10: 30.0, 15: 120.0, 20: 300.0,
               30: 450.0, 40: 600.0, 50: 750.0, 100: 1200.0}
INSTANCES = [1, 2, 3, 4, 5]
ENV_EPS = 1e-4

def _instance_to_dict(inst) -> dict:
    return {
        "n": int(inst.n),
        "coords": np.asarray(inst.coords, dtype=float).tolist(),
        "profits": np.asarray(inst.profits, dtype=float).tolist(),
        "weights": np.asarray(inst.weights, dtype=float).tolist(),
        "dist": np.asarray(inst.dist, dtype=float).tolist(),
        "W": float(inst.W), "v_max": float(inst.v_max), "v_min": float(inst.v_min),
        "v_D": float(inst.v_D), "R": float(inst.R),
        "node_ids": list(inst.node_ids),
    }

def _load_bench(n: int, instance: int):
    data = load_a280(hub_instance("a280", f"bench_{n}_{instance}.txt"))
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return problem_from_instance(inst), inst

def _load_sampled(n: int, sample_seed: int):
    data = load_a280(A280)
    rng = np.random.default_rng(sample_seed)
    inst = sample_instance(data, n=n, rng=rng, scale_capacity=True)
    return problem_from_instance(inst), inst

def _verify_replay(inst, truck, sorties, z, actions) -> tuple[float | None, str | None]:
    env = TTPDEnv(a280_path=A280, n=inst.n, scale_capacity=False)
    env.reset(options={"instance": inst})
    g = 0.0
    term = trunc = False
    try:
        for a in actions:
            _, r, term, trunc, _ = env.step(a)
            g += float(r)
            if term or trunc:
                break
    except (ValueError, RuntimeError) as e:
        return None, f"env rejected action: {e}"
    if not term:
        return None, "did not terminate"
    return g, None

def _solve_one(task: dict) -> dict | None:
    n, seed, budget = task["n"], task["seed"], task["budget"]
    if task["source"] == "bench":
        tag = f"bench_{n}_{task['instance']}"
        P, inst = _load_bench(n, task["instance"])
    else:
        tag = f"sample_n{n}_s{task['sample_seed']}"
        P, inst = _load_sampled(n, task["sample_seed"])

    res = sa(P, time_limit=budget, seed=seed)
    truck, sorties, z = res.truck, res.sorties, res.z
    sa_obj = evaluate(P, truck, sorties, z)

    if check_solution(P, truck, sorties, z) is not None:
        return {"_error": f"{tag} seed={seed}: structurally infeasible"}
    actions = _build_actions(inst, truck, sorties, z)
    env_obj, err = _verify_replay(inst, truck, sorties, z, actions)
    if err is not None or abs(env_obj - sa_obj) > ENV_EPS:
        return {"_error": f"{tag} seed={seed}: replay failed ({err or 'obj mismatch'})"}

    return {
        "id": f"{tag}_seed{seed}", "source": tag, "n": n, "seed": seed,
        "objective": float(sa_obj), "n_actions": len(actions),
        "instance": _instance_to_dict(inst), "actions": actions,
    }

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--bench", action="store_true", help="include the 5 bench files/size")
    ap.add_argument("--sample", type=int, default=0, help="fresh sampled instances/size")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="start sampled index at this offset (for disjoint enrichment)")
    ap.add_argument("--seeds", type=int, default=1, help="SA seeds per instance")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiplier on the full per-size SA budgets")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--out", type=str, required=True, help="output .jsonl path")
    args = ap.parse_args()

    if not args.bench and args.sample <= 0:
        ap.error("pick at least one source: --bench and/or --sample COUNT")

    tasks: list[dict] = []
    for n in args.sizes:
        budget = max(0.5, FULL_BUDGET.get(n, 60.0) * args.time_scale)
        if args.bench:
            for instance in INSTANCES:
                for seed in range(args.seeds):
                    tasks.append({"source": "bench", "n": n, "instance": instance,
                                  "seed": seed, "budget": budget})
        for s in range(args.sample_offset, args.sample_offset + args.sample):
            for seed in range(args.seeds):
                tasks.append({"source": "sample", "n": n, "sample_seed": 10_000 * n + s,
                              "seed": seed, "budget": budget})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"[gen] {len(tasks)} tasks -> {args.out}  (jobs={args.jobs}, "
          f"time-scale={args.time_scale})")
    t0 = time.perf_counter()

    n_ok = n_err = 0
    with open(args.out, "w") as fh:
        if args.jobs > 1:
            with Pool(args.jobs) as pool:
                it = pool.imap_unordered(_solve_one, tasks)
                for i, rec in enumerate(it, 1):
                    n_ok, n_err = _handle(rec, fh, n_ok, n_err)
                    if i % 25 == 0 or i == len(tasks):
                        print(f"  [{i}/{len(tasks)}] ok={n_ok} err={n_err} "
                              f"({time.perf_counter()-t0:.0f}s)")
        else:
            for i, task in enumerate(tasks, 1):
                rec = _solve_one(task)
                n_ok, n_err = _handle(rec, fh, n_ok, n_err)
                if i % 5 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] ok={n_ok} err={n_err} "
                          f"({time.perf_counter()-t0:.0f}s)")

    print(f"\n[gen] done: {n_ok} records, {n_err} errors, "
          f"{time.perf_counter()-t0:.0f}s -> {args.out}")

def _handle(rec, fh, n_ok, n_err):
    if rec is None or "_error" in rec:
        if rec is not None:
            print(f"    SKIP {rec['_error']}")
        return n_ok, n_err + 1
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    return n_ok + 1, n_err

if __name__ == "__main__":
    main()
