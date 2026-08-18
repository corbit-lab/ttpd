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

from ttpd import _paths
_paths.use("core", "heuristics")

from ttpd.hub import ensure_local   
import time
from multiprocessing import get_context

import numpy as np

from ttd300_io import (FULL_BUDGET, LAYOUTS, ED_FRACS, TEMPLATE,        
                       bench_label, load_ttd300, sample_ttd300)
from core import check_solution, evaluate
from ttpd_common import _build_actions
from sa import sa
from simulator import TTPDEnv 

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
        "ED": None if inst.ED is None else float(inst.ED),
    }

def _verify_replay(inst, actions) -> tuple[float | None, str | None]:
    env = TTPDEnv(a280_path=TEMPLATE, n=inst.n, scale_capacity=False)
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
        tag = bench_label(n, task["layout"], task["ed_frac"])
        P, inst = load_ttd300(n, task["layout"], task["ed_frac"])
        ed_frac = task["ed_frac"]
    else:
        P, inst, ed_frac = sample_ttd300(n, task["sample_seed"],
                                         ed_fracs=task["ed_fracs"])
        tag = f"sample_n{n}_s{task['sample_seed']}"

    res = sa(P, time_limit=budget, seed=seed)
    truck, sorties, z = res.truck, res.sorties, res.z
    sa_obj = evaluate(P, truck, sorties, z)

    if check_solution(P, truck, sorties, z) is not None:
        return {"_error": f"{tag} seed={seed}: structurally infeasible"}
    actions = _build_actions(inst, truck, sorties, z)
    env_obj, err = _verify_replay(inst, actions)
    if err is not None or abs(env_obj - sa_obj) > ENV_EPS:
        return {"_error": f"{tag} seed={seed}: replay failed ({err or 'obj mismatch'})"}

    return {
        "id": f"{tag}_seed{seed}", "source": tag, "n": n, "seed": seed,
        "ED": None if inst.ED is None else float(inst.ED),
        "ed_frac": float(ed_frac),
        "objective": float(sa_obj), "n_actions": len(actions),
        "instance": _instance_to_dict(inst), "actions": actions,
    }

def _task_id(t: dict) -> str:
    """Record id a task will produce (must mirror _solve_one) -- for --resume."""
    if t["source"] == "bench":
        return f"{bench_label(t['n'], t['layout'], t['ed_frac'])}_seed{t['seed']}"
    return f"sample_n{t['n']}_s{t['sample_seed']}_seed{t['seed']}"

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--bench", action="store_true",
                    help="include the Data/ benchmark instances")
    ap.add_argument("--layouts", type=int, nargs="+", default=LAYOUTS,
                    help="bench layouts to include")
    ap.add_argument("--ed-fracs", type=float, nargs="+", default=ED_FRACS,
                    help="bench ED fracs to include / sampling pool for --sample")
    ap.add_argument("--sample", type=int, default=0, help="fresh sampled instances/size")
    ap.add_argument("--sample-offset", type=int, default=0,
                    help="start sampled index at this offset (for disjoint enrichment)")
    ap.add_argument("--seeds", type=int, default=1, help="SA seeds per instance")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiplier on the full per-size SA budgets")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--out", type=str, required=True, help="output .jsonl path")
    ap.add_argument("--resume", action="store_true",
                    help="append to --out, skipping records already in it "
                         "(errored/missing ones are retried)")
    args = ap.parse_args()

    if not args.bench and args.sample <= 0:
        ap.error("pick at least one source: --bench and/or --sample COUNT")

    tasks: list[dict] = []
    for n in args.sizes:
        budget = max(0.5, FULL_BUDGET.get(n, 300.0) * args.time_scale)
        if args.bench:
            for L in args.layouts:
                for frac in args.ed_fracs:
                    for seed in range(args.seeds):
                        tasks.append({"source": "bench", "n": n, "layout": L,
                                      "ed_frac": frac, "seed": seed,
                                      "budget": budget})
        for s in range(args.sample_offset, args.sample_offset + args.sample):
            for seed in range(args.seeds):
                tasks.append({"source": "sample", "n": n,
                              "sample_seed": 10_000 * n + s,
                              "ed_fracs": tuple(args.ed_fracs),
                              "seed": seed, "budget": budget})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mode = "w"
    if args.resume and os.path.exists(args.out):
        done: set[str] = set()
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
        n_before = len(tasks)
        tasks = [t for t in tasks if _task_id(t) not in done]
        mode = "a"
        print(f"[gen] resume: {n_before - len(tasks)} records already in "
              f"{args.out}, {len(tasks)} to run")
    print(f"[gen] {len(tasks)} tasks -> {args.out}  (jobs={args.jobs}, "
          f"time-scale={args.time_scale})")
    t0 = time.perf_counter()

    n_ok = n_err = 0
    with open(args.out, mode) as fh:
        if args.jobs > 1:
            ctx = get_context("spawn")
            with ctx.Pool(args.jobs) as pool:
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
