from __future__ import annotations

import argparse
import csv
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

HERE = os.path.dirname(os.path.abspath(__file__))
import time
import traceback
from datetime import datetime

from instance import load_a280, _build_instance     
from core import problem_from_instance, check_solution, evaluate  
import ttpd_common as C                              
from vns import vns                                  

SIZES = [5, 10, 15, 20, 30, 40, 50, 100]
INSTANCES = [1, 2, 3, 4, 5]         
SEEDS = list(range(10))              
VNS_KMAX = 8                       

TIME_BUDGET = {
    5:   10.0,
    10:  30.0,
    15:  120.0,
    20:  300.0,
    30:  450.0,
    40:  600.0,
    50:  750.0,
    100: 1200.0,
}

MILP_REFS: dict = {}   

def load_milp_refs():
    refs = {}
    for n in SIZES:
        path = os.path.join(HERE, f"MILP{n}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    inst_no = int(str(row["instance"]).split("_")[-1])
                    refs[(n, inst_no)] = float(row["objective"])
                except (ValueError, KeyError, TypeError):
                    continue
    return refs

# Flat CSV schema -- one row per run.
COLUMNS = [
    "method", "n", "seed", "instance", "status",
    "objective", "gap_%", "time_s",
    "n_items", "n_truck", "n_drone", "n_rendv",
    "profit", "rental", "arrival",
    "W_final", "W_capacity", "W_actual",
    "evals", "time_to_best_s",
    "collected_node_ids", "truck_node_ids", "drone_node_ids",
    "rendezvous_node_ids", "truck_route_node_ids", "drone_arc_node_ids",
    "W_sequence",
]

class Tee:
    def __init__(self, path):
        self.term = sys.stdout
        self.f = open(path, "a", buffering=1)

    def write(self, s):
        self.term.write(s)
        self.f.write(s)

    def flush(self):
        self.term.flush()
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except (OSError, ValueError):
            pass

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def load_bench_instance(path):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None,
                           scale_capacity=False)
    return problem_from_instance(inst), inst

def gap_pct(n, inst_no, obj):
    ref = MILP_REFS.get((n, inst_no))
    if ref is None or ref == 0:
        return ""
    g = (ref - obj) / abs(ref) * 100.0
    return round(0.0 if g == 0 else g, 4)

def append_row(path, row):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            pass

def run_one(n, inst_no, seed, P, inst, bench_path, budget):
    t0 = time.perf_counter()
    res = vns(P, time_limit=budget, k_max=VNS_KMAX, seed=seed)
    wall = time.perf_counter() - t0

    # Structural feasibility check (cheap) then env replay (validates + stats).
    err = check_solution(P, res.truck, res.sorties, res.z)
    if err is not None:
        raise RuntimeError(f"infeasible VNS solution: {err}")
    det = C.replay_details(inst, res.truck, res.sorties, res.z,
                           a280_path=bench_path)

    # Cross-check the standalone evaluator agrees with the env replay.
    own = evaluate(P, res.truck, res.sorties, res.z)
    if own is None or abs(own - det["objective"]) > 1e-6:
        raise RuntimeError(f"evaluator/env mismatch eval={own} "
                           f"env={det['objective']}")

    obj = det["objective"]
    return {
        "method": "VNS", "n": n, "seed": seed,
        "instance": f"bench_{n}_{inst_no}", "status": "ok",
        "objective": round(obj, 4),
        "gap_%": gap_pct(n, inst_no, obj),
        "time_s": round(wall, 4),
        "n_items": det["n_items"], "n_truck": det["n_truck"],
        "n_drone": det["n_drone"], "n_rendv": det["n_rendv"],
        "profit": round(det["profit"], 4), "rental": round(det["rental"], 4),
        "arrival": round(det["arrival"], 4),
        "W_final": round(det["W_final"], 4),
        "W_capacity": round(float(inst.W), 4),
        "W_actual": round(float(sum(inst.weights)), 4),
        "evals": getattr(res, "evals", ""),
        "time_to_best_s": round(getattr(res, "time_to_best", 0.0), 4),
        "collected_node_ids": det["collected_node_ids"],
        "truck_node_ids": det["truck_node_ids"],
        "drone_node_ids": det["drone_node_ids"],
        "rendezvous_node_ids": det["rendezvous_node_ids"],
        "truck_route_node_ids": det["truck_route_node_ids"],
        "drone_arc_node_ids": det["drone_arc_node_ids"],
        "W_sequence": det["W_sequence"],
    }

def main():
    ap = argparse.ArgumentParser(
        description="Sequential VNS benchmark runner (flat, batch-ready).")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES),
                    help="comma-separated n values (default: all, in order)")
    ap.add_argument("--instances", type=int, default=len(INSTANCES),
                    help="number of instances per n (default 5)")
    ap.add_argument("--inst-list", default=None,
                    help="explicit comma-separated instance numbers to run "
                         "(overrides --instances, e.g. '5' for only bench_<n>_5)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of seeds per instance (default 10)")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed to run, inclusive (for sharding; default 0)")
    ap.add_argument("--seed-end", type=int, default=None,
                    help="last seed to run, EXCLUSIVE (for sharding; "
                         "default = --seeds). Shard seeds [seed-start, seed-end).")
    ap.add_argument("--seed-list", default=None,
                    help="explicit comma-separated seeds to run (overrides "
                         "--seeds/--seed-start/--seed-end). Used by run_parallel.py "
                         "to hand each shard its exact, load-balanced seed set.")
    ap.add_argument("--out", default=os.path.join(HERE, "results_vns.csv"),
                    help="results CSV (appended, fsync'd per row)")
    ap.add_argument("--logfile", default=os.path.join(HERE, "run_vns.log"))
    cli = ap.parse_args()

    sys.stdout = Tee(cli.logfile)

    sizes = [int(s) for s in cli.sizes.split(",")]
    if cli.inst_list is not None:
        instances = [int(s) for s in cli.inst_list.split(",") if s.strip() != ""]
    else:
        instances = list(range(1, cli.instances + 1))
    if cli.seed_list is not None:
        seeds = [int(s) for s in cli.seed_list.split(",") if s.strip() != ""]
    else:
        seed_end = cli.seeds if cli.seed_end is None else cli.seed_end
        seeds = list(range(cli.seed_start, seed_end))
    out_path = os.path.abspath(cli.out)

    global MILP_REFS
    MILP_REFS = load_milp_refs()

    log("=" * 78)
    log(f"VNS BENCHMARK RUNNER  pid={os.getpid()}  host={os.uname().nodename}")
    log(f"sizes={sizes}  instances={instances}  seeds={seeds}")
    log(f"time budgets={ {n: TIME_BUDGET[n] for n in sizes if n in TIME_BUDGET} }")
    log(f"loaded {len(MILP_REFS)} per-instance MILP refs for gap_% "
        f"(blank where absent)")
    log(f"out={out_path}")
    log("=" * 78)

    grand_total = sum(len(instances) * len(seeds) for _ in sizes)
    done = 0
    t_start = time.perf_counter()

    for n in sizes:
        budget = TIME_BUDGET.get(n)
        if budget is None:
            log(f"  [skip] no time budget configured for n={n}")
            continue
        log("-" * 78)
        log(f"n={n}  budget={budget}s/run  "
            f"({len(instances)} instances x {len(seeds)} seeds = "
            f"{len(instances) * len(seeds)} runs)")
        for inst_no in instances:
            bench_path = hub_instance("a280", f"bench_{n}_{inst_no}.txt")
            if not os.path.exists(bench_path):
                log(f"  MISSING {bench_path} -- skipping instance")
                continue
            P, inst = load_bench_instance(bench_path)
            for seed in seeds:
                done += 1
                try:
                    row = run_one(n, inst_no, seed, P, inst,
                                  bench_path, budget)
                except Exception as e:
                    log(f"  [ERROR] n={n} inst={inst_no} seed={seed}: {e}")
                    traceback.print_exc()
                    row = {
                        "method": "VNS", "n": n, "seed": seed,
                        "instance": f"bench_{n}_{inst_no}",
                        "status": f"error: {e}",
                    }
                # crash-safe: write+fsync this run's row to disk right now.
                append_row(out_path, row)
                log(f"  [{done}/{grand_total}] n={n} inst={inst_no} "
                    f"seed={seed}  obj={row.get('objective')}  "
                    f"gap={row.get('gap_%')}%  t={row.get('time_s')}s  "
                    f"T/D/R={row.get('n_truck')}/{row.get('n_drone')}/"
                    f"{row.get('n_rendv')}")

    elapsed = time.perf_counter() - t_start
    log("=" * 78)
    log(f"ALL DONE.  {done} runs in {elapsed/60:.1f} min.  out={out_path}")
    log("=" * 78)

if __name__ == "__main__":
    main()
