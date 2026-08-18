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
_paths.use("rl/a280/gat", "core", "heuristics")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _ROOT
import time
import traceback
from datetime import datetime

from env.instance import load_a280, _build_instance        # noqa: E402
from core import problem_from_instance                      # noqa: E402
import ttpd_common as C                                     # noqa: E402
from sa import sa                                           # noqa: E402
from vns import vns                                         # noqa: E402

SIZES = [5, 10, 15, 20, 30, 40, 50, 100]
INSTANCES = [1, 2, 3, 4, 5]          # bench_<n>_<inst>.txt
SEEDS = list(range(10))              # 10 seeds per instance
METHODS = ["SA", "VNS"]              # SA fully, then VNS

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

ALPHA = 0.97       
VNS_KMAX = 8        
FINALE_COLUMNS = [
    "method", "n", "seed", "instance", "status",
    "objective", "gap_%", "time_s",
    "n_items", "n_truck", "n_drone", "n_rendv",
    "profit", "rental", "arrival",
    "W_final", "W_capacity", "W_actual",
    "n_nodes_explored", "build_s", "solve_s",
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

def gap_pct(ref, obj):
    if ref is None or ref == 0:
        return ""
    g = (ref - obj) / abs(ref) * 100.0
    return round(0.0 if g == 0 else g, 4)

def write_csv(path, rows, append=False):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if (append and file_exists) else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINALE_COLUMNS, extrasaction="ignore")
        if mode == "w" or not file_exists:
            w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in FINALE_COLUMNS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            pass

def run_one(method, n, inst_no, seed, P, inst, bench_path, refs, budget):
    t0 = time.perf_counter()
    if method == "SA":
        res = sa(P, time_limit=budget, seed=seed, alpha=ALPHA)
    elif method == "VNS":
        res = vns(P, time_limit=budget, k_max=VNS_KMAX, seed=seed)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0

    # Validate + extract rich stats by replaying in the real env.
    det = C.replay_details(inst, res.truck, res.sorties, res.z,
                           a280_path=bench_path)

    obj = det["objective"]
    ref = refs.get((n, inst_no))
    row = {
        "method": method, "n": n, "seed": seed,
        "instance": f"bench_{n}_{inst_no}", "status": "ok",
        "objective": round(obj, 4),
        "gap_%": gap_pct(ref, obj),
        "time_s": round(wall, 4),
        "n_items": det["n_items"], "n_truck": det["n_truck"],
        "n_drone": det["n_drone"], "n_rendv": det["n_rendv"],
        "profit": round(det["profit"], 4), "rental": round(det["rental"], 4),
        "arrival": round(det["arrival"], 4),
        "W_final": round(det["W_final"], 4),
        "W_capacity": round(float(inst.W), 4),
        "W_actual": round(float(sum(inst.weights)), 4),
        "n_nodes_explored": getattr(res, "evals", ""),
        "build_s": "", "solve_s": round(wall, 4),
        "collected_node_ids": det["collected_node_ids"],
        "truck_node_ids": det["truck_node_ids"],
        "drone_node_ids": det["drone_node_ids"],
        "rendezvous_node_ids": det["rendezvous_node_ids"],
        "truck_route_node_ids": det["truck_route_node_ids"],
        "drone_arc_node_ids": det["drone_arc_node_ids"],
        "W_sequence": det["W_sequence"],
    }
    return row


def main():
    ap = argparse.ArgumentParser(
        description="Sequential SA+VNS benchmark runner (batch-ready).")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES),
                    help="comma-separated n values (default: all)")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated methods (default: SA,VNS)")
    ap.add_argument("--instances", type=int, default=len(INSTANCES),
                    help="number of instances per n (default 5)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of seeds per instance (default 10)")
    ap.add_argument("--finale", default=os.path.join(ROOT, "finale.csv"),
                    help="master flat results CSV (appended)")
    ap.add_argument("--logfile", default=os.path.join(HERE, "sa_vns_run.log"))
    cli = ap.parse_args()

    sys.stdout = Tee(cli.logfile)

    sizes = [int(s) for s in cli.sizes.split(",")]
    methods = [m.strip().upper() for m in cli.methods.split(",")]
    instances = list(range(1, cli.instances + 1))
    seeds = list(range(cli.seeds))
    finale_path = os.path.abspath(cli.finale)

    log("=" * 78)
    log(f"SA+VNS BENCHMARK RUNNER  pid={os.getpid()}  host={os.uname().nodename}")
    log(f"sizes={sizes}  methods={methods}  instances={instances}  seeds={seeds}")
    log(f"time budgets={ {n: TIME_BUDGET[n] for n in sizes} }")
    log(f"finale={finale_path}")
    log("=" * 78)

    refs = load_milp_refs()
    log(f"loaded {len(refs)} MILP reference objectives for gap_%")

    grand_total = len(methods) * sum(
        min(cli.instances, len(INSTANCES)) * cli.seeds for _ in sizes)
    done = 0
    t_start = time.perf_counter()

    for method in methods:
        for n in sizes:
            budget = TIME_BUDGET.get(n)
            if budget is None:
                log(f"  [skip] no time budget configured for n={n}")
                continue
            per_n_rows = []
            log("-" * 78)
            log(f"METHOD={method}  n={n}  budget={budget}s/run  "
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
                        row = run_one(method, n, inst_no, seed, P, inst,
                                      bench_path, refs, budget)
                    except Exception as e:
                        log(f"  [ERROR] {method} n={n} inst={inst_no} "
                            f"seed={seed}: {e}")
                        traceback.print_exc()
                        row = {
                            "method": method, "n": n, "seed": seed,
                            "instance": f"bench_{n}_{inst_no}",
                            "status": f"error: {e}",
                        }
                    per_n_rows.append(row)
                    # crash-safe: append each run to finale.csv immediately
                    write_csv(finale_path, [row], append=True)
                    g = row.get("gap_%", "")
                    log(f"  [{done}/{grand_total}] {method} n={n} "
                        f"inst={inst_no} seed={seed}  obj={row.get('objective')}"
                        f"  gap={g}%  t={row.get('time_s')}s  "
                        f"T/D/R={row.get('n_truck')}/{row.get('n_drone')}/"
                        f"{row.get('n_rendv')}")

            per_n_path = os.path.join(HERE, f"{method}_{n}.csv")
            write_csv(per_n_path, per_n_rows, append=False)
            log(f"  wrote {per_n_path}  ({len(per_n_rows)} rows)")

    elapsed = time.perf_counter() - t_start
    log("=" * 78)
    log(f"ALL DONE.  {done} runs in {elapsed/60:.1f} min.  "
        f"finale={finale_path}")
    log("=" * 78)

if __name__ == "__main__":
    main()
