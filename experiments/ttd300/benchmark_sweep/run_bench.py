from __future__ import annotations
import argparse
import csv
import fcntl
import io
import multiprocessing as mp
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

SIZES_SA = [30, 40, 50, 75, 100]
SIZES_VNS = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2, 3, 4]         
BUDGET = {10: 120.0, 20: 300.0, 30: 450.0, 40: 600.0,
          50: 750.0, 75: 975.0, 100: 1200.0}
SA_ALPHA = 0.97
VNS_KMAX = 8
WORKERS = 4

OUT_CSV = os.path.join(HERE, "results.csv")
RUN_LOG = os.path.join(HERE, "run.log")

COLUMNS = [
    "method", "dataset", "n", "layout", "ED_frac", "ED", "d_max", "instance",
    "seed", "status", "objective", "profit", "rental", "arrival",
    "time_s", "time_to_best_s", "n_items", "n_truck", "n_drone", "n_rendv",
    "W_final", "W_capacity", "evals", "reheats", "error",
]

def frac_tag(frac):
    return f"f{int(round(frac * 100)):03d}"

def build_jobs():
    jobs = []
    for method, sizes in (("SA", SIZES_SA), ("VNS", SIZES_VNS)):
        for n in sizes:
            for frac in FRACS:
                for L in LAYOUTS:
                    for seed in SEEDS:
                        jobs.append((method, n, L, frac, seed))
    jobs.sort(key=lambda j: (BUDGET[j[1]], j[1], j[4], j[3], j[2], j[0]))
    return jobs

def job_key(method, n, L, frac, seed):
    return (method, f"ttd300_n{n}_L{L}_{frac_tag(frac)}", str(seed))

def done_keys(path):
    """(method, instance, seed) of rows already in the CSV (any status ok);
    errored rows are retried on resume."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((row["method"], row["instance"], row["seed"]))
    return done

def locked_append(path, text):
    with open(path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def log(logfile, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    locked_append(logfile, line + "\n")

def append_row(csv_path, row):
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=COLUMNS).writerow(row)
    locked_append(csv_path, buf.getvalue())

def run_one(method, n, L, frac, seed, budget):
    from instance import load_a280, _build_instance
    from core import problem_from_instance, check_solution, evaluate
    import ttpd_common as C
    from sa import sa
    from vns import vns

    tag = frac_tag(frac)
    path = hub_instance("ttd300", f"ttd300_n{n}_L{L}_{tag}.txt")
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    P = problem_from_instance(inst)
    ED = inst.ED
    assert ED is not None, f"no DRONE ENDURANCE header in {path}"

    t0 = time.perf_counter()
    if method == "SA":
        res = sa(P, time_limit=budget, seed=seed, alpha=SA_ALPHA)
    else:
        res = vns(P, time_limit=budget, k_max=VNS_KMAX, seed=seed)
    wall = time.perf_counter() - t0

    err = check_solution(P, res.truck, res.sorties, res.z)
    if err is not None:
        raise RuntimeError(f"infeasible {method} solution: {err}")
    for (Ln, D, J) in res.sorties:                      # endurance must hold
        reach = P.dist[Ln][D] + P.dist[D][J]
        assert reach <= ED + 1e-6, \
            f"SORTIE EXCEEDS ED {reach:.2f} > {ED} ({Ln},{D},{J})"

    det = C.replay_details(inst, res.truck, res.sorties, res.z, a280_path=path)
    own = evaluate(P, res.truck, res.sorties, res.z)
    if own is None or abs(own - det["objective"]) > 1e-6:
        raise RuntimeError(f"evaluator/env mismatch eval={own} env={det['objective']}")

    return {
        "method": method, "dataset": "ttd300", "n": n, "layout": L,
        "ED_frac": frac, "ED": ED, "d_max": round(float(inst.d_max), 1),
        "instance": f"ttd300_n{n}_L{L}_{tag}",
        "seed": seed, "status": "ok",
        "objective": round(det["objective"], 4),
        "profit": round(det["profit"], 4),
        "rental": round(det["rental"], 4),
        "arrival": round(det["arrival"], 4),
        "time_s": round(wall, 4),
        "time_to_best_s": round(getattr(res, "time_to_best", 0.0), 4),
        "n_items": det["n_items"], "n_truck": det["n_truck"],
        "n_drone": det["n_drone"], "n_rendv": det["n_rendv"],
        "W_final": round(det["W_final"], 4),
        "W_capacity": round(float(inst.W), 4),
        "evals": getattr(res, "evals", ""),
        "reheats": getattr(res, "reheats", ""),
        "error": "",
    }

def error_row(method, n, L, frac, seed, exc):
    row = {c: "" for c in COLUMNS}
    row.update({
        "method": method, "dataset": "ttd300", "n": n, "layout": L,
        "ED_frac": frac, "instance": f"ttd300_n{n}_L{L}_{frac_tag(frac)}",
        "seed": seed, "status": "error",
        "error": repr(exc)[:300],
    })
    return row

def worker(wid, queue, csv_path, logfile, budget_override):
    log(logfile, f"worker {wid} up (pid {os.getpid()})")
    while True:
        job = queue.get()
        if job is None:
            log(logfile, f"worker {wid} done")
            return
        method, n, L, frac, seed = job
        budget = budget_override if budget_override else BUDGET[n]
        name = f"{method:<3} n={n:<3} L{L} {frac_tag(frac)} seed={seed}"
        try:
            row = run_one(method, n, L, frac, seed, budget)
            append_row(csv_path, row)
            log(logfile, f"w{wid} OK   {name}  obj={row['objective']:>12}  "
                         f"profit={row['profit']:>8}  T/D/R={row['n_truck']}/"
                         f"{row['n_drone']}/{row['n_rendv']}  t={row['time_s']:.0f}s")
        except Exception as exc:
            append_row(csv_path, error_row(method, n, L, frac, seed, exc))
            log(logfile, f"w{wid} FAIL {name}  {exc!r}")
            locked_append(logfile, traceback.format_exc() + "\n")

def summarize(csv_path, logfile):
    if not os.path.exists(csv_path):
        return
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])   # runs,obj,profit,t,ttb
    n_err = 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                n_err += 1
                continue
            k = (r["method"], int(r["n"]))
            a = agg[k]
            a[0] += 1
            a[1] += float(r["objective"])
            a[2] += float(r["profit"])
            a[3] += float(r["time_s"])
            a[4] += float(r["time_to_best_s"])
    log(logfile, "SUMMARY (means over layouts x fracs x seeds)")
    log(logfile, f"{'method':>6} {'n':>4} {'runs':>5} {'mean_obj':>13} "
                 f"{'mean_profit':>12} {'mean_time_s':>12} {'mean_ttb_s':>11}")
    for k in sorted(agg, key=lambda x: (x[0], x[1])):
        c, o, p, t, b = agg[k]
        log(logfile, f"{k[0]:>6} {k[1]:>4} {c:>5} {o/c:>13.1f} "
                     f"{p/c:>12.1f} {t/c:>12.1f} {b/c:>11.1f}")
    if n_err:
        log(logfile, f"!! {n_err} errored runs (status=error rows in CSV)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-run budget (smoke tests only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N pending jobs (smoke tests only)")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--log", default=RUN_LOG)
    ap.add_argument("--summary-only", action="store_true",
                    help="print the summary of --out and exit")
    args = ap.parse_args()

    if args.summary_only:
        summarize(args.out, args.log)
        return

    if not os.path.exists(args.out):
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    jobs = build_jobs()
    done = done_keys(args.out)
    pending = [j for j in jobs if job_key(*j) not in done]
    if args.limit:
        pending = pending[:args.limit]

    est = sum((args.budget or BUDGET[j[1]]) for j in pending)
    log(args.log, f"the benchmark sweep sweep: {len(jobs)} total jobs, {len(done)} already ok, "
                  f"{len(pending)} to run on {args.workers} workers "
                  f"(~{est/args.workers/3600:.1f} h wall)")

    queue = mp.Queue()
    for j in pending:
        queue.put(j)
    for _ in range(args.workers):
        queue.put(None)

    procs = [mp.Process(target=worker,
                        args=(i, queue, args.out, args.log, args.budget))
             for i in range(args.workers)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    log(args.log, f"ALL DONE in {(time.perf_counter()-t0)/3600:.2f} h")
    summarize(args.out, args.log)

if __name__ == "__main__":
    main()
