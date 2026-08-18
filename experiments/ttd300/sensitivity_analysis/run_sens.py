from __future__ import annotations

import argparse
import csv
import fcntl
import io
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SIZES = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]
SEED = 1
WORKERS = 4

BUDGET = {10: 120.0, 20: 300.0, 30: 450.0, 40: 600.0,
          50: 750.0, 75: 975.0, 100: 1200.0}

SPEED_FACTORS = [0.0, 0.5, 1.0, 2.0, 3.0]
SENS_SWEEPS = {
    # base R = 50 in-file; {1, R/4, R/2, R, 2R, 4R}, mirroring the a280 sweep
    "R":      [1.0, 12.5, 25.0, 50.0, 100.0, 200.0],
    "Wcap":   [0.25, 0.5, 1.0, 2.0, 4.0],      # multiples of in-file capacity
    "vmin":   [0.05, 0.10, 0.25, 0.50, 0.90],
    "vDfact": [0.5, 1.0, 2.0, 3.0, 4.0],
}
EXPS = ["multiitem", "speed", "sens"]

RUN_LOG = os.path.join(HERE, "run.log")

_SHARED = ["dataset", "n", "layout", "ED_frac", "ED", "d_max", "instance",
           "seed", "status", "objective", "profit", "rental", "arrival",
           "time_s", "time_to_best_s", "n_items", "n_truck", "n_drone",
           "n_rendv", "W_final", "W_capacity", "evals", "reheats", "error"]

COLUMNS = {
    "multiitem": ["mode"] + _SHARED + [
        "cities_collected", "total_items", "avg_items_per_city",
        "multi_pick_cities", "max_items_one_city", "items_dist",
        "W_collected", "W_actual", "cap_util_%",
        "n_truck_cities", "n_drone_cities"],
    "speed": ["factor"] + _SHARED + ["vD"],
    "sens":  ["param", "value"] + _SHARED + ["R", "vmin", "vD"],
}


_FILE_NAME = {"multiitem": "multiitem", "speed": "speed",
              "sens": "sensitivity"}


def csv_path(exp, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(HERE, f"results_{_FILE_NAME[exp]}{suffix}.csv")


def _vstr(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def frac_tag(frac):
    return f"f{int(round(frac * 100)):03d}"


def instance_name(n, L, frac):
    return f"ttd300_n{n}_L{L}_{frac_tag(frac)}"


def build_jobs(exps, sizes, fracs, layouts):
    jobs = []
    for n in sizes:
        for frac in fracs:
            for L in layouts:
                if "multiitem" in exps:
                    for mode in ("SI", "MI"):
                        jobs.append(("multiitem", n, L, frac, mode))
                if "speed" in exps:
                    for f in SPEED_FACTORS:
                        jobs.append(("speed", n, L, frac, f))
                if "sens" in exps:
                    jobs.append(("sens", n, L, frac, ("baseline", "-")))
                    for param, values in SENS_SWEEPS.items():
                        for v in values:
                            jobs.append(("sens", n, L, frac, (param, v)))
    # cheapest sizes first, experiments interleaved at equal n
    jobs.sort(key=lambda j: (BUDGET[j[1]], j[1], j[3], j[2], j[0], str(j[4])))
    return jobs


def job_key(exp, n, L, frac, setting):
    inst = instance_name(n, L, frac)
    if exp == "multiitem":
        return (exp, inst, setting, str(SEED))
    if exp == "speed":
        return (exp, inst, _vstr(setting), str(SEED))
    param, value = setting
    return (exp, inst, param, _vstr(value), str(SEED))


def done_keys(exps, tag=""):
    """Keys of rows already ok in the per-experiment CSVs; errored rows are
    retried on resume."""
    done = set()
    for exp in exps:
        path = csv_path(exp, tag)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                if exp == "multiitem":
                    done.add((exp, row["instance"], row["mode"], row["seed"]))
                elif exp == "speed":
                    done.add((exp, row["instance"],
                              _vstr(float(row["factor"])), row["seed"]))
                else:
                    done.add((exp, row["instance"], row["param"],
                              row["value"], row["seed"]))
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


def append_row(exp, tag, row):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS[exp], extrasaction="ignore")
    w.writerow({c: row.get(c, "") for c in COLUMNS[exp]})
    locked_append(csv_path(exp, tag), buf.getvalue())

def run_one(exp, n, L, frac, setting, budget):
    import exp_common as E
    import exp_multiitem as M

    path = E.bench_path(n, L, frac)
    base = {
        "dataset": "ttd300", "n": n, "layout": L, "ED_frac": frac,
        "instance": instance_name(n, L, frac), "seed": SEED, "status": "ok",
        "error": "",
    }

    if exp == "multiitem":
        if setting == "SI":
            r = E.run_config(path, budget, seed=SEED)
        else:
            r = M.run_mi(path, budget, seed=SEED)
            r["items_dist"] = str(r.get("items_dist", ""))
        return {**base, **r, "mode": setting}

    if exp == "speed":
        f = setting
        kw = {"drone_disabled": True} if f == 0.0 else {"vD_factor": f}
        r = E.run_config(path, budget, seed=SEED, **kw)
        return {**base, **r, "factor": _vstr(f)}

    # sens
    param, value = setting
    kw = {}
    if param == "R":
        kw = {"R": value}
    elif param == "Wcap":
        kw = {"Wcap_mult": value}
    elif param == "vmin":
        kw = {"vmin": value}
    elif param == "vDfact":
        kw = {"vD_factor": value}
    r = E.run_config(path, budget, seed=SEED, **kw)
    return {**base, **r, "param": param, "value": _vstr(value)}


def error_row(exp, n, L, frac, setting, exc):
    row = {c: "" for c in COLUMNS[exp]}
    row.update({
        "dataset": "ttd300", "n": n, "layout": L, "ED_frac": frac,
        "instance": instance_name(n, L, frac), "seed": SEED,
        "status": "error", "error": repr(exc)[:300],
    })
    if exp == "multiitem":
        row["mode"] = setting
    elif exp == "speed":
        row["factor"] = _vstr(setting)
    else:
        row["param"], row["value"] = setting[0], _vstr(setting[1])
    return row


def _setting_str(exp, setting):
    if exp == "multiitem":
        return setting
    if exp == "speed":
        return f"phi={_vstr(setting)}"
    return f"{setting[0]}={_vstr(setting[1])}"


def worker(wid, queue, tag, logfile, budget_override):
    log(logfile, f"worker {wid} up (pid {os.getpid()})")
    while True:
        job = queue.get()
        if job is None:
            log(logfile, f"worker {wid} done")
            return
        exp, n, L, frac, setting = job
        budget = budget_override if budget_override else BUDGET[n]
        name = (f"{exp:<9} n={n:<3} L{L} {frac_tag(frac)} "
                f"{_setting_str(exp, setting)}")
        try:
            row = run_one(exp, n, L, frac, setting, budget)
            append_row(exp, tag, row)
            log(logfile, f"w{wid} OK   {name:<44} obj={row['objective']:>12} "
                         f"t={float(row['time_s']):.0f}s")
        except Exception as exc:
            append_row(exp, tag, error_row(exp, n, L, frac, setting, exc))
            log(logfile, f"w{wid} FAIL {name}  {exc!r}")
            locked_append(logfile, traceback.format_exc() + "\n")

def summarize(exps, tag, logfile):
    from collections import defaultdict
    for exp in exps:
        path = csv_path(exp, tag)
        if not os.path.exists(path):
            continue
        agg = defaultdict(lambda: [0, 0.0])
        n_err = 0
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] != "ok":
                    n_err += 1
                    continue
                a = agg[(int(r["n"]), float(r["ED_frac"]))]
                a[0] += 1
                a[1] += float(r["objective"])
        log(logfile, f"===== {exp}: mean objective over rows per (n, ED_frac) =====")
        for k in sorted(agg):
            c, o = agg[k]
            log(logfile, f"  n={k[0]:>4} f={k[1]:.2f}  rows={c:>4}  "
                         f"mean_obj={o / c:>13.1f}")
        if n_err:
            log(logfile, f"!! {exp}: {n_err} errored rows (status=error)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--exps", default=",".join(EXPS),
                    help="subset of experiments: multiitem,speed,sens")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--fracs", default=",".join(str(f) for f in FRACS))
    ap.add_argument("--layouts", default=",".join(str(l) for l in LAYOUTS))
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-run budget (smoke tests only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N pending jobs (smoke tests only)")
    ap.add_argument("--tag", default="",
                    help="suffix for the results CSVs (e.g. 'smoke')")
    ap.add_argument("--log", default=RUN_LOG)
    ap.add_argument("--summary-only", action="store_true",
                    help="print the summary of the results CSVs and exit")
    args = ap.parse_args()

    exps = [e for e in args.exps.split(",") if e in EXPS]
    sizes = [int(s) for s in args.sizes.split(",")]
    fracs = [float(f) for f in args.fracs.split(",")]
    layouts = [int(l) for l in args.layouts.split(",")]

    if args.summary_only:
        summarize(exps, args.tag, args.log)
        return

    for exp in exps:
        path = csv_path(exp, args.tag)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS[exp]).writeheader()

    jobs = build_jobs(exps, sizes, fracs, layouts)
    done = done_keys(exps, args.tag)
    pending = [j for j in jobs if job_key(*j) not in done]
    if args.limit:
        pending = pending[:args.limit]

    est = sum((args.budget or BUDGET[j[1]]) for j in pending)
    log(args.log, f"the sensitivity analysis sweep: {len(jobs)} total jobs, {len(done)} already ok, "
                  f"{len(pending)} to run on {args.workers} workers "
                  f"(~{est / args.workers / 3600:.1f} h wall)")

    queue = mp.Queue()
    for j in pending:
        queue.put(j)
    for _ in range(args.workers):
        queue.put(None)

    procs = [mp.Process(target=worker,
                        args=(i, queue, args.tag, args.log, args.budget))
             for i in range(args.workers)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    log(args.log, f"ALL DONE in {(time.perf_counter() - t0) / 3600:.2f} h")
    summarize(exps, args.tag, args.log)


if __name__ == "__main__":
    main()
