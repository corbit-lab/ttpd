from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ttd300_io import FULL_BUDGET, SIZES                 

PY = sys.executable
N_BENCH = 20                     

PLAN = {
    10:  dict(sample=200, seeds=2, time_scale=1.0),
    20:  dict(sample=150, seeds=2, time_scale=0.5),
    30:  dict(sample=120, seeds=1, time_scale=0.5),
    40:  dict(sample=100, seeds=1, time_scale=0.4),
    50:  dict(sample=100, seeds=1, time_scale=1.0 / 3.0),
    75:  dict(sample=80,  seeds=1, time_scale=0.25),
    100: dict(sample=80,  seeds=1, time_scale=0.2),
}
SMOKE_PLAN = {n: dict(sample=1, seeds=1, time_scale=0.002) for n in SIZES}


def log(run_log, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(run_log, "a") as f:
        f.write(line + "\n")


def sh(cmd, log_path, threads):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)


def gen_core_seconds(n, plan):
    p = plan[n]
    return (N_BENCH + p["sample"]) * p["seeds"] * FULL_BUDGET[n] * p["time_scale"]


def print_estimate(sizes, plan, jobs):
    total = 0.0
    print(f"{'n':>4} {'instances':>10} {'seeds':>6} {'budget_s':>9} "
          f"{'solves':>7} {'core-h':>7}")
    for n in sizes:
        p = plan[n]
        solves = (N_BENCH + p["sample"]) * p["seeds"]
        budget = FULL_BUDGET[n] * p["time_scale"]
        cost = gen_core_seconds(n, plan)
        total += cost
        print(f"{n:>4} {N_BENCH}+{p['sample']:<7} {p['seeds']:>6} "
              f"{budget:>9.1f} {solves:>7} {cost/3600:>7.2f}")
    print(f"\n[gen] total ~{total/3600:.1f} core-h "
          f"(~{total/3600/max(jobs,1):.1f} h wall on {jobs} workers); "
          f"BC chain wall time is extra (logged per size).")


def stage_gen(sizes, plan, dirs, args):
    for n in sizes:
        marker = os.path.join(dirs["data"], f"n{n}.done")
        out = os.path.join(dirs["data"], f"n{n}.jsonl")
        if os.path.exists(marker):
            log(dirs["run_log"], f"gen  n={n:<3} SKIP (done)")
            continue
        p = plan[n]
        cmd = [PY, os.path.join(HERE, "gen_dataset.py"),
               "--sizes", str(n), "--bench",
               "--sample", str(p["sample"]), "--seeds", str(p["seeds"]),
               "--time-scale", f"{p['time_scale']:.6f}",
               "--jobs", str(args.gen_jobs), "--resume", "--out", out]
        t0 = time.perf_counter()
        log(dirs["run_log"], f"gen  n={n:<3} start "
            f"({N_BENCH}+{p['sample']} inst x {p['seeds']} seeds, "
            f"budget {FULL_BUDGET[n] * p['time_scale']:.1f}s, "
            f"jobs={args.gen_jobs})")
        rc = sh(cmd, os.path.join(dirs["logs"], f"gen_n{n}.log"), threads=1)
        dt = (time.perf_counter() - t0) / 60
        if rc != 0:
            log(dirs["run_log"], f"gen  n={n:<3} FAIL rc={rc} after {dt:.1f} min "
                f"-- fix and relaunch (resumes)")
            return False
        open(marker, "w").close()
        log(dirs["run_log"], f"gen  n={n:<3} OK   ({dt:.1f} min)")
    return True


def stage_train(sizes, dirs, args):
    prev = None
    for n in SIZES:                       # walk the full chain to keep warm starts
        ckpt = os.path.join(dirs["weights"], f"n{n}.pt")
        marker = ckpt + ".done"
        if n not in sizes:
            if os.path.exists(marker):
                prev = ckpt
            continue
        if os.path.exists(marker):
            log(dirs["run_log"], f"train n={n:<3} SKIP (done)")
            prev = ckpt
            continue
        data = os.path.join(dirs["data"], f"n{n}.jsonl")
        if not os.path.exists(data):
            log(dirs["run_log"], f"train n={n:<3} FAIL: {data} missing (run gen)")
            return False
        cmd = [PY, os.path.join(HERE, "bc_train.py"),
               "--data", data, "--out", ckpt, "--eval"]
        if args.smoke:
            cmd += ["--epochs", "2", "--n-aug", "1", "--patience", "0"]
        if prev is not None:
            cmd += ["--init-from", prev]
        t0 = time.perf_counter()
        log(dirs["run_log"], f"train n={n:<3} start"
            + (f" (init-from {os.path.basename(prev)})" if prev else " (from scratch)"))
        rc = sh(cmd, os.path.join(dirs["logs"], f"train_n{n}.log"),
                threads=args.train_threads)
        dt = (time.perf_counter() - t0) / 60
        if rc != 0:
            log(dirs["run_log"], f"train n={n:<3} FAIL rc={rc} after {dt:.1f} min")
            return False
        open(marker, "w").close()
        prev = ckpt
        log(dirs["run_log"], f"train n={n:<3} OK   ({dt:.1f} min) -> {ckpt}")
    return True


def stage_pareto(sizes, dirs, args):
    if args.smoke:
        sizes = sizes[:1]                 # one cheap cell is enough to rehearse
    cmd = [PY, os.path.join(HERE, "run_sa_lisa_pareto.py"),
           "--sizes", *map(str, sizes)]
    if args.smoke:
        cmd += ["--fracs", "5%", "--layouts", "1", "--ed-fracs", "0.25",
                "--beam-width", "16", "--n-aug", "2",
                "--out", os.path.join(dirs["base"], "sa_lisa_smoke.csv"),
                "--weights-file", os.path.join(dirs["weights"], "n10.pt")]
    t0 = time.perf_counter()
    log(dirs["run_log"], f"pareto start (sizes {sizes})")
    rc = sh(cmd, os.path.join(dirs["logs"], "pareto.log"),
            threads=args.train_threads)
    dt = (time.perf_counter() - t0) / 60
    log(dirs["run_log"], f"pareto {'OK' if rc == 0 else f'FAIL rc={rc}'} "
        f"({dt:.1f} min)")
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=str, default="gen,train",
                    help="comma list from {gen,train,pareto}")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--gen-jobs", type=int, default=os.cpu_count(),
                    help="parallel SA solves in the gen stage")
    ap.add_argument("--train-threads", type=int, default=os.cpu_count(),
                    help="torch/BLAS threads for BC training")
    ap.add_argument("--estimate", action="store_true",
                    help="print the dataset plan + cost and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny rehearsal of every stage into smoke/ "
                         "(~minutes; does not touch the real data/weights)")
    args = ap.parse_args()

    plan = SMOKE_PLAN if args.smoke else PLAN
    sizes = [n for n in args.sizes if n in plan]
    if args.estimate:
        print_estimate(sizes, plan, args.gen_jobs)
        return

    base = os.path.join(HERE, "smoke") if args.smoke else HERE
    dirs = {"base": base,
            "data": os.path.join(base, "data"),
            "weights": os.path.join(base, "weights"),
            "logs": os.path.join(base, "logs"),
            "run_log": os.path.join(base, "run.log")}
    for k in ("data", "weights", "logs"):
        os.makedirs(dirs[k], exist_ok=True)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.smoke and "pareto" not in stages:
        stages.append("pareto")           # a rehearsal should cover everything
    log(dirs["run_log"], f"run_lisa start: stages={stages} sizes={sizes} "
        f"gen_jobs={args.gen_jobs} train_threads={args.train_threads}"
        + (" [SMOKE]" if args.smoke else ""))
    if "gen" in stages:
        print_estimate(sizes, plan, args.gen_jobs)
        if not stage_gen(sizes, plan, dirs, args):
            sys.exit(1)
    if "train" in stages:
        if not stage_train(sizes, dirs, args):
            sys.exit(1)
    if "pareto" in stages:
        if not stage_pareto(sizes, dirs, args):
            sys.exit(1)
    log(dirs["run_log"], "run_lisa ALL DONE")


if __name__ == "__main__":
    main()
