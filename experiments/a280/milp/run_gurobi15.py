import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("exact")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import csv
import argparse

from bench_solver import solve_instance, log  # noqa: E402

N_CITIES = 15
DEFAULT_SEEDS = [2, 3, 4, 5]   # seed 1 is shipped for reference but skipped

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

METHOD = "MILP"

def result_row(instance, seed, r):
    return {
        "method": METHOD, "n": r["n"], "seed": seed, "instance": instance,
        "status": r["status"], "objective": r["objective"],
        "gap_%": r["mip_gap"], "time_s": r["total_s"],
        "n_items": r["n_items"], "n_truck": r["n_truck"],
        "n_drone": r["n_drone"], "n_rendv": r["n_rendv"],
        "profit": r["profit"], "rental": r["rental"], "arrival": r["arrival"],
        "W_final": r["W_final"], "W_capacity": r["W_capacity"],
        "W_actual": r["W_actual"],
        "n_nodes_explored": r["n_nodes"],
        "build_s": r["build_s"], "solve_s": r["solve_s"],
        "collected_node_ids": r["collected_node_ids"],
        "truck_node_ids": r["truck_node_ids"],
        "drone_node_ids": r["drone_node_ids"],
        "rendezvous_node_ids": r["rendezvous_node_ids"],
        "truck_route_node_ids": r["truck_route_node_ids"],
        "drone_arc_node_ids": r["drone_arc_node_ids"],
        "W_sequence": r["W_sequence"],
    }

def append_row_fsync(path, row):
    """Append one row to the CSV, write header if new, then fsync to disk."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINALE_COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in FINALE_COLUMNS})
        f.flush()
        os.fsync(f.fileno())

def main():
    ap = argparse.ArgumentParser(description="batch TTP-D MILP runner, n=15.")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="Time limit per instance, seconds (default 15).")
    ap.add_argument("--seed-list", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="Comma-separated seeds to run (default 2,3,4,5).")
    ap.add_argument("--threads", type=int, default=2,
                    help="Gurobi threads per solve (default 2).")
    ap.add_argument("--mip-gap", type=float, default=0.0001,
                    help="Gurobi MIPGap (default 0.0001 = 0.01%%).")
    ap.add_argument("--out", default=os.path.join(HERE, "results_gurobi15.csv"))
    ap.add_argument("--logfile", default=os.path.join(HERE, "run_gurobi15.log"))
    ap.add_argument("--sol-out", default=None,
                    help="Write best incumbent to this .sol on every new best "
                         "(enables resume). Default: <out_dir>/seed<k>.sol.")
    ap.add_argument("--warmstart-sol", default=None,
                    help="Resume from this .sol (a previous run's incumbent). "
                         "Default: same path as --sol-out if it exists.")
    cli = ap.parse_args()

    # Pin Gurobi thread count for every solve in this process.
    os.environ["GRB_THREADS"] = str(cli.threads)

    seeds = [int(s) for s in cli.seed_list.split(",") if s.strip()]
    out_path = os.path.abspath(cli.out)

    # mirror stdout to the logfile
    logf = open(os.path.abspath(cli.logfile), "a", buffering=1)
    _orig = sys.stdout

    class _Tee:
        def write(self, s):
            _orig.write(s)
            logf.write(s)
        def flush(self):
            _orig.flush()
            logf.flush()
    sys.stdout = _Tee()

    log("=" * 70)
    log(f"GUROBI n={N_CITIES}  seeds={seeds}  limit={cli.seconds}s/inst  "
        f"threads={cli.threads}  mip_gap={cli.mip_gap}")
    log(f"out={out_path}")
    log("=" * 70)

    for seed in seeds:
        inst_file = hub_instance("a280", f"bench_{N_CITIES}_{seed}.txt")
        inst_name = f"bench_{N_CITIES}_{seed}"
        if not os.path.exists(inst_file):
            log(f"  MISSING {inst_file} -- skipping")
            continue

        sol_out = cli.sol_out or os.path.join(HERE, f"seed{seed}.sol")
        warm = cli.warmstart_sol
        if warm is None and os.path.exists(sol_out) and os.path.getsize(sol_out) > 0:
            warm = sol_out
        r = solve_instance(inst_file, time_limit=cli.seconds, mip_gap=cli.mip_gap,
                           warmstart_sol=warm, sol_out=sol_out)
        row = result_row(inst_name, seed, r)
        log(f"  DONE {inst_name}: status={r['status']} obj={r['objective']} "
            f"gap={r['mip_gap']}% t={r['total_s']}s items={r['n_items']}")
        append_row_fsync(out_path, row)   # crash-safe, per-row

    log("ALL DONE.")

if __name__ == "__main__":
    main()
