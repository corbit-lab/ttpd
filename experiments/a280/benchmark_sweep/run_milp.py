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

SEEDS = [1, 2, 3, 4, 5]

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

def write_csv(path, rows, append=False):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if (append and file_exists) else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINALE_COLUMNS, extrasaction="ignore")
        if mode == "w" or not file_exists:
            w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in FINALE_COLUMNS})

def main():
    ap = argparse.ArgumentParser(description="Run TTP-D MILP over Bench/ instances.")
    ap.add_argument("--n", type=int, nargs="+", required=True,
                    help="n values to run (instance files must already exist).")
    ap.add_argument("--hours", type=float, default=1.0,
                    help="Time limit per instance, hours (default 1).")
    ap.add_argument("--mip-gap", type=float, default=0.0001,
                    help="Gurobi MIPGap (default 0.0001 = 0.01%%).")
    ap.add_argument("--finale", default=os.path.join(HERE, os.pardir, "finale.csv"))
    cli = ap.parse_args()

    finale_path = os.path.abspath(cli.finale)
    time_limit = cli.hours * 3600

    for n in cli.n:
        per_n_rows = []
        log("=" * 70)
        log(f"RUNNING MILP n={n}  ({len(SEEDS)} instances, {cli.hours}h limit each, "
            f"mip_gap={cli.mip_gap})")
        log("=" * 70)
        for seed in SEEDS:
            inst_file = hub_instance("a280", f"bench_{n}_{seed}.txt")
            inst_name = f"bench_{n}_{seed}"
            if not os.path.exists(inst_file):
                log(f"  MISSING {inst_file} -- skipping")
                continue
            r = solve_instance(inst_file, time_limit=time_limit,
                               mip_gap=cli.mip_gap)
            row = result_row(inst_name, seed, r)
            per_n_rows.append(row)
            log(f"  DONE {inst_name}: status={r['status']} obj={r['objective']} "
                f"gap={r['mip_gap']}% t={r['total_s']}s items={r['n_items']}")
            # append each run to finale.csv immediately (crash-safe)
            write_csv(finale_path, [row], append=True)

        milp_path = os.path.join(HERE, f"MILP{n}.csv")
        write_csv(milp_path, per_n_rows, append=False)
        log(f"  wrote {milp_path}  ({len(per_n_rows)} runs)")
        log(f"  appended n={n} block to {finale_path}")

    log("ALL DONE.")

if __name__ == "__main__":
    main()
