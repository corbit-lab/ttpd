
from __future__ import annotations

import argparse
import os
import time

import exp_common as E

FACTORS = [0.0, 0.5, 1.0, 2.0, 3.0]
SIZES = [1, 5, 10, 20, 30, 40, 50, 100]
INST_NO = 1
SEED = 1

COLUMNS = [
    "factor", "n", "instance", "objective", "rental", "arrival", "time_s",
    "gap_%", "n_truck", "n_drone", "n_rendv", "n_items", "profit",
    "time_to_best_s", "reheats", "evals", "vD", "W_final", "W_capacity",
    "truck_node_ids", "drone_node_ids", "rendezvous_node_ids",
    "truck_route_node_ids", "drone_arc_node_ids",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS))
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-run budget (s); default = per-size TIME_BUDGET")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(E.HERE, "results", "speed.csv"))
    cli = ap.parse_args()

    sizes = [int(s) for s in cli.sizes.split(",")]
    factors = [float(f) for f in cli.factors.split(",")]
    rows = []
    t_start = time.perf_counter()

    for f in factors:
        for n in sizes:
            path = E.bench_path(n, INST_NO)
            if not os.path.exists(path):
                print(f"  MISSING {path} -- skip")
                continue
            budget = cli.budget if cli.budget is not None else E.TIME_BUDGET[n]
            kw = {"drone_disabled": True} if f == 0.0 else {"vD_factor": f}
            r = E.run_config(path, budget, seed=cli.seed, **kw)
            r["factor"] = f
            r["n"] = n
            r["instance"] = f"bench_{n}_{INST_NO}"
            r["gap_%"] = E.gap_pct(n, INST_NO, r["objective"]) if f == 2.0 else ""
            rows.append(r)
            print(f"  factor={f:<4} n={n:<4} obj={r['objective']:>14} "
                  f"rent={r['rental']:>13} time={r['arrival']:>11} "
                  f"T/D/R={r['n_truck']}/{r['n_drone']}/{r['n_rendv']} "
                  f"({r['time_s']}s wall)")
            E.write_csv(cli.out, COLUMNS, rows)  # incremental, crash-safe

    print(f"\nDONE {len(rows)} runs in {(time.perf_counter()-t_start)/60:.1f} min "
          f"-> {cli.out}")


if __name__ == "__main__":
    main()
