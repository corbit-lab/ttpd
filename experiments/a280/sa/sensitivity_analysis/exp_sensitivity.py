
from __future__ import annotations

import argparse
import os
import time

import exp_common as E

INST_NO = 1
SEED = 1

R_BASE = 72.70
WCAP_BASE = None   # read from the instance file at runtime
VMIN_BASE = 0.10

SWEEPS = {
    "R":      [1.0, 18.18, 36.35, 72.70, 145.40, 290.80],
    "Wcap":   [0.25, 0.5, 1.0, 2.0, 4.0],          # multiples of file capacity
    "vmin":   [0.05, 0.10, 0.25, 0.50, 0.90],
    "vDfact": [0.5, 1.0, 2.0, 3.0, 4.0],
}

COLUMNS = [
    "param", "value", "objective", "rental", "arrival", "time_s", "gap_%",
    "n_truck", "n_drone", "n_rendv", "n_items", "profit",
    "R", "Wcap", "vmin", "vD", "W_final",
    "truck_node_ids", "drone_node_ids", "rendezvous_node_ids",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="instance size (default 20)")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--params", default=",".join(SWEEPS),
                    help="which params to sweep (comma-separated subset)")
    ap.add_argument("--out", default=os.path.join(E.HERE, "results", "sensitivity.csv"))
    cli = ap.parse_args()

    N = cli.n
    path = E.bench_path(N, INST_NO)
    _, inst = E.load_bench_instance(path)
    wcap_base = float(inst.W)
    budget = cli.budget if cli.budget is not None else E.TIME_BUDGET[N]
    params = [p for p in cli.params.split(",") if p in SWEEPS]
    rows = []
    t_start = time.perf_counter()

    def emit(param, value, **overrides):
        r = E.run_config(path, budget, seed=cli.seed, **overrides)
        r["param"] = param
        r["value"] = value
        r["Wcap"] = r["W_capacity"]
        r["gap_%"] = ""
        rows.append(r)
        print(f"  {param:<7} {str(value):<10} obj={r['objective']:>14} "
              f"rent={r['rental']:>13} time={r['arrival']:>11} "
              f"T/D/R={r['n_truck']}/{r['n_drone']}/{r['n_rendv']}")
        E.write_csv(cli.out, COLUMNS, rows)

    # baseline reference row first
    emit("baseline", "-")

    for param in params:
        for v in SWEEPS[param]:
            if param == "R":
                emit("R", v, R=v)
            elif param == "Wcap":
                emit("Wcap", f"{v}x", Wcap=v * wcap_base)
            elif param == "vmin":
                emit("vmin", v, vmin=v)
            elif param == "vDfact":
                emit("vDfact", v, vD_factor=v)

    print(f"\nDONE {len(rows)} runs in {(time.perf_counter()-t_start)/60:.1f} min "
          f"-> {cli.out}")


if __name__ == "__main__":
    main()
