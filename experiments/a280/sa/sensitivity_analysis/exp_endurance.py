from __future__ import annotations

import argparse
import math
import os
import time

import exp_common as E

# ED as multiples of d_max (the largest pairwise distance in the instance).
ED_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, math.inf]
SIZES = [10]
INST_NO = 1
SEED = 1

COLUMNS = [
    "ED_frac", "ED", "n", "instance", "objective", "rental", "arrival",
    "time_s", "gap_%", "n_truck", "n_drone", "n_rendv", "n_items", "profit",
    "d_max", "time_to_best_s", "reheats",
    "truck_node_ids", "drone_node_ids", "rendezvous_node_ids",
    "drone_arc_node_ids",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(E.HERE, "results", "endurance.csv"))
    cli = ap.parse_args()

    sizes = [int(s) for s in cli.sizes.split(",")]
    rows = []
    t_start = time.perf_counter()

    for n in sizes:
        path = E.bench_path(n, INST_NO)
        if not os.path.exists(path):
            print(f"  MISSING {path} -- skip")
            continue
        _, inst = E.load_bench_instance(path)
        d_max = float(inst.dist.max())
        budget = cli.budget if cli.budget is not None else E.TIME_BUDGET[n]

        for frac in ED_FRACS:
            if frac == 0.0:
                kw = {"drone_disabled": True}   # ED=0 -> no sortie fits
                ED_val = 0.0
            elif math.isinf(frac):
                kw = {}                          # unbounded (base model)
                ED_val = math.inf
            else:
                ED_val = frac * d_max
                kw = {"ED": ED_val}
            r = E.run_config(path, budget, seed=cli.seed, **kw)
            r["ED_frac"] = frac
            r["ED"] = ("inf" if math.isinf(ED_val) else round(ED_val, 2))
            r["n"] = n
            r["instance"] = f"bench_{n}_{INST_NO}"
            r["d_max"] = round(d_max, 2)
            r["gap_%"] = E.gap_pct(n, INST_NO, r["objective"]) if math.isinf(frac) else ""
            rows.append(r)
            print(f"  n={n} ED_frac={frac!s:<5} ED={r['ED']:<10} "
                  f"obj={r['objective']:>14} rent={r['rental']:>13} "
                  f"time={r['arrival']:>11} "
                  f"T/D/R={r['n_truck']}/{r['n_drone']}/{r['n_rendv']}")
            E.write_csv(cli.out, COLUMNS, rows)

    print(f"\nDONE {len(rows)} runs in {(time.perf_counter()-t_start)/60:.1f} min "
          f"-> {cli.out}")


if __name__ == "__main__":
    main()
