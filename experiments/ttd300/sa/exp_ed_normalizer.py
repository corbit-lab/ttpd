import csv
import math
import os
import statistics as st
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

from instance import load_a280, _build_instance          # noqa: E402
from core import problem_from_instance, check_solution, evaluate  # noqa: E402
from sa import sa                                         # noqa: E402

OUT = os.path.join(_ROOT, "artifacts", "data", "results", "ttd300", "sa",
                   "exp_ed_normalizer.csv")
DIAG = math.ceil(math.hypot(300, 300))                    # 425
SIZES = [10, 20, 30]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]
BUDGET = 15.0
SEED = 1

def base_instance(n, L):
    d = load_a280(hub_instance("ttd300", f"ttd300_n{n}_L{L}.txt"))   # reference (no ED)
    best = d["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    return _build_instance(d, chosen, depot_id=1, R=None, scale_capacity=False)

def one(inst, ED):
    inst.ED = float(ED)
    P = problem_from_instance(inst)
    res = sa(P, time_limit=BUDGET, seed=SEED, alpha=0.97)
    assert check_solution(P, res.truck, res.sorties, res.z) is None
    for (L, D, J) in res.sorties:
        assert P.dist[L][D] + P.dist[D][J] <= ED + 1e-6
    n_rendv = len(res.sorties)
    n_drone = sum(1 for (L, D, J) in res.sorties if res.z[D] == 1)
    obj = evaluate(P, res.truck, res.sorties, res.z)
    return n_drone, n_rendv, obj, inst.n

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    print(f"DIAG(fixed)={DIAG}   budget={BUDGET}s/run\n")
    for norm in ("diagonal", "diameter"):
        print(f"=== normalizer: {norm} ===")
        for n in SIZES:
            for frac in FRACS:
                nds, nrs, objs, eds = [], [], [], []
                for L in LAYOUTS:
                    inst = base_instance(n, L)
                    dm = float(inst.dist.max())
                    ED = round(frac * (DIAG if norm == "diagonal" else dm))
                    nd, nr, obj, _ = one(inst, ED)
                    nds.append(nd); nrs.append(nr); objs.append(obj); eds.append(ED)
                row = {
                    "normalizer": norm, "n": n, "frac": frac,
                    "ED_mean": round(st.mean(eds), 1),
                    "n_drone_mean": round(st.mean(nds), 2),
                    "n_rendv_mean": round(st.mean(nrs), 2),
                    "obj_mean": round(st.mean(objs), 1),
                    "drone_util_pct": round(100 * st.mean(nds) / n, 1),
                }
                rows.append(row)
                print(f"  n={n:<3} frac={frac:.2f}  ED~{row['ED_mean']:<6} "
                      f"drone={row['n_drone_mean']:<5} rdv={row['n_rendv_mean']:<5} "
                      f"util={row['drone_util_pct']:>4}%  obj={row['obj_mean']:>11}")
        print()
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {OUT}")

if __name__ == "__main__":
    main()
