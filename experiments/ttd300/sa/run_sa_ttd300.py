from __future__ import annotations

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
_paths.use("core", "heuristics")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

REPO = _ROOT
import time

from instance import load_a280, _build_instance          # noqa: E402
from core import problem_from_instance, check_solution, evaluate  # noqa: E402
import ttpd_common as C                                   # noqa: E402
from sa import sa                                         # noqa: E402

OUT_DIR = os.path.join(REPO, "artifacts", "data", "results", "ttd300", "sa")
OUT_CSV = os.path.join(OUT_DIR, "result_sa_15.csv")

SIZES = [10, 20, 30]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]
BUDGET = {10: 120.0, 20: 300.0, 30: 450.0}       # a280 budgets; n=10 bumped up
SEED = 1
ALPHA = 0.97

COLUMNS = [
    "method", "dataset", "n", "layout", "ED_frac", "ED", "d_max", "instance",
    "seed", "status", "objective", "profit", "rental", "arrival",
    "time_s", "time_to_best_s", "n_items", "n_truck", "n_drone", "n_rendv",
    "W_final", "W_capacity", "evals", "reheats",
]

def load_bench_instance(path):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return problem_from_instance(inst), inst

def run_one(n, layout, frac, path, budget):
    P, inst = load_bench_instance(path)
    ED = inst.ED
    assert ED is not None, f"no DRONE ENDURANCE header in {path}"
    t0 = time.perf_counter()
    res = sa(P, time_limit=budget, seed=SEED, alpha=ALPHA)
    wall = time.perf_counter() - t0

    err = check_solution(P, res.truck, res.sorties, res.z)
    if err is not None:
        raise RuntimeError(f"infeasible SA solution: {err}")
    for (L, D, J) in res.sorties:                       # endurance must hold
        reach = P.dist[L][D] + P.dist[D][J]
        assert reach <= ED + 1e-6, f"SORTIE EXCEEDS ED {reach:.2f} > {ED} ({L},{D},{J})"

    det = C.replay_details(inst, res.truck, res.sorties, res.z, a280_path=path)
    own = evaluate(P, res.truck, res.sorties, res.z)
    if own is None or abs(own - det["objective"]) > 1e-6:
        raise RuntimeError(f"evaluator/env mismatch eval={own} env={det['objective']}")

    return {
        "method": "SA", "dataset": "ttd300", "n": n, "layout": layout,
        "ED_frac": frac, "ED": ED, "d_max": round(float(inst.d_max), 1),
        "instance": f"ttd300_n{n}_L{layout}_f{int(round(frac*100)):03d}",
        "seed": SEED, "status": "ok",
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
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        t_start = time.perf_counter()
        for n in SIZES:
            budget = BUDGET[n]
            for frac in FRACS:
                for L in LAYOUTS:
                    tag = f"f{int(round(frac*100)):03d}"
                    path = hub_instance("ttd300", f"ttd300_n{n}_L{L}_{tag}.txt")
                    if not os.path.exists(path):
                        print(f"  MISSING {path} -- skip"); continue
                    row = run_one(n, L, frac, path, budget)
                    rows.append(row)
                    w.writerow(row); fh.flush(); os.fsync(fh.fileno())
                    print(f"  n={n:<3} f{frac:.2f} L{L}  ED={row['ED']:<4} "
                          f"obj={row['objective']:>12}  arr={row['arrival']:>9}  "
                          f"T/D/R={row['n_truck']}/{row['n_drone']}/{row['n_rendv']}  "
                          f"t={row['time_s']:.0f}s")
        dt = time.perf_counter() - t_start
    print(f"\nDONE {len(rows)} runs in {dt/60:.1f} min -> {OUT_CSV}")

if __name__ == "__main__":
    main()
