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

HERE = os.path.dirname(os.path.abspath(__file__))
import time

from instance import load_a280, _build_instance          # noqa: E402
from core import problem_from_instance, check_solution, evaluate  # noqa: E402
import ttpd_common as C                                   # noqa: E402
from sa import sa                                          # noqa: E402

TIME_BUDGET = {
    10: 120.0, 20: 300.0, 30: 450.0, 40: 600.0,
    50: 750.0, 75: 975.0, 100: 1200.0,
}
SA_ALPHA = 0.97

SIZES = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1, 2, 3, 4, 5]
FRACS = [0.25, 0.50, 0.75, 1.00]

def frac_tag(frac: float) -> str:
    return f"f{int(round(frac * 100)):03d}"

def bench_path(n: int, layout: int, frac: float) -> str:
    return hub_instance("ttd300", f"ttd300_n{n}_L{layout}_{frac_tag(frac)}.txt")

def instance_name(n: int, layout: int, frac: float) -> str:
    return f"ttd300_n{n}_L{layout}_{frac_tag(frac)}"

def load_bench_instance(path: str):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return problem_from_instance(inst), inst

def build_problem(path: str, *, vD_factor=None, R=None, Wcap_mult=None,
                  vmin=None, drone_disabled=False):
    _, inst = load_bench_instance(path)
    assert inst.ED is not None, f"no DRONE ENDURANCE header in {path}"
    if drone_disabled:
        inst.v_D = inst.v_max * 1e-6
    elif vD_factor is not None:
        inst.v_D = inst.v_max * float(vD_factor)
    if R is not None:
        inst.R = float(R)
    if Wcap_mult is not None:
        inst.W = float(Wcap_mult) * float(inst.W)
    if vmin is not None:
        inst.v_min = float(vmin)
    P = problem_from_instance(inst)
    return P, inst

def run_config(path: str, budget: float, seed: int = 1, **overrides) -> dict:
    P, inst = build_problem(path, **overrides)
    ED = float(inst.ED)
    t0 = time.perf_counter()
    res = sa(P, time_limit=budget, seed=seed, alpha=SA_ALPHA)
    wall = time.perf_counter() - t0

    err = check_solution(P, res.truck, res.sorties, res.z)
    if err is not None:
        raise RuntimeError(f"infeasible SA solution: {err}")
    for (L, D, J) in res.sorties:                       # endurance must hold
        reach = P.dist[L][D] + P.dist[D][J]
        if reach > ED + 1e-6:
            raise RuntimeError(
                f"SORTIE EXCEEDS ED {reach:.2f} > {ED} ({L},{D},{J})")
    det = C.replay_details(inst, res.truck, res.sorties, res.z, a280_path=path)
    own = evaluate(P, res.truck, res.sorties, res.z)
    if own is None or abs(own - det["objective"]) > 1e-6:
        raise RuntimeError(f"evaluator/env mismatch eval={own} env={det['objective']}")

    return {
        "objective": round(det["objective"], 4),
        "profit": round(det["profit"], 4),
        "rental": round(det["rental"], 4),
        "arrival": round(det["arrival"], 4),
        "time_s": round(wall, 4),
        "time_to_best_s": round(getattr(res, "time_to_best", 0.0), 4),
        "evals": getattr(res, "evals", ""),
        "reheats": getattr(res, "reheats", ""),
        "n_items": det["n_items"], "n_truck": det["n_truck"],
        "n_drone": det["n_drone"], "n_rendv": det["n_rendv"],
        "W_final": round(det["W_final"], 4), "W_capacity": round(float(inst.W), 4),
        "W_actual": round(float(sum(inst.weights)), 4),
        "vD": round(float(inst.v_D), 6), "vmin": float(inst.v_min),
        "R": float(inst.R), "ED": round(ED, 4),
        "d_max": round(float(inst.d_max), 1),
    }

def write_csv(path: str, columns, rows):

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
