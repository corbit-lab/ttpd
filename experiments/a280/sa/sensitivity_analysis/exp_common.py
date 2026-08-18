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

from ttpd.hub import ensure_local  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import time

batch = os.path.dirname(HERE)

from instance import load_a280, _build_instance          # noqa: E402
from core import problem_from_instance, check_solution, evaluate  # noqa: E402
import ttpd_common as C                                   # noqa: E402
from sa import sa                                          # noqa: E402

TIME_BUDGET = {
    1: 5.0, 5: 10.0, 10: 30.0, 15: 120.0, 20: 300.0,
    30: 450.0, 40: 600.0, 50: 750.0, 100: 1200.0,
}
SA_ALPHA = 0.97

_MILP_REFS: dict | None = None

def bench_path(n: int, inst_no: int = 1) -> str:
    return os.path.join(batch, f"bench_{n}_{inst_no}.txt")

def load_bench_instance(path: str):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return problem_from_instance(inst), inst

def milp_refs() -> dict:
    global _MILP_REFS
    if _MILP_REFS is not None:
        return _MILP_REFS
    refs = {}
    for fn in os.listdir(batch):
        if not (fn.startswith("MILP") and fn.endswith(".csv")):
            continue
        try:
            n = int(fn[len("MILP"):-len(".csv")])
        except ValueError:
            continue
        with open(os.path.join(batch, fn), newline="") as f:
            for row in csv.DictReader(f):
                try:
                    inst_no = int(str(row["instance"]).split("_")[-1])
                    refs[(n, inst_no)] = float(row["objective"])
                except (ValueError, KeyError, TypeError):
                    continue
    _MILP_REFS = refs
    return refs

def gap_pct(n: int, inst_no: int, obj: float):
    ref = milp_refs().get((n, inst_no))
    if ref is None or ref == 0:
        return ""
    return round((ref - obj) / abs(ref) * 100.0, 4)

def build_problem(path: str, *, vD_factor=None, R=None, Wcap=None, vmin=None,
                  ED=None, drone_disabled=False):

    _, inst = load_bench_instance(path)
    if drone_disabled:
  
        inst.v_D = inst.v_max * 1e-6
    elif vD_factor is not None:
        inst.v_D = inst.v_max * float(vD_factor)
    if R is not None:
        inst.R = float(R)
    if Wcap is not None:
        inst.W = float(Wcap)
    if vmin is not None:
        inst.v_min = float(vmin)
    P = problem_from_instance(inst)
    if ED is not None:
        P.ED = float(ED)
    return P, inst

def run_config(path: str, budget: float, seed: int = 1, **overrides) -> dict:
    P, inst = build_problem(path, **overrides)
    t0 = time.perf_counter()
    res = sa(P, time_limit=budget, seed=seed, alpha=SA_ALPHA)
    wall = time.perf_counter() - t0

    err = check_solution(P, res.truck, res.sorties, res.z)
    if err is not None:
        raise RuntimeError(f"infeasible SA solution: {err}")
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
        "R": float(inst.R), "ED": ("" if P.ED is None else round(float(P.ED), 4)),
        "collected_node_ids": det["collected_node_ids"],
        "truck_node_ids": det["truck_node_ids"],
        "drone_node_ids": det["drone_node_ids"],
        "rendezvous_node_ids": det["rendezvous_node_ids"],
        "truck_route_node_ids": det["truck_route_node_ids"],
        "drone_arc_node_ids": det["drone_arc_node_ids"],
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
