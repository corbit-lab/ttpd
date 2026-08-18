from __future__ import annotations
import glob
import math
import os
import random
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths
_paths.use("core")

from ttpd.hub import ensure_local, ensure_bench_dir  

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _ROOT
DATA = ensure_bench_dir("ttd300")
TEMPLATE = os.path.join(DATA, "ttd300_n10_L1.txt")

from core import problem_from_instance                     
from instance import load_a280, _build_instance            

BOX = 300
ITEMS_PER_NODE = 5
W_LO, W_HI = 1000, 1009
P_LO, P_HI = 1, 1000
R = 50.0
V_MIN, V_MAX = 0.1, 1
CAP_PER_NODE = 637010 / 280

SIZES = [10, 20, 30, 40, 50, 75, 100]
LAYOUTS = [1, 2, 3, 4, 5]
ED_FRACS = [0.25, 0.50, 0.75, 1.00]
# full per-size SA budgets (seconds) -- the the benchmark sweep ladder (run_bench.py / run.log)
FULL_BUDGET = {10: 120.0, 20: 300.0, 30: 450.0, 40: 600.0,
               50: 750.0, 75: 975.0, 100: 1200.0}

def frac_tag(frac: float) -> str:
    return f"f{int(round(frac * 100)):03d}"

def reference_sources() -> tuple[list[str], list[str]]:
    """(csv_paths, log_paths) holding SA/VNS full-budget results, used to build
    the pareto reference. Missing paths are fine -- callers skip them."""
    if BUNDLED:
        ref = os.path.join(HERE, "ref")
        return (sorted(glob.glob(os.path.join(ref, "*.csv"))),
                sorted(glob.glob(os.path.join(ref, "*.log"))))
    ref = os.path.join(_ROOT, "artifacts", "data", "results", "ttd300")
    return (sorted(glob.glob(os.path.join(ref, "benchmark_sweep", "*.csv")))
            + sorted(glob.glob(os.path.join(ref, "sa", "result*.csv"))),
            sorted(glob.glob(os.path.join(ref, "benchmark_sweep", "*.log"))))

def bench_label(n: int, L: int, frac: float) -> str:
    return f"ttd300_n{n}_L{L}_{frac_tag(frac)}"

def load_ttd300(n: int, L: int, frac: float, data_dir: str = DATA):
    """One benchmark instance -> (Problem, TTPDInstance); ED from the header."""
    path = os.path.join(data_dir, bench_label(n, L, frac) + ".txt")
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    assert inst.ED is not None, f"no DRONE ENDURANCE header in {path}"
    return problem_from_instance(inst), inst

def _d_max_ceil2d(coords: dict) -> int:
    pts = list(coords.values())
    m = 0
    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            d = math.ceil(math.hypot(xi - pts[j][0], yi - pts[j][1]))
            if d > m:
                m = d
    return m

def sample_ttd300(n: int, sample_seed: int, ed_fracs=tuple(ED_FRACS)):
    rng = random.Random(sample_seed)
    coords = {i: (rng.randint(0, BOX), rng.randint(0, BOX))
              for i in range(1, n + 2)}
    best_item = {}
    for node in range(2, n + 2):
        items = [(rng.randint(P_LO, P_HI), rng.randint(W_LO, W_HI))
                 for _ in range(ITEMS_PER_NODE)]
        best_item[node] = max(items, key=lambda x: x[0])
    frac = rng.choice(list(ed_fracs))
    ED = int(round(frac * _d_max_ceil2d(coords)))
    params = {"n_nodes": n + 1, "W": float(int(round(CAP_PER_NODE * n))),
              "v_min": V_MIN, "v_max": V_MAX, "R": R, "ED": float(ED)}
    data = {"params": params, "coords": coords, "best_item": best_item}
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best_item.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return problem_from_instance(inst), inst, frac
