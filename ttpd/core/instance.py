#a
import math
import random
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from constants import (
    DRONE_SPEED_FACTOR,
    R_DEFAULT,
    R_LOGU_RANGE,
    V_MAX,
    V_MIN,
    scaled_capacity,
)

@dataclass
class TTPDInstance:
    n: int
    coords: np.ndarray
    profits: np.ndarray
    weights: np.ndarray
    dist: np.ndarray
    W: float
    v_max: float
    v_min: float
    v_D: float
    R: float
    node_ids: list
    ED: float | None = None   # per-sortie drone endurance (round-trip distance budget); None = unbounded

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return self.n + 1

    @property
    def d_max(self) -> float:
        return float(self.dist.max())

def _parse_dataset(filepath: str):
    with open(filepath) as f:
        lines = f.readlines()
    params, coords = {}, {}
    node_items = defaultdict(list)
    coord_section = items_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("DIMENSION"):
            params["n_nodes"] = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("CAPACITY OF KNAPSACK"):
            params["W"] = float(line.split(":")[1].strip())
        elif line.startswith("MIN SPEED"):
            params["v_min"] = float(line.split(":")[1].strip())
        elif line.startswith("MAX SPEED"):
            params["v_max"] = float(line.split(":")[1].strip())
        elif line.startswith("RENTING RATIO"):
            params["R"] = float(line.split(":")[1].strip())
        elif line.startswith("DRONE ENDURANCE"):
            params["ED"] = float(line.split(":")[1].strip())   # optional (ttd300)
        elif "NODE_COORD_SECTION" in line:
            coord_section, items_section = True, False
        elif "ITEMS SECTION" in line:
            items_section, coord_section = True, False
        elif coord_section:
            p = line.split()
            if len(p) == 3:
                coords[int(p[0])] = (float(p[1]), float(p[2]))
        elif items_section:
            p = line.split()
            if len(p) == 4:
                node_items[int(p[3])].append((int(p[1]), int(p[2])))

    # CEIL_2D distances and one best (max-profit) item per node, matching the MILP.
    best_item = {nid: max(items, key=lambda x: x[0]) for nid, items in node_items.items()}
    return params, coords, best_item

def _euclidean_ceil(x1, y1, x2, y2):
    return math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def load_a280(filepath: str) -> dict:
    params, coords, best_item = _parse_dataset(filepath)
    return {"params": params, "coords": coords, "best_item": best_item}

def _build_instance(
    a280_data: dict,
    chosen: list[tuple[int, float, float]],
    depot_id: int,
    R: float | None,
    scale_capacity: bool,
) -> TTPDInstance:
    coords = a280_data["coords"]
    params = a280_data["params"]

    n_eff = len(chosen)
    node_count = n_eff + 2
    node_ids = [depot_id] + [c[0] for c in chosen] + [depot_id]

    coord_arr = np.zeros((node_count, 2), dtype=np.float64)
    profit_arr = np.zeros(node_count, dtype=np.float64)
    weight_arr = np.zeros(node_count, dtype=np.float64)
    for i, nid in enumerate(node_ids):
        coord_arr[i] = coords[nid]
    for i in range(1, n_eff + 1):
        _, p, w = chosen[i - 1]
        profit_arr[i] = float(p)
        weight_arr[i] = float(w)

    dist = np.zeros((node_count, node_count), dtype=np.float64)
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            dist[i, j] = _euclidean_ceil(
                coord_arr[i, 0], coord_arr[i, 1], coord_arr[j, 0], coord_arr[j, 1]
            )

    if R is None:
        R = params.get("R", R_DEFAULT)
    W = params.get("W", 637_010.0)
    if scale_capacity:
        W = scaled_capacity(W, n_eff)
    v_max = params.get("v_max", V_MAX)
    v_min = params.get("v_min", V_MIN)

    return TTPDInstance(
        n=n_eff,
        coords=coord_arr,
        profits=profit_arr,
        weights=weight_arr,
        dist=dist,
        W=float(W),
        v_max=float(v_max),
        v_min=float(v_min),
        v_D=float(v_max * DRONE_SPEED_FACTOR),
        R=float(R),
        node_ids=node_ids,
        ED=params.get("ED"),
    )


def milp_instance(
    a280_data: dict,
    n: int,
    seed: int | None = None,
    depot_id: int = 1,
    scale_capacity: bool = True,
) -> TTPDInstance:
    if seed is None:
        seed = n
    best_item = a280_data["best_item"]
    customers = [
        (nid, p, w) for nid, (p, w) in best_item.items() if nid != depot_id
    ]
    chosen = random.Random(seed).sample(customers, min(n, len(customers)))
    chosen = [(nid, float(p), float(w)) for nid, p, w in chosen]
    return _build_instance(a280_data, chosen, depot_id, R=None,
                           scale_capacity=scale_capacity)


# Scale
def sample_instance(
    a280_data: dict,
    n: int,
    rng: np.random.Generator,
    R: float | None = None,
    sample_R: bool = False,
    depot_id: int = 1,
    scale_capacity: bool = True,
) -> TTPDInstance:
    best_item = a280_data["best_item"]

    customers = [
        (nid, p, w) for nid, (p, w) in best_item.items() if nid != depot_id
    ]
    chosen_idx = rng.choice(len(customers), size=min(n, len(customers)), replace=False)
    chosen = [(nid, float(p), float(w)) for nid, p, w in (customers[i] for i in chosen_idx)]

    if R is None and sample_R:
        lo, hi = R_LOGU_RANGE
        R = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    return _build_instance(a280_data, chosen, depot_id, R=R,
                           scale_capacity=scale_capacity)
