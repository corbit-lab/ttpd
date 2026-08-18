from __future__ import annotations

import sys

import argparse
import math
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("core")

from ttpd.hub import ensure_local  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import random
import time
from collections import defaultdict

import exp_common as E
from core import nn_route  # reuse the nearest-neighbour truck constructor

SIZES = [5, 10, 15, 20]
INST_NO = 1
SEED = 1
eps = 1e-9

def _ceil2d(x1, y1, x2, y2):
    return math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

class MProblem:

    def __init__(self, path):
        coords, items, params = self._parse(path)
        ids = sorted(coords)
        self.node_ids = ids
        self.n = len(ids) - 1                       # customers (depot excluded)
        self.src, self.snk = 0, self.n + 1
        nn = self.n + 2
        slot_xy = [coords[ids[0]]] + [coords[ids[i]] for i in range(1, self.n + 1)] \
            + [coords[ids[0]]]
        self.dist = [[0.0] * nn for _ in range(nn)]
        for i in range(nn):
            for j in range(nn):
                if i != j:
                    self.dist[i][j] = float(_ceil2d(*slot_xy[i], *slot_xy[j]))
        # items per slot: list of (profit, weight)
        self.items = [[] for _ in range(nn)]
        for i in range(1, self.n + 1):
            self.items[i] = [(float(p), float(w)) for (p, w) in items.get(ids[i], [])]
        self.Wcap = float(params["W"])
        self.vmax = float(params.get("v_max", 1.0))
        self.vmin = float(params.get("v_min", 0.1))
        self.vD = self.vmax * 2.0
        self.R = float(params.get("R", 72.70))
        self.slope = (self.vmax - self.vmin) / self.Wcap
        self.W_actual = float(sum(w for it in self.items for (_, w) in it))

    @staticmethod
    def _parse(path):
        coords, items, params = {}, defaultdict(list), {}
        cs = it = False
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("CAPACITY OF KNAPSACK"):
                    params["W"] = float(s.split(":")[1])
                elif s.startswith("MIN SPEED"):
                    params["v_min"] = float(s.split(":")[1])
                elif s.startswith("MAX SPEED"):
                    params["v_max"] = float(s.split(":")[1])
                elif s.startswith("RENTING RATIO"):
                    params["R"] = float(s.split(":")[1])
                elif "NODE_COORD_SECTION" in s:
                    cs, it = True, False
                elif "ITEMS SECTION" in s:
                    it, cs = True, False
                elif cs:
                    p = s.split()
                    if len(p) == 3:
                        coords[int(p[0])] = (float(p[1]), float(p[2]))
                elif it:
                    p = s.split()
                    if len(p) == 4:
                        items[int(p[3])].append((int(p[1]), int(p[2])))
        return coords, items, params

def ws_ps(P, node, mask):
    w = p = 0.0
    its = P.items[node]
    m = mask
    i = 0
    while m:
        if m & 1:
            pr, wt = its[i]
            p += pr
            w += wt
        m >>= 1
        i += 1
    return w, p

def evaluate_m(P, truck, sorties, sel):
    route = (0,) + tuple(truck) + (P.n + 1,)
    pos = {nd: i for i, nd in enumerate(route)}
    rejoin_at, launch_at = {}, {}
    if sorties:
        try:
            sord = sorted(sorties, key=lambda s: pos[s[0]])
        except KeyError:
            return None
        last_end = 0
        for idx, (L, D, J) in enumerate(sord):
            pL, pJ = pos.get(L), pos.get(J)
            if pL is None or pJ is None or pL >= pJ or pL < last_end:
                return None
            last_end = pJ
            launch_at[pL] = idx
            rejoin_at[pJ] = idx
    else:
        sord = ()

    dist, vD, vmax, slope = P.dist, P.vD, P.vmax, P.slope
    Wcap = P.Wcap + eps
    tau = Wt = prof = 0.0
    lt = [0.0] * len(sord)
    last = len(route) - 1

    for i, nd in enumerate(route):
        si = rejoin_at.get(i)
        if si is not None:
            L, D, _ = sord[si]
            arr = lt[si] + (dist[L][D] + dist[D][nd]) / vD
            if arr > tau:
                tau = arr
            w, p = ws_ps(P, D, sel[D])
            if w:
                Wt += w
                if Wt > Wcap:
                    return None
                prof += p
        if 0 < nd <= P.n and sel[nd]:
            w, p = ws_ps(P, nd, sel[nd])
            Wt += w
            if Wt > Wcap:
                return None
            prof += p
        si = launch_at.get(i)
        if si is not None:
            lt[si] = tau
        if i < last:
            tau += dist[nd][route[i + 1]] / (vmax - Wt * slope)
    return prof - P.R * tau

def _anchors(sorties):
    s = set()
    for L, _, J in sorties:
        s.add(L)
        s.add(J)
    return s

def _repaired(truck, sorties, P):
    pos = {nd: i for i, nd in enumerate((0,) + tuple(truck) + (P.n + 1,))}
    out, changed = [], False
    for L, D, J in sorties:
        pL, pJ = pos.get(L), pos.get(J)
        if pL is None or pJ is None:
            return None
        if pL > pJ:
            out.append((J, D, L)); changed = True
        else:
            out.append((L, D, J))
    return tuple(out) if changed else None

def shake_m(P, truck, sorties, sel, rng):
    """One random feasible multi-item neighbour (route/sortie + item moves)."""
    for _ in range(40):
        t2, s2, sl2 = truck, sorties, sel
        op = rng.randint(0, 6)

        if op in (0, 1) and P.n > 0:           # item add/remove/toggle (weighted)
            i = rng.randint(1, P.n)
            k = len(P.items[i])
            if k == 0:
                continue
            m = rng.randrange(k)
            sl = list(sel)
            sl[i] ^= (1 << m)
            sl2 = tuple(sl)

        elif op == 2 and sorties:              # move a sortie anchor pair
            si = rng.randrange(len(sorties))
            _, D, _ = sorties[si]
            route = (0,) + truck + (P.n + 1,)
            ia = rng.randrange(len(route) - 1)
            ib = rng.randrange(ia + 1, len(route))
            s2 = sorties[:si] + ((route[ia], D, route[ib]),) + sorties[si + 1:]

        elif op == 3 and truck:                # truck -> drone
            t = rng.choice(truck)
            if t in _anchors(sorties):
                continue
            t2 = tuple(x for x in truck if x != t)
            route = (0,) + t2 + (P.n + 1,)
            ia = rng.randrange(len(route) - 1)
            ib = rng.randrange(ia + 1, len(route))
            s2 = sorties + ((route[ia], t, route[ib]),)

        elif op == 4 and sorties:              # drone -> truck
            si = rng.randrange(len(sorties))
            D = sorties[si][1]
            s2 = sorties[:si] + sorties[si + 1:]
            b = rng.randint(0, len(truck))
            t2 = truck[:b] + (D,) + truck[b:]

        elif op == 5 and len(truck) >= 2:      # or-opt relocate
            a = rng.randrange(len(truck))
            t = truck[a]
            rest = truck[:a] + truck[a + 1:]
            b = rng.randint(0, len(rest))
            t2 = rest[:b] + (t,) + rest[b:]
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                s2 = rep

        elif op == 6 and len(truck) >= 3:      # 2-opt reverse
            a = rng.randrange(len(truck) - 1)
            b = rng.randrange(a + 1, len(truck))
            t2 = truck[:a] + truck[a:b + 1][::-1] + truck[b + 1:]
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                s2 = rep
        else:
            continue

        if (t2, s2, sl2) == (truck, sorties, sel):
            continue
        v = evaluate_m(P, t2, s2, sl2)
        if v is not None:
            return t2, s2, sl2, v
    return None

def item_polish(P, truck, sorties, sel, obj):
    """First-improvement single-item add/remove sweep until no gain."""
    improved = True
    while improved:
        improved = False
        for i in range(1, P.n + 1):
            for m in range(len(P.items[i])):
                sl = list(sel)
                sl[i] ^= (1 << m)
                sl2 = tuple(sl)
                v = evaluate_m(P, truck, sorties, sl2)
                if v is not None and v > obj + 1e-9:
                    sel, obj = sl2, v
                    improved = True
    return sel, obj

def sa_multi(P, budget, seed=1, alpha=0.97):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    truck = nn_route(P, range(1, P.n + 1))
    sorties = ()
    sel = tuple([0] * (P.n + 2))
    sel, obj = item_polish(P, truck, sorties, sel, evaluate_m(P, truck, sorties, sel))
    cur = (truck, sorties, sel, obj)
    best = cur

    # T0 calibration: average worsening of random moves -> ~80% accept.
    worse = []
    for _ in range(80):
        out = shake_m(P, cur[0], cur[1], cur[2], rng)
        if out and out[3] - cur[3] < 0:
            worse.append(cur[3] - out[3])
    T0 = max((sum(worse) / len(worse)) / -math.log(0.8), 1.0) if worse \
        else max(abs(obj) * 0.05, 1.0)
    T, T_min = T0, T0 * 1e-4
    moves = max(20, 12 * (P.n + 1))
    evals = 0

    while time.perf_counter() - t0 < budget:
        for _ in range(moves):
            if time.perf_counter() - t0 >= budget:
                break
            out = shake_m(P, cur[0], cur[1], cur[2], rng)
            evals += 1
            if out is None:
                continue
            t2, s2, sl2, v = out
            d = v - cur[3]
            if d >= 0 or rng.random() < math.exp(d / T):
                cur = (t2, s2, sl2, v)
                if v > best[3] + 1e-9:
                    best = cur
        T *= alpha
        if T < T_min:
            T = T0
            cur = best

    sel, obj = item_polish(P, best[0], best[1], best[2], best[3])
    return best[0], best[1], sel, obj, evals, time.perf_counter() - t0

def solution_stats(P, truck, sorties, sel):
    drone_targets = {s[1] for s in sorties}
    per_city, total_items, total_w, total_p = {}, 0, 0.0, 0.0
    for i in range(1, P.n + 1):
        cnt = bin(sel[i]).count("1")
        if cnt:
            per_city[i] = cnt
            total_items += cnt
            w, p = ws_ps(P, i, sel[i])
            total_w += w
            total_p += p
    cities = len(per_city)
    multi = {i: c for i, c in per_city.items() if c > 1}
    truck_cities = [i for i in per_city if i not in drone_targets]
    drone_cities = [i for i in per_city if i in drone_targets]
    return {
        "cities_collected": cities,
        "total_items": total_items,
        "multi_pick_cities": len(multi),
        "max_items_one_city": max(per_city.values()) if per_city else 0,
        "avg_items_per_city": round(total_items / cities, 3) if cities else 0.0,
        "items_dist": dict(sorted(defaultdict(int, _hist(per_city.values())).items())),
        "n_truck_cities": len(truck_cities),
        "n_drone_cities": len(drone_cities),
        "W_collected": round(total_w, 1),
        "profit": round(total_p, 1),
        "cap_util_%": round(100.0 * total_w / P.Wcap, 1),
    }

def _hist(vals):
    h = defaultdict(int)
    for v in vals:
        h[v] += 1
    return h

COLUMNS = [
    "n", "instance", "mi_objective", "si_objective", "delta_obj",
    "cities_collected", "total_items", "avg_items_per_city",
    "multi_pick_cities", "max_items_one_city", "items_dist",
    "W_collected", "W_capacity", "W_actual", "cap_util_%",
    "profit", "n_truck_cities", "n_drone_cities", "time_s",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(E.HERE, "results", "multiitem.csv"))
    cli = ap.parse_args()

    sizes = [int(s) for s in cli.sizes.split(",")]
    rows = []
    for n in sizes:
        path = E.bench_path(n, INST_NO)
        if not os.path.exists(path):
            print(f"  MISSING {path} -- skip")
            continue
        budget = cli.budget if cli.budget is not None else E.TIME_BUDGET[n]

        # single-item baseline (base SA, one max-profit item per city)
        base = E.run_config(path, budget, seed=cli.seed)

        # multi-item run
        P = MProblem(path)
        truck, sorties, sel, obj, evals, wall = sa_multi(P, budget, seed=cli.seed)
        st = solution_stats(P, truck, sorties, sel)

        r = {
            "n": n, "instance": f"bench_{n}_{INST_NO}",
            "mi_objective": round(obj, 4), "si_objective": base["objective"],
            "delta_obj": round(obj - base["objective"], 4),
            "W_capacity": round(P.Wcap, 1), "W_actual": round(P.W_actual, 1),
            "time_s": round(wall, 2), **st,
        }
        rows.append(r)
        print(f"  n={n:<4} MI_obj={r['mi_objective']:>14} "
              f"SI_obj={r['si_objective']:>14} "
              f"cities={st['cities_collected']} items={st['total_items']} "
              f"multi={st['multi_pick_cities']} max/city={st['max_items_one_city']} "
              f"cap={st['cap_util_%']}% dist={st['items_dist']}")
        E.write_csv(cli.out, COLUMNS, rows)

    print(f"\nDONE -> {cli.out}")

if __name__ == "__main__":
    main()
