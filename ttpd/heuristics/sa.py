from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "common"), os.path.join(ROOT, "GAT")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import (Problem, check_solution, construct_nn, evaluate,  # noqa: E402
                  vnd)
from core import _shake_once  # noqa: E402  (single random feasible neighbor)

def _max_cands_for(n: int) -> int | None:
    return None if n <= 20 else 30 * n

# Same per-size time budgets and conventions as VNS.
DEFAULT_TIME = {5: 10.0, 10: 30.0, 15: 120.0, 20: 300.0}

@dataclass
class SAResult:
    objective: float
    truck: tuple
    sorties: tuple
    z: tuple
    seconds: float
    evals: int
    time_to_best: float
    reheats: int

def _neighbor(P, truck, sorties, z, rng, counter):
    out = _shake_once(P, truck, sorties, z, rng)
    counter[0] += 1
    return out  # (t2, s2, z2, v) or None


def _calibrate_T0(P, truck, sorties, z, rng, counter, target_accept=0.8,
                  samples=80):
    base = evaluate(P, truck, sorties, z)
    worse = []
    for _ in range(samples):
        out = _neighbor(P, truck, sorties, z, rng, counter)
        if out is None:
            continue
        d = out[3] - base
        if d < 0:
            worse.append(-d)
    if worse:
        avg = sum(worse) / len(worse)
        return max(avg / -math.log(target_accept), 1e-6)
    return max(abs(base) * 0.05, 1.0)


def sa(P: Problem, time_limit: float = 60.0, seed: int = 0,
       alpha: float = 0.97, moves_per_level: int | None = None,
       t_min_ratio: float = 1e-4, final_polish: bool = True,
       log: bool = False, trace: list | None = None,
       init: tuple | None = None) -> SAResult:
    rng = random.Random(seed)
    counter = [0]
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    max_cands = _max_cands_for(P.n)

    def rec(obj_val):
        if trace is not None:
            trace.append((time.perf_counter() - t0, obj_val))

    if init is not None:
        truck, sorties, z = init
    else:
        truck, sorties, z = construct_nn(P)
    obj = evaluate(P, truck, sorties, z)
    truck, sorties, z, obj = vnd(P, truck, sorties, z, obj, rng, counter,
                                 deadline=deadline, max_cands=max_cands)

    cur = (truck, sorties, z, obj)
    best = cur
    t_best = time.perf_counter() - t0
    rec(best[3])

    if moves_per_level is None:
        moves_per_level = max(20, 12 * (P.n + 1))

    T0 = _calibrate_T0(P, truck, sorties, z, rng, counter)
    T = T0
    T_min = T0 * t_min_ratio
    reheats = 0

    while time.perf_counter() - t0 < time_limit:
        for _ in range(moves_per_level):
            if time.perf_counter() - t0 >= time_limit:
                break
            out = _neighbor(P, cur[0], cur[1], cur[2], rng, counter)
            if out is None:
                continue
            t2, s2, z2, v = out
            d = v - cur[3]
            if d >= 0 or rng.random() < math.exp(d / T):
                cur = (t2, s2, z2, v)
                if v > best[3] + 1e-9:
                    best = cur
                    t_best = time.perf_counter() - t0
                    rec(best[3])
                    if log:
                        print(f"    [sa seed={seed}] improved to {v:.2f} "
                              f"at {t_best:.1f}s (T={T:.3g})")
        # Geometric cooling; reheat (keeping the best) once frozen.
        T *= alpha
        if T < T_min:
            T = T0
            reheats += 1
            # Resume the search from the best found so far after reheating.
            cur = best

    if final_polish:
        t2, s2, z2, v = vnd(P, best[0], best[1], best[2], best[3], rng, counter,
                            max_cands=max_cands)
        if v > best[3] + 1e-9:
            best = (t2, s2, z2, v)
            t_best = time.perf_counter() - t0
            rec(best[3])

    err = check_solution(P, best[0], best[1], best[2])
    if err is not None:
        raise RuntimeError(f"SA produced an infeasible best solution: {err}")
    return SAResult(
        objective=best[3], truck=best[0], sorties=best[1], z=best[2],
        seconds=time.perf_counter() - t0, evals=counter[0],
        time_to_best=t_best, reheats=reheats,
    )
