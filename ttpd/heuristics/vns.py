from __future__ import annotations
import os
import random
import sys
import time
from dataclasses import dataclass

# Flat layout: every module this needs (core, instance, ...) sits beside it.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core import (Problem, check_solution, construct_nn, evaluate,  # noqa: E402
                  shake, vnd)

def max_cands_for(n: int) -> int | None:
    if n <= 20:
        return None
    return 30 * n

# Same per-size time budgets as before.
DEFAULT_TIME = {5: 10.0, 10: 30.0, 15: 120.0, 20: 300.0}

@dataclass
class VNSResult:
    objective: float
    truck: tuple
    sorties: tuple
    z: tuple
    seconds: float
    evals: int
    time_to_best: float

def vns(P: Problem, time_limit: float = 60.0, k_max: int = 8,
        seed: int = 0, log: bool = False,
        trace: list | None = None) -> VNSResult:
    rng = random.Random(seed)
    counter = [0]
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    max_cands = max_cands_for(P.n)

    def rec(obj_val):
        if trace is not None:
            trace.append((time.perf_counter() - t0, obj_val))

    # Construction + initial descent (deadline-bounded so it can't overrun)
    truck, sorties, z = construct_nn(P)
    obj = evaluate(P, truck, sorties, z)
    cur = vnd(P, truck, sorties, z, obj, rng, counter,
              deadline=deadline, max_cands=max_cands)

    best = cur
    t_best = time.perf_counter() - t0
    rec(best[3])

    k = 1
    while time.perf_counter() < deadline:
        # Shake in the k-th neighborhood, then descend with VND
        t2, s2, z2, v = shake(P, cur[0], cur[1], cur[2], k, rng)
        t2, s2, z2, v = vnd(P, t2, s2, z2, v, rng, counter,
                            deadline=deadline, max_cands=max_cands)

        if v > cur[3] + 1e-9:
            # Improving move: recentre and reset the neighborhood index
            cur = (t2, s2, z2, v)
            k = 1
            if v > best[3] + 1e-9:
                best = cur
                t_best = time.perf_counter() - t0
                rec(best[3])
                if log:
                    print(f"    [vns seed={seed}] improved to {v:.2f} "
                          f"at {t_best:.1f}s")
        else:
            # No improvement: widen the shake, cycling k back to 1 at k_max
            k = k + 1 if k < k_max else 1

    rec(best[3])
    err = check_solution(P, best[0], best[1], best[2])
    if err is not None:
        raise RuntimeError(f"VNS produced an infeasible best solution: {err}")
    return VNSResult(
        objective=best[3], truck=best[0], sorties=best[1], z=best[2],
        seconds=time.perf_counter() - t0, evals=counter[0],
        time_to_best=t_best,
    )
