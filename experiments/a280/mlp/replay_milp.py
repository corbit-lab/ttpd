#a
import argparse
import ast
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
_paths.use("rl/a280/mlp")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

ROOT = _ROOT

from env.instance import _build_instance, load_a280
from env.simulator import TTPDEnv
from env.masking import NO_LAUNCH

DEPOT_ID = 1

def build_fixed_instance(a280, node_ids, scale_capacity, W_override=None):
    best_item = a280["best_item"]
    chosen = []
    for nid in node_ids:
        if nid == DEPOT_ID:
            continue
        p, w = best_item[nid]
        chosen.append((nid, float(p), float(w)))
    inst = _build_instance(a280, chosen, DEPOT_ID, R=None, scale_capacity=scale_capacity)
    if W_override is not None:
        inst.W = float(W_override)
    return inst

def reconstruct_drone_tour(arcs):
    if not arcs:
        return []
    succ = {int(u): int(v) for u, v in arcs}
    if DEPOT_ID not in succ:
        return []
    tour, cur, guard = [DEPOT_ID], DEPOT_ID, 0
    while cur in succ and guard < 10_000:
        cur = succ[cur]
        tour.append(cur)
        guard += 1
        if cur == DEPOT_ID:
            break
    return tour

def split_into_sorties(tour, rendezvous, truck_ids=None):
    meeting = set(rendezvous) | {DEPOT_ID}
    if truck_ids is not None:
        meeting |= (set(tour) & set(truck_ids))
    sorties = []
    seg_start = tour[0]
    deliveries = []
    for node in tour[1:]:
        if node in meeting:
            if len(deliveries) == 1:
                sorties.append((seg_start, deliveries[0], node))
            elif len(deliveries) > 1:
                raise ValueError(f"sortie {seg_start}->{node} has >1 delivery {deliveries}; "
                                 "violates drone_anchor; replay mapping cannot be built.")
            seg_start = node
            deliveries = []
        else:
            deliveries.append(node)
    return sorties

def replay_row(a280, a280_path, row, scale_capacity):
    milp_obj = float(row["objective"])
    truck_ids = ast.literal_eval(row["truck_route_node_ids"])
    arcs = ast.literal_eval(row["drone_arc_node_ids"])
    rendezvous = {int(x) for x in ast.literal_eval(row["rendezvous_node_ids"])}
    collected = {int(x) for x in ast.literal_eval(row["collected_node_ids"])}

    node_set = {int(x) for x in truck_ids if int(x) != DEPOT_ID}
    for u, v in arcs:
        node_set.add(int(u)); node_set.add(int(v))
    node_set.discard(DEPOT_ID)
    w_cap = row.get("W_capacity")
    W_override = float(w_cap) if (scale_capacity and w_cap) else None
    inst = build_fixed_instance(a280, sorted(node_set), scale_capacity, W_override)

    id_to_env = {nid: i for i, nid in enumerate(inst.node_ids) if nid != DEPOT_ID}
    truck_env = [0] + [id_to_env[i] for i in truck_ids if i != DEPOT_ID] + [inst.sink]
    sorties = split_into_sorties(reconstruct_drone_tour(arcs), rendezvous,
                                 truck_ids={int(x) for x in truck_ids})

    launch_at, rejoin_at = {}, {}
    depot_launch = None
    for si, (L, D, J) in enumerate(sorties):
        if L == DEPOT_ID:
            depot_launch = si
        else:
            launch_at[L] = si
        if J != DEPOT_ID:
            rejoin_at[J] = si

    env = TTPDEnv(a280_path=a280_path, n=inst.n, scale_capacity=scale_capacity)
    env.reset(options={"instance": inst})

    total = 0.0
    cur_sortie = None
    for pos in range(len(truck_env) - 1):
        c_env = truck_env[pos]
        c_id = inst.node_ids[c_env]
        j_env = truck_env[pos + 1]

        action = {"j": int(j_env), "z_curr": 0, "rejoin": 0, "z_drone": 0, "k": NO_LAUNCH}

        if cur_sortie is not None and c_env != 0:
            L, D, J = sorties[cur_sortie]
            if c_id == J:
                action["rejoin"] = 1
                action["z_drone"] = 1 if D in collected else 0
                cur_sortie = None

        if c_env not in (0, inst.sink) and c_id in collected:
            action["z_curr"] = 1

        si = None
        if pos == 0 and depot_launch is not None:
            si = depot_launch
        elif c_id in launch_at:
            si = launch_at[c_id]
        if si is not None and cur_sortie is None:
            L, D, J = sorties[si]
            action["k"] = int(id_to_env[D])
            cur_sortie = si

        _, r, term, trunc, _ = env.step(action)
        total += r

    if cur_sortie is not None:   # drone still out -> terminal landing at the sink
        L, D, J = sorties[cur_sortie]
        zc = 1 if D in collected else 0
        _, r, term, trunc, _ = env.step({"rejoin": 1, "z_drone": zc})
        total += r

    rel = abs(total - milp_obj) / max(abs(milp_obj), 1.0)
    return total, milp_obj, rel < 1e-3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(ROOT, "..", "MILP", "milp_results.csv"))
    p.add_argument("--a280", default=hub_instance("a280", "a280_benchmark.txt"))
    p.add_argument("--scale-capacity", dest="scale_capacity", action="store_true", default=True)
    p.add_argument("--no-scale-capacity", dest="scale_capacity", action="store_false")
    args = p.parse_args()

    a280 = load_a280(args.a280)
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    print(f"{'n':>4} {'env_return':>14} {'MILP_obj':>14} {'rel_diff':>10} {'match':>7}")
    print("-" * 56)
    all_ok = True
    for row in rows:
        try:
            env_ret, milp, ok = replay_row(a280, args.a280, row, args.scale_capacity)
            rel = abs(env_ret - milp) / max(abs(milp), 1.0)
            if not ok:
                all_ok = False
            print(f"{int(row['n']):>4} {env_ret:>14.2f} {milp:>14.2f} {rel*100:>9.3f}% "
                  f"{'OK' if ok else 'CHECK':>7}")
        except Exception as e:
            all_ok = False
            print(f"{int(row['n']):>4} {'ERROR':>14} {'':>14} {'':>10}  {type(e).__name__}: {e}")
    print("-" * 56)
    note = ("n=5/n=10 (true optima) must match exactly; larger n carry the MILP's velocity "
            "PL-approx error (env is exact, ~0.03%).")
    print(("ALL WITHIN TOLERANCE. " if all_ok else "SOME EXCEED 0.1%; inspect. ") + note)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
