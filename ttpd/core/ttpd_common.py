from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulator import TTPDEnv          # noqa: E402
from masking import NO_LAUNCH          # noqa: E402


def _build_actions(inst, truck, sorties, z):
    route = [inst.source] + list(truck) + [inst.sink]
    pos = {nd: i for i, nd in enumerate(route)}

    sord = sorted(sorties, key=lambda s: pos[s[0]])
    launch_at = {pos[L]: (L, D, J) for (L, D, J) in sord}
    rejoin_at = {pos[J]: (L, D, J) for (L, D, J) in sord}

    actions = []
    for i in range(len(route) - 1):
        nd = route[i]
        action = {"j": route[i + 1], "z_curr": 0, "rejoin": 0,
                  "z_drone": 0, "k": NO_LAUNCH}
        if i in rejoin_at:
            _, D, _ = rejoin_at[i]
            action["rejoin"] = 1
            action["z_drone"] = int(z[D])
        if 0 < nd <= inst.n:
            action["z_curr"] = int(z[nd])
        if i in launch_at:
            action["k"] = int(launch_at[i][1])
        actions.append(action)

    last = len(route) - 1
    if last in rejoin_at:
        _, D, _ = rejoin_at[last]
        actions.append({"rejoin": 1, "z_drone": int(z[D])})
    return actions


def replay_details(inst, truck, sorties, z, a280_path: str | None = None):
    env = TTPDEnv(a280_path=a280_path, n=inst.n, scale_capacity=True)
    env.reset(options={"instance": inst})
    nid = inst.node_ids                

    actions = _build_actions(inst, truck, sorties, z)

    truck_route = [nid[inst.source]]
    drone_arcs, truck_items, drone_items, rendezvous = [], [], [], []
    W_seq = [(nid[inst.source], 0.0)]
    launch_from = None
    G = 0.0
    terminated = False

    for a in actions:
        c_before = env.c_t
        flying_before = env.in_flight
        target_before = env.drone_D
        _, r, terminated, truncated, _ = env.step(a)
        G += float(r)
        if flying_before and a["rejoin"] == 1:
            drone_arcs.append((nid[launch_from], nid[target_before]))
            drone_arcs.append((nid[target_before], nid[c_before]))
            rendezvous.append(nid[c_before])
            if a["z_drone"] == 1:
                drone_items.append(nid[target_before])
        if a.get("z_curr", 0) == 1:
            truck_items.append(nid[c_before])
        if a.get("k", NO_LAUNCH) != NO_LAUNCH:
            launch_from = c_before
        if "j" in a:
            truck_route.append(nid[a["j"]])
            W_seq.append((nid[a["j"]], float(env.W_t)))
        if terminated or truncated:
            break

    if not terminated:
        raise RuntimeError("Replay finished without the env terminating "
                           "(unserved customers or drone still in flight).")

    collected = truck_items + drone_items
    profit = float(sum(inst.profits[i] for i in range(1, inst.n + 1)
                       if nid[i] in collected))
    arrival = float(env.tau_t)
    rental = float(inst.R) * arrival
    return {
        "objective": G, "profit": profit, "rental": rental, "arrival": arrival,
        "n_items": len(collected), "n_truck": len(truck_items),
        "n_drone": len(drone_items), "n_rendv": len(rendezvous),
        "W_final": float(env.W_t),
        "truck_route_node_ids": truck_route, "drone_arc_node_ids": drone_arcs,
        "collected_node_ids": collected, "truck_node_ids": truck_items,
        "drone_node_ids": drone_items, "rendezvous_node_ids": rendezvous,
        "W_sequence": W_seq,
    }
