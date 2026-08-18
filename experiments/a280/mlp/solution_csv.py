#a
import argparse
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

from ttpd.hub import ensure_local, weights_dir as hub_weights, instance as hub_instance  # noqa: E402

ROOT = _ROOT
import time

import numpy as np
import torch

from env import TTPDEnv
from env.instance import load_a280, milp_instance
from env.masking import NO_LAUNCH
from policy.attention_policy import AttentionPolicy
from policy.beam import beam_search
from policy.decoder import DecoderConfig
from policy.encoder import EncoderConfig
from train.eval import load_milp_refs, MILP_PROVEN_OPTIMAL

def load_policy(ckpt, device):
    payload = torch.load(ensure_local(ckpt), map_location=device, weights_only=False)
    enc = EncoderConfig(); dec = DecoderConfig(d_model=enc.d_model)
    p = AttentionPolicy(enc, dec, device=device)
    p.load_state_dict(payload["policy"]); p.eval()
    return p

def _trace_rollout(policy, env, inst, ctx, first_j, deterministic, temperature=1.0):
    obs, _ = env.reset(options={"instance": inst})
    G, s, acts, done = 0.0, 0, [], False
    while not done:
        if s == 0 and first_j is not None:
            sample = policy.act_force(obs, env, {"j": int(first_j)}, ctx=ctx,
                                      deterministic=deterministic)
        else:
            sample = policy.act(obs, env, deterministic=deterministic, ctx=ctx,
                                temperature=temperature)
        try:
            obs, r, term, trunc, _ = env.step(sample.action)
        except ValueError:
            return None, None
        acts.append(sample.action)
        G += r; s += 1
        done = term or trunc
    return G, acts

def best_with_trace(policy, env, inst, *, n_sample, beam_width, seed=0):

    torch.manual_seed(seed)
    env.reset(options={"instance": inst})
    feasible = [int(j) for j in np.flatnonzero(env.current_masks()["j"])
                if j not in (inst.source, inst.sink)]
    best = (None, None)
    for aug in range(8):
        ctx = policy.encode(inst, aug_idx=aug)
        for fj in feasible:
            G, acts = _trace_rollout(policy, env, inst, ctx, fj, True)
            if G is not None and (best[0] is None or G > best[0]):
                best = (G, acts)
            for _ in range(n_sample):
                G, acts = _trace_rollout(policy, env, inst, ctx, fj, False)
                if G is not None and (best[0] is None or G > best[0]):
                    best = (G, acts)
    G, acts = beam_search(policy, env, inst, beam_width=beam_width, n_aug=8,
                          stratify_k0=True, return_actions=True)
    if G is not None and (best[0] is None or G > best[0]):
        best = (G, acts)
    return best

def replay_details(inst, a280_path, actions):
    env = TTPDEnv(a280_path=a280_path, n=inst.n, scale_capacity=True)
    env.reset(options={"instance": inst})
    nid = inst.node_ids                      # slot -> a280 node id
    truck_route = [nid[inst.source]]
    drone_arcs, truck_items, drone_items, rendezvous = [], [], [], []
    W_seq = [(nid[inst.source], 0.0)]
    launch_from = None
    G = 0.0
    for a in actions:
        c_before = env.c_t
        flying_before = env.in_flight
        target_before = env.drone_D
        obs, r, term, trunc, _ = env.step(a)
        G += float(r)
        if flying_before and a["rejoin"] == 1:
            drone_arcs.append((nid[launch_from], nid[target_before]))
            drone_arcs.append((nid[target_before], nid[c_before]))
            rendezvous.append(nid[c_before])
            if a["z_drone"] == 1:
                drone_items.append(nid[target_before])
        if a["z_curr"] == 1:
            truck_items.append(nid[c_before])
        if a["k"] != NO_LAUNCH:
            launch_from = c_before
        truck_route.append(nid[a["j"]])
        W_seq.append((nid[a["j"]], float(env.W_t)))
        if term or trunc:
            break
    collected = truck_items + drone_items
    profit = float(sum(inst.profits[i] for i in range(1, inst.n + 1)
                       if nid[i] in collected))
    arrival = float(env.tau_t)
    rental = float(inst.R) * arrival
    return {
        "objective": G, "profit": profit, "rental": rental, "arrival": arrival,
        "n_items": len(collected), "n_truck": len(truck_items),
        "n_drone": len(drone_items), "n_rendv": len(rendezvous),
        "W_final": float(env.W_t), "W_capacity": float(inst.W),
        "W_actual": float(inst.weights.sum()),
        "truck_route_node_ids": truck_route, "drone_arc_node_ids": drone_arcs,
        "collected_node_ids": collected, "truck_node_ids": truck_items,
        "drone_node_ids": drone_items, "rendezvous_node_ids": rendezvous,
        "W_sequence": W_seq,
    }

COLUMNS = ["n", "method", "status", "objective", "profit", "rental", "arrival",
           "n_items", "n_truck", "n_drone", "n_rendv", "W_final", "W_capacity",
           "W_actual", "mip_gap_%", "gap_vs_milp_%", "n_nodes_explored",
           "build_s", "solve_s", "total_s",
           "truck_route_node_ids", "drone_arc_node_ids", "collected_node_ids",
           "truck_node_ids", "drone_node_ids", "rendezvous_node_ids", "W_sequence"]

# NOTE: these are the GAT final checkpoints, as in the original
# script -- this table is shared with experiments/a280/gat/solution_csv.py.
CKPTS = {
    5: os.path.join(hub_weights("a280", "gat", "specialists"), "n5_small.pt"),
    10: os.path.join(hub_weights("a280", "gat", "specialists"), "n10_small.pt"),
    15: os.path.join(hub_weights("a280", "gat", "specialists"), "n15_critic.pt"),
    20: os.path.join(hub_weights("a280", "gat", "specialists"), "n20_small.pt"),
}
DECODE = "greedy POMO + sampling(256) + stratified beam(1024), 8 augs"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="5,10,15,20")
    p.add_argument("--sample", type=int, default=256)
    p.add_argument("--beam", type=int, default=1024)
    p.add_argument("--out", default="comparison_milp_vs_rl.csv")
    args = p.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    repo = os.path.join(ROOT, "..")
    a280_path = hub_instance("a280", "a280_benchmark.txt")
    a280 = load_a280(a280_path)
    refs = load_milp_refs(os.path.join(repo, "MILP"))

    milp_rows = {}
    with open(os.path.join(repo, "MILP", "ttpd_results.csv")) as f:
        for r in csv.DictReader(f):
            milp_rows[int(r["n"])] = r

    out_rows = []
    for n in sizes:
        m = milp_rows[n]
        ref = refs[n]
        mrow = {c: m.get(c, "") for c in COLUMNS}
        mrow.update({"n": n, "method": "MILP (Gurobi)",
                     "status": "Optimal" if n in MILP_PROVEN_OPTIMAL else "Incumbent",
                     "gap_vs_milp_%": 0.0})
        out_rows.append(mrow)

        policy = load_policy(os.path.join(repo, CKPTS[n]), "cpu")
        inst = milp_instance(a280, n, seed=n, scale_capacity=True)
        env = TTPDEnv(a280_path=a280_path, n=n, scale_capacity=True)
        t0 = time.time()
        G, acts = best_with_trace(policy, env, inst,
                                  n_sample=args.sample, beam_width=args.beam)
        wall = time.time() - t0
        det = replay_details(inst, a280_path, acts)
        assert abs(det["objective"] - G) < 1e-6, (det["objective"], G)
        gap = (ref - G) / abs(ref) * 100.0
        rrow = {c: "" for c in COLUMNS}
        rrow.update({"n": n, "method": "DRL (GAT + POMO-PPO)", "status": "RL",
                     "mip_gap_%": "", "gap_vs_milp_%": round(gap, 3),
                     "solve_s": round(wall, 1), "total_s": round(wall, 1)})
        for k, v in det.items():
            rrow[k] = v
        out_rows.append(rrow)
        print(f"n={n}: RL={G:+.2f} ref={ref:+.2f} gap={gap:+.3f}%  ({wall:.0f}s)",
              flush=True)

    out_path = os.path.join(repo, args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print("wrote", out_path)

if __name__ == "__main__":
    main()
