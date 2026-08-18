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
_paths.use("rl/a280/gat")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

ROOT = _ROOT
REPO = _ROOT
import time

import numpy as np
import torch

from env import TTPDEnv
from env.instance import load_a280, milp_instance
from policy.beam import beam_search
from solution_csv import load_policy, _trace_rollout, replay_details

def _starts(env, inst):
    env.reset(options={"instance": inst})
    return [int(j) for j in np.flatnonzero(env.current_masks()["j"])
            if j not in (inst.source, inst.sink)]

def light_decode(policy, env, inst, *, n_aug, n_sample, seed, beam_width=0):

    torch.manual_seed(seed)
    best = (None, None)
    starts = _starts(env, inst)
    for aug in range(n_aug):
        ctx = policy.encode(inst, aug_idx=aug)
        for fj in starts:
            G, acts = _trace_rollout(policy, env, inst, ctx, fj, True)   # greedy
            if G is not None and (best[0] is None or G > best[0]):
                best = (G, acts)
            for _ in range(n_sample):
                G, acts = _trace_rollout(policy, env, inst, ctx, fj, False)  # sample
                if G is not None and (best[0] is None or G > best[0]):
                    best = (G, acts)
    if beam_width > 0:
        Gb, acts_b = beam_search(policy, env, inst, beam_width=beam_width,
                                 n_aug=n_aug, stratify_k0=True, return_actions=True)
        if Gb is not None and (best[0] is None or Gb > best[0]):
            best = (Gb, acts_b)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained n=20 weights")
    ap.add_argument("--sizes", default="50,100")
    ap.add_argument("--a280", default=hub_instance("a280", "a280_benchmark.txt"))
    ap.add_argument("--n-aug", type=int, default=8, help="dihedral augmentations")
    ap.add_argument("--sample", type=int, default=16,
                    help="sampled rollouts per (start, aug); keep small for speed")
    ap.add_argument("--beam", type=int, default=0,
                    help="beam-search width (0 = off); combined best-of with sampling")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "generalization_n50_n100.csv"))
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    sizes = [int(s) for s in args.sizes.split(",")]
    a280 = load_a280(args.a280)
    policy = load_policy(args.checkpoint, device)
    print(f"device={device}  checkpoint={args.checkpoint}")
    print(f"decode: POMO greedy (all starts x {args.n_aug} augs) + {args.sample} samples each"
          + (f" + beam{args.beam}" if args.beam else "") + "\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["n", "objective", "profit", "rental", "arrival", "W_final",
            "n_items", "n_truck", "n_drone", "n_rendv", "wall_s",
            "truck_route_node_ids", "drone_arc_node_ids", "collected_node_ids",
            "truck_node_ids", "drone_node_ids", "rendezvous_node_ids", "W_sequence"]
    rows = []
    for n in sizes:
        inst = milp_instance(a280, n, seed=n, scale_capacity=True)
        env = TTPDEnv(a280_path=args.a280, n=n, scale_capacity=True)
        t0 = time.perf_counter()
        G, acts = light_decode(policy, env, inst, n_aug=args.n_aug,
                               n_sample=args.sample, seed=n, beam_width=args.beam)
        dt = time.perf_counter() - t0
        if G is None or acts is None:
            print(f"n={n:<4} FAILED (no feasible rollout)")
            continue
        det = replay_details(inst, args.a280, acts)
        row = {"n": n, "wall_s": round(dt, 1)}
        for c in cols:
            if c in det and c not in row:
                v = det[c]
                row[c] = v if not isinstance(v, (list, tuple)) else str(v)
        rows.append(row)
        print(f"==== n={n} ({dt:.1f}s) ====")
        print(f"  objective      : {det['objective']:.2f}")
        print(f"  profit         : {det['profit']:.1f}")
        print(f"  rental         : {det['rental']:.1f}")
        print(f"  arrival (time) : {det['arrival']:.1f}")
        print(f"  W_final        : {det.get('W_final')}")
        print(f"  items total    : {det['n_items']}  (truck={det['n_truck']}, drone={det['n_drone']})")
        print(f"  rendezvous pts : {det['n_rendv']}")
        print(f"  truck route    : {det['truck_route_node_ids']}")
        print(f"  drone arcs     : {det['drone_arc_node_ids']}")
        print(f"  rendezvous ids : {det['rendezvous_node_ids']}\n")

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
