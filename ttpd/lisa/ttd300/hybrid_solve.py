from __future__ import annotations
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths
_paths.use("rl/ttd300/gat", "core", "heuristics")

from ttpd.hub import ensure_local 
import time
import multiprocessing as mp

import numpy as np
import torch

from ttd300_io import (GAT, TEMPLATE, LAYOUTS, ED_FRACS,  
                       bench_label, load_ttd300)

from env import TTPDEnv                                    
from env.masking import NO_LAUNCH                          
from policy.attention_policy import AttentionPolicy        
from policy.encoder import EncoderConfig                   
from policy.decoder import DecoderConfig                   
from policy.beam import beam_search                        
from core import evaluate, check_solution                
from sa import sa                                          

CONS_EPS = 1e-3

def _replay_actions_to_solution(inst, actions):
    n = inst.n
    env = TTPDEnv(a280_path=TEMPLATE, n=n, scale_capacity=False)
    obs, _ = env.reset(options={"instance": inst})
    truck: list[int] = []
    sorties: list[tuple] = []
    z = [0] * (n + 2)
    pending = None
    g = 0.0
    for a in actions:
        c = int(obs["c_t"])
        if a["rejoin"] == 1 and pending is not None:
            L, D = pending
            sorties.append((L, D, c))
            if a["z_drone"] == 1:
                z[D] = 1
            pending = None
        if a.get("z_curr", 0) == 1 and 0 < c <= n:
            z[c] = 1
        if a.get("k", NO_LAUNCH) != NO_LAUNCH:
            pending = (c, int(a["k"]))
        if 1 <= a.get("j", inst.sink) <= n:
            truck.append(int(a["j"]))
        obs, r, term, trunc, _ = env.step(a)
        g += float(r)
        if term:
            return tuple(truck), tuple(sorties), tuple(z), g
        if trunc:
            return None
    return None

@torch.no_grad()
def _greedy_actions(policy, inst):
    env = TTPDEnv(a280_path=TEMPLATE, n=inst.n, scale_capacity=False)
    obs, _ = env.reset(options={"instance": inst})
    ctx = policy.encode(inst)
    acts = []
    for _ in range(2 * (inst.n + 2) + 2):
        a = policy.act(obs, env, deterministic=True, ctx=ctx).action
        acts.append(a)
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return acts

@torch.no_grad()
def gat_decode_solution(policy, inst, mode="beam", beam_width=64, n_aug=8):
    if mode == "beam":
        env = TTPDEnv(a280_path=TEMPLATE, n=inst.n, scale_capacity=False)
        env.reset(options={"instance": inst})
        _, acts = beam_search(policy, env, inst, beam_width=beam_width,
                              n_aug=n_aug, return_actions=True)
        if not acts:
            return None
        acts = list(acts)
    else:
        acts = _greedy_actions(policy, inst)
    return _replay_actions_to_solution(inst, acts)

def _sa_job(job):
    P, budget, seed, init = job
    return sa(P, time_limit=budget, seed=seed, init=init).objective

def run(args):
    device = torch.device(args.device)
    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    payload = torch.load(ensure_local(args.weights), map_location=device, weights_only=False)
    ckpt_in = payload["policy"]["decoder.ctx_proj.0.weight"].shape[1]
    if ckpt_in != policy.decoder.ctx_proj[0].in_features:
        import torch.nn as nn
        ckpt_scalars = ckpt_in - 3 * dec_cfg.d_model
        old = policy.decoder.ctx_proj
        new_first = nn.Linear(ckpt_in, old[0].out_features,
                              bias=old[0].bias is not None)
        policy.decoder.ctx_proj = nn.Sequential(new_first, *list(old)[1:]).to(device)
        _orig_ctx = policy.decoder._context

        def _ctx_trunc(h_nodes, c_t, U_mask, scalars, node_mask=None):
            return _orig_ctx(h_nodes, c_t, U_mask,
                             scalars[..., :ckpt_scalars], node_mask=node_mask)

        policy.decoder._context = _ctx_trunc
        print(f"[hybrid] adapted ckpt: {ckpt_scalars} scalars (truncating extra)",
              flush=True)
    missing, unexpected = policy.load_state_dict(payload["policy"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has unknown tensors: {unexpected}")
    bad = [k for k in missing if ".feat_mlp." not in k]
    if bad:
        raise RuntimeError(f"checkpoint is missing non-feature tensors: {bad}")
    if missing:
        print(f"[hybrid] pre-feature checkpoint "
              f"({len(missing)} feat_mlp tensors zero-init)", flush=True)
    policy.eval()
    print(f"[hybrid] {args.weights}  (device={args.device}, decode={args.decode} "
          f"beam={args.beam_width}/{args.n_aug}, budget={args.budget}s, "
          f"jobs={args.jobs})\n", flush=True)

    insts = []                              # (label, P, gat_obj, init solution)
    for n in args.sizes:
        for L in args.layouts:
            for frac in args.ed_fracs:
                P, inst = load_ttd300(n, L, frac)
                label = bench_label(n, L, frac)
                sol = gat_decode_solution(policy, inst, mode=args.decode,
                                          beam_width=args.beam_width,
                                          n_aug=args.n_aug)
                if sol is None:
                    print(f"{label}: GAT decode did not terminate -- skipped",
                          flush=True)
                    continue
                truck, sorties, z, gat_env_obj = sol
                gat_obj = evaluate(P, truck, sorties, z)
                err = check_solution(P, truck, sorties, z)
                if err is not None or abs(gat_obj - gat_env_obj) > CONS_EPS:
                    print(f"{label}: GAT solution inconsistent ({err}) -- skipped",
                          flush=True)
                    continue
                for (Ln, D, J) in sorties:              # endurance must hold
                    reach = P.dist[Ln][D] + P.dist[D][J]
                    assert reach <= inst.ED + 1e-6, \
                        f"{label}: GAT sortie exceeds ED {reach:.2f} > {inst.ED}"
                print(f"{label}: GAT beam decoded (obj {gat_obj:.1f}), SA queued",
                      flush=True)
                insts.append((label, P, gat_obj, (truck, sorties, z)))

    jobs = []
    for (_lbl, P, _g, init) in insts:
        jobs.append((P, args.budget, args.seed, None))              # SA from scratch
        jobs.append((P, args.budget, args.seed, init))              # SA warm-started
    print(f"[hybrid] running {len(jobs)} SA solves on {args.jobs} cores "
          f"(budget {args.budget}s each)...", flush=True)
    if jobs:
        # spawn so the GAT/torch state in the parent never leaks into workers
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.jobs, len(jobs))) as pool:
            res = pool.map(_sa_job, jobs)
    else:
        res = []

    hdr = (f"{'instance':<22} {'GAT':>11} {'SA_scratch':>11} {'SA_warm':>11} "
           f"{'PORTF':>11} {'warm-SA%':>9} {'winner':>7}")
    print(hdr, flush=True)
    agg = {"gat": [], "sa": [], "warm": [], "portf": []}
    for i, (label, P, gat_obj, _init) in enumerate(insts):
        sa_o = res[2 * i]
        warm_o = res[2 * i + 1]
        portf = max(sa_o, warm_o)                                   # no-regret portfolio
        improve = 100.0 * (warm_o - sa_o) / abs(sa_o) if sa_o != 0 else 0.0
        best = max(gat_obj, sa_o, warm_o, portf)
        winner = ("PORTF" if abs(best - portf) < 1e-6 else
                  "WARM" if abs(best - warm_o) < 1e-6 else
                  "SA" if abs(best - sa_o) < 1e-6 else "GAT")
        print(f"{label:<22} {gat_obj:11.1f} {sa_o:11.1f} "
              f"{warm_o:11.1f} {portf:11.1f} {improve:+9.2f} {winner:>7}", flush=True)
        agg["gat"].append(gat_obj); agg["sa"].append(sa_o)
        agg["warm"].append(warm_o); agg["portf"].append(portf)

    if agg["sa"]:
        n_ok = len(agg["sa"])
        gat_m = float(np.mean(agg["gat"])); sa_m = float(np.mean(agg["sa"]))
        warm_m = float(np.mean(agg["warm"])); portf_m = float(np.mean(agg["portf"]))
        warm_wins = sum(w >= s - 1e-6 for w, s in zip(agg["warm"], agg["sa"]))
        portf_wins = sum(p >= s - 1e-6 for p, s in zip(agg["portf"], agg["sa"]))
        print("-" * len(hdr))
        print(f"{'MEAN':<22} {gat_m:11.1f} {sa_m:11.1f} {warm_m:11.1f} "
              f"{portf_m:11.1f} {100.0*(warm_m-sa_m)/abs(sa_m):+9.2f}")
        print(f"\nSA_warm >= SA on {warm_wins}/{n_ok}  |  "
              f"PORTFOLIO >= SA on {portf_wins}/{n_ok} (no-regret) "
              f"| mean portfolio vs SA: {100.0*(portf_m-sa_m)/abs(sa_m):+.2f}%")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=str, required=True, help="BC checkpoint")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10])
    ap.add_argument("--layouts", type=int, nargs="+", default=LAYOUTS)
    ap.add_argument("--ed-fracs", type=float, nargs="+", default=ED_FRACS)
    ap.add_argument("--budget", type=float, default=5.0, help="SA time budget (s)")
    ap.add_argument("--decode", choices=["beam", "greedy"], default="beam",
                    help="GAT warm-start decode method")
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="parallel SA solves across cores (default: all cores)")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
