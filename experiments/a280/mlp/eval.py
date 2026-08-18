#a
import argparse
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

import numpy as np
import torch

from env import TTPDEnv
from env.instance import load_a280, milp_instance
from policy.attention_policy import AttentionPolicy
from policy.decoder import DecoderConfig
from policy.encoder import EncoderConfig
from train.eval import (
    build_eval_set, evaluate, load_milp_refs, MILP_PROVEN_OPTIMAL, _rollout_once,
)

def load_policy(checkpoint: str, device: str) -> AttentionPolicy:
    payload = torch.load(ensure_local(checkpoint), map_location=device, weights_only=False)
    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy

def best_of(policy, env, inst, *, n_aug: int, n_starts: int, n_sample: int, temp: float = 1.0):

    env.reset(options={"instance": inst})
    feasible = [int(j) for j in np.flatnonzero(env.current_masks()["j"])
                if j not in (inst.source, inst.sink)]
    starts = feasible if n_starts <= 0 else (feasible[: max(1, n_starts)] or [None])
    best_G = None
    for aug in range(max(1, n_aug)):
        ctx = policy.encode(inst, aug_idx=aug)
        for fj in starts:
            G, _, _ = _rollout_once(policy, env, inst, ctx, fj, True)
            if G is not None and (best_G is None or G > best_G):
                best_G = G
            for _ in range(n_sample):
                G, _, _ = _rollout_once(policy, env, inst, ctx, fj, False, temperature=temp)
                if G is not None and (best_G is None or G > best_G):
                    best_G = G
    return best_G

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a280", default=hub_instance("a280", "a280_benchmark.txt"))
    p.add_argument("--n-instances", type=int, default=40)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--device", default=None)
    p.add_argument("--milp-only", action="store_true",
                   help="only the exact-MILP-instance gap report")
    p.add_argument("--sample", type=int, default=0,
                   help="extra sampled rollouts per (start, aug) on the MILP instances")
    p.add_argument("--sizes", default="5,10,15,20",
                   help="comma-separated benchmark sizes to evaluate")
    p.add_argument("--temp", type=float, default=1.0,
                   help="sampling temperature (>1 flattens the distribution)")
    p.add_argument("--beam", type=int, default=0,
                   help="beam-search width (0 = off)")
    p.add_argument("--beam-stratify", action="store_true",
                   help="POMO-style multi-start over the first launch target (one beam "
                        "per first-launch option; pinned head contributes zero log-prob)")
    p.add_argument("--beam-collect-all", action="store_true",
                   help="restrict the beam to all-collection solutions "
                        "(constrained decode; report separately if used)")
    args = p.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(","))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    milp_dir = os.path.join(ROOT, "..", "MILP Result")
    refs = load_milp_refs(milp_dir if os.path.isdir(milp_dir) else None)
    a280 = load_a280(args.a280)

    print("== TRUE gap: exact MILP instances (scaled capacity, seed=n city draw) ==")
    print(f"   decode: greedy POMO (all starts x 8 augs)"
          + (f" + {args.sample} samples each" if args.sample else ""))
    mean_gaps = []
    for n in sizes:
        ref = refs.get(n)
        if ref is None:
            continue
        inst = milp_instance(a280, n, seed=n, scale_capacity=True)
        env = TTPDEnv(a280_path=args.a280, n=n, scale_capacity=True)
        G = best_of(policy, env, inst, n_aug=8, n_starts=0, n_sample=args.sample, temp=args.temp)
        if args.beam > 0:
            from policy.beam import beam_search
            G_beam = beam_search(policy, env, inst, beam_width=args.beam, n_aug=8,
                                 stratify_k0=args.beam_stratify,
                                 force_collect=args.beam_collect_all)
            if G_beam is not None and (G is None or G_beam > G):
                G = G_beam
        kind = "optimum " if n in MILP_PROVEN_OPTIMAL else "incumbent"
        if G is None:
            print(f"  n={n:<3} FAILED (infeasible rollouts)")
            continue
        gap = (ref - G) / abs(ref) * 100.0
        mean_gaps.append(gap)
        flag = "  <1% TARGET MET" if gap < 1.0 else ""
        print(f"  n={n:<3} RL={G:+.2f}  MILP {kind}={ref:+.2f}  gap={gap:+.3f}%{flag}")
    if mean_gaps:
        print(f"  mean gap over {len(mean_gaps)} sizes: {np.mean(mean_gaps):+.3f}%")

    if not args.milp_only:
        print("== random held-out set ==")
        eval_set = build_eval_set(args.a280, args.n_instances, 5, 20, args.seed)
        res = evaluate(policy, args.a280, eval_set, with_milp_gap=False)
        print(f"  mean={res.mean_return:+.1f}±{res.std_return:.1f}  "
              f"sel={res.select_score:+.3f}  per_size={res.per_size}")

if __name__ == "__main__":
    main()
