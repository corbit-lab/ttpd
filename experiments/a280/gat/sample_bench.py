
from __future__ import annotations

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

from ttpd.hub import ensure_local  # noqa: E402

ROOT = _ROOT
import time

import torch  # noqa: E402

from env import TTPDEnv  # noqa: E402
from env.instance import load_a280, _build_instance  # noqa: E402
from policy.attention_policy import AttentionPolicy  # noqa: E402
from policy.decoder import DecoderConfig  # noqa: E402
from policy.encoder import EncoderConfig  # noqa: E402
from eval import best_of  # noqa: E402  (best_of defined in scripts/eval.py)

def load_ckpt_adapted(checkpoint: str, device: str) -> AttentionPolicy:
    payload = torch.load(ensure_local(checkpoint), map_location=device, weights_only=False)
    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    ckpt_in = payload["policy"]["decoder.ctx_proj.0.weight"].shape[1]
    if ckpt_in != policy.decoder.ctx_proj[0].in_features:
        import torch.nn as nn
        ckpt_scalars = ckpt_in - 3 * dec_cfg.d_model
        old = policy.decoder.ctx_proj
        new_first = nn.Linear(ckpt_in, old[0].out_features, bias=old[0].bias is not None)
        policy.decoder.ctx_proj = nn.Sequential(new_first, *list(old)[1:]).to(device)
        orig = policy.decoder._context

        def _ctx_trunc(h_nodes, c_t, U_mask, scalars, node_mask=None):
            return orig(h_nodes, c_t, U_mask, scalars[..., :ckpt_scalars], node_mask=node_mask)

        policy.decoder._context = _ctx_trunc
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy

def load_bench_instance(path):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    return _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--items", required=True,
                   help="comma list of n:inst pairs, e.g. 10:3,15:5,5:4")
    p.add_argument("--ckpt-map", required=True, help="n:path[,n:path...]")
    p.add_argument("--n-sample", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--n-aug", type=int, default=8)
    p.add_argument("--seed", type=int, default=2004)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = [tuple(int(x) for x in it.split(":")) for it in args.items.split(",")]
    ckpt_map = {}
    for pr in args.ckpt_map.split(","):
        k, v = pr.split(":", 1)
        ckpt_map[int(k)] = v

    rows = []
    for n, inst_no in pairs:
        ckpt = ckpt_map.get(n)
        bench = os.path.join(args.bench_dir, f"bench_{n}_{inst_no}.txt")
        bench = ensure_local(bench)
        if not ckpt or not os.path.exists(ckpt) or not os.path.exists(bench):
            print(f"  [skip] n={n} inst={inst_no}: missing ckpt/bench")
            continue
        policy = load_ckpt_adapted(ckpt, device)
        inst = load_bench_instance(bench)
        env = TTPDEnv(a280_path=bench, n=n, scale_capacity=False)
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        # best_of: greedy POMO pass + n_sample sampled rollouts per (start, aug)
        G = best_of(policy, env, inst, n_aug=args.n_aug, n_starts=0,
                    n_sample=args.n_sample, temp=args.temp)
        wall = time.perf_counter() - t0
        obj = None if G is None else float(G)
        rows.append({"model_dir": os.path.basename(ROOT), "n": n,
                     "instance": f"bench_{n}_{inst_no}", "temp": args.temp,
                     "n_sample": args.n_sample, "seed": args.seed,
                     "objective": obj, "time_s": round(wall, 3)})
        print(f"  bench_{n}_{inst_no}  temp={args.temp}  nS={args.n_sample}  "
              f"obj={obj}  t={wall:.1f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_dir", "n", "instance", "temp",
                                          "n_sample", "seed", "objective", "time_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
