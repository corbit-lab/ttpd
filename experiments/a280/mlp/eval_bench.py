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
_paths.use("rl/a280/mlp", "core")

from ttpd.hub import ensure_local  # noqa: E402
import time

import torch  # noqa: E402

from env import TTPDEnv  # noqa: E402
from env.instance import load_a280, _build_instance  # noqa: E402
from env.masking import NO_LAUNCH  # noqa: E402
from policy.attention_policy import AttentionPolicy  # noqa: E402
from policy.beam import beam_search  # noqa: E402
from policy.decoder import DecoderConfig  # noqa: E402
from policy.encoder import EncoderConfig  # noqa: E402

from core import evaluate, check_solution, problem_from_instance  # noqa: E402  the solver core

CONS_EPS = 1e-3

def load_ckpt_adapted(checkpoint: str, device: str) -> AttentionPolicy:
    payload = torch.load(ensure_local(checkpoint), map_location=device, weights_only=False)
    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)

    ckpt_in = payload["policy"]["decoder.ctx_proj.0.weight"].shape[1]
    D = dec_cfg.d_model
    ckpt_scalars = ckpt_in - 3 * D

    if ckpt_in != policy.decoder.ctx_proj[0].in_features:
        import torch.nn as nn
        old = policy.decoder.ctx_proj
        new_first = nn.Linear(ckpt_in, old[0].out_features,
                              bias=old[0].bias is not None)
        policy.decoder.ctx_proj = nn.Sequential(new_first, *list(old)[1:]).to(device)
        orig_context = policy.decoder._context

        def _context_trunc(h_nodes, c_t, U_mask, scalars, node_mask=None):
            return orig_context(h_nodes, c_t, U_mask,
                                scalars[..., :ckpt_scalars], node_mask=node_mask)

        policy.decoder._context = _context_trunc
        print(f"    [adapt] ckpt expects {ckpt_scalars} scalars; truncating")

    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy

def load_bench_instance(path):
    data = load_a280(path)
    best = data["best_item"]
    chosen = [(nid, float(p), float(w)) for nid, (p, w) in sorted(best.items())]
    inst = _build_instance(data, chosen, depot_id=1, R=None, scale_capacity=False)
    return inst

def replay_actions_to_solution(bench_path, n, inst, actions):
    env = TTPDEnv(a280_path=bench_path, n=n, scale_capacity=False)
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

def beam_decode_validated(policy, bench_path, n, inst, beam_width, n_aug):
    env = TTPDEnv(a280_path=bench_path, n=n, scale_capacity=False)
    env.reset(options={"instance": inst})
    _, acts = beam_search(policy, env, inst, beam_width=beam_width, n_aug=n_aug,
                          return_actions=True)
    if not acts:
        return None, "beam produced no actions"
    sol = replay_actions_to_solution(bench_path, n, inst, list(acts))
    if sol is None:
        return None, "replay truncated (incomplete tour)"
    truck, sorties, z, env_obj = sol
    P = problem_from_instance(inst)
    true_obj = evaluate(P, truck, sorties, z)
    err = check_solution(P, truck, sorties, z)
    if err is not None:
        return None, f"check_solution failed: {err}"
    if abs(true_obj - env_obj) > CONS_EPS:
        return None, f"env/true objective mismatch ({env_obj:.2f} vs {true_obj:.2f})"
    return true_obj, None

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--sizes", default="20")
    p.add_argument("--inst", type=int, default=5)
    p.add_argument("--beam", type=int, default=256)
    p.add_argument("--n-aug", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-map", required=True,
                   help="n:path[,n:path...] checkpoint map")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [int(s) for s in args.sizes.split(",")]
    ckpt_map = {}
    for pair in args.ckpt_map.split(","):
        k, v = pair.split(":", 1)
        ckpt_map[int(k)] = v

    rows = []
    for n in sizes:
        ckpt = ckpt_map.get(n)
        bench = os.path.join(args.bench_dir, f"bench_{n}_{args.inst}.txt")
        bench = ensure_local(bench)
        if not ckpt or not os.path.exists(ckpt):
            print(f"  [skip] n={n}: no checkpoint")
            continue
        if not os.path.exists(bench):
            print(f"  [skip] n={n}: no bench file {bench}")
            continue
        policy = load_ckpt_adapted(ckpt, device)
        inst = load_bench_instance(bench)
        t0 = time.perf_counter()
        obj, err = beam_decode_validated(policy, bench, n, inst, args.beam, args.n_aug)
        wall = time.perf_counter() - t0
        if err is not None:
            print(f"  n={n:<3} beam={args.beam}  SKIPPED: {err}  t={wall:.1f}s")
            continue
        rows.append({"method": "MLP", "n": n, "instance": f"bench_{n}_{args.inst}",
                     "beam": args.beam, "objective": obj, "time_s": round(wall, 4)})
        print(f"  n={n:<3} beam={args.beam}  obj={obj:.4f} (validated)  t={wall:.1f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "instance", "beam",
                                          "objective", "time_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
