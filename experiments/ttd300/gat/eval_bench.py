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
_paths.use("rl/ttd300/gat", "core")

from ttpd.hub import (ensure_local, ensure_bench_dir,  # noqa: E402
                      weights_dir as hub_weights)

ROOT = _ROOT
import time

DATA_DIR = ensure_bench_dir("ttd300")

import torch  # noqa: E402

from env import TTPDEnv  # noqa: E402
from env.instance import full_instance, load_ttd300  # noqa: E402
from env.masking import NO_LAUNCH  # noqa: E402
from policy.attention_policy import AttentionPolicy  # noqa: E402
from policy.beam import beam_search  # noqa: E402
from policy.decoder import DecoderConfig  # noqa: E402
from policy.encoder import EncoderConfig  # noqa: E402

from core import evaluate, check_solution, problem_from_instance  # noqa: E402  solver core

CONS_EPS = 1e-3

def load_ckpt(checkpoint: str, device: str) -> AttentionPolicy:
    payload = torch.load(ensure_local(checkpoint), map_location=device, weights_only=False)
    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    # Pre-feature checkpoints lack the pointer heads' feat_mlp tensors; those are
    # zero-initialised (bias contributes exactly 0), so loading non-strictly is
    # numerically identical to evaluating under the old architecture. Anything
    # else missing/unexpected is a real mismatch and still fails loudly.
    missing, unexpected = policy.load_state_dict(payload["policy"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has unknown tensors: {unexpected}")
    bad = [k for k in missing if ".feat_mlp." not in k]
    if bad:
        raise RuntimeError(f"checkpoint is missing non-feature tensors: {bad}")
    if missing:
        print(f"  [info] {os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))}: "
              f"pre-feature checkpoint ({len(missing)} feat_mlp tensors zero-init)")
    policy.eval()
    return policy

def replay_actions_to_solution(inst, actions):
    """Mirror a280 eval_bench: replay the beam's action sequence into a
    (truck, sorties, z) solution. Returns (truck, sorties, z, env_return) or
    None if the episode truncates (not a valid complete tour)."""
    n = inst.n
    env = TTPDEnv(n=n, scale_capacity=False)
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

def beam_decode_validated(policy, inst, beam_width, n_aug):
    """beam_search -> replay -> core.evaluate/check_solution (ED re-checked).
    Returns (validated objective, None) or (None, reason)."""
    env = TTPDEnv(n=inst.n, scale_capacity=False)
    env.reset(options={"instance": inst})
    _, acts = beam_search(policy, env, inst, beam_width=beam_width, n_aug=n_aug,
                          return_actions=True)
    if not acts:
        return None, "beam produced no actions"
    sol = replay_actions_to_solution(inst, list(acts))
    if sol is None:
        return None, "replay truncated (incomplete tour)"
    truck, sorties, z, env_obj = sol
    P = problem_from_instance(inst)
    true_obj = evaluate(P, truck, sorties, z)
    if true_obj is None:
        return None, "core.evaluate rejects (anchors/capacity/endurance)"
    err = check_solution(P, truck, sorties, z)
    if err is not None:
        return None, f"check_solution failed: {err}"
    if abs(true_obj - env_obj) > CONS_EPS:
        return None, f"env/true objective mismatch ({env_obj:.2f} vs {true_obj:.2f})"
    return true_obj, None

def load_bks(csv_path: str | None) -> dict:
    """instance stem -> best-known objective (max over methods/seeds) from a
    benchmark-sweep results.csv-style file."""
    if not csv_path:
        return {}
    bks: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            key = row["instance"]
            try:
                obj = float(row["objective"])
            except (KeyError, ValueError):
                continue
            if key not in bks or obj > bks[key]:
                bks[key] = obj
    return bks

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--ckpt-dir", default=hub_weights("ttd300", "gat", "benchmark-tuned"),
                   help="dir with <tag>_n<size>/checkpoints/best.pt")
    p.add_argument("--ckpt-map", default=None,
                   help="explicit n:path[,n:path...] checkpoint map (overrides --ckpt-dir)")
    p.add_argument("--tag", default="gat", help="checkpoint dir prefix (gat/mlp)")
    p.add_argument("--sizes", default="10,20,30,40,50,75,100")
    p.add_argument("--layouts", default="1,2,3,4,5")
    p.add_argument("--fracs", default="f025,f050,f075,f100")
    p.add_argument("--beam", type=int, default=256)
    p.add_argument("--n-aug", type=int, default=8)
    p.add_argument("--bks-csv", default=None,
                   help="benchmark-sweep results.csv for best-known refs; adds bks + gap_pct columns")
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [int(s) for s in args.sizes.split(",")]
    layouts = [int(s) for s in args.layouts.split(",")]
    fracs = [s.strip() for s in args.fracs.split(",")]
    bks = load_bks(args.bks_csv)

    ckpt_map = {}
    if args.ckpt_map:
        for pair in args.ckpt_map.split(","):
            k, v = pair.split(":", 1)
            ckpt_map[int(k)] = v

    method = args.tag.upper()
    rows = []
    for n in sizes:
        ckpt = ckpt_map.get(
            n, os.path.join(args.ckpt_dir, f"n{n}", "best.pt"))
        if not os.path.exists(ckpt):
            print(f"  [skip] n={n}: no checkpoint {ckpt}")
            continue
        policy = load_ckpt(ckpt, device)
        for L in layouts:
            for frac in fracs:
                stem = f"ttd300_n{n}_L{L}_{frac}"
                path = os.path.join(args.data_dir, stem + ".txt")
                if not os.path.exists(path):
                    print(f"  [skip] {stem}: no data file")
                    continue
                inst = full_instance(load_ttd300(path), scale_capacity=False)
                t0 = time.perf_counter()
                obj, err = beam_decode_validated(policy, inst, args.beam, args.n_aug)
                wall = time.perf_counter() - t0
                if err is not None:
                    print(f"  {stem:<26} beam={args.beam}  SKIPPED: {err}  t={wall:.1f}s")
                    continue
                row = {"method": method, "n": n, "layout": L, "frac": frac,
                       "instance": stem, "ED": inst.ED, "beam": args.beam,
                       "objective": round(obj, 4), "time_s": round(wall, 4)}
                if stem in bks:
                    ref = bks[stem]
                    row["bks"] = round(ref, 4)
                    row["gap_pct"] = round((ref - obj) / abs(ref) * 100.0, 4)
                rows.append(row)
                extra = (f"  bks={row['bks']:.1f} gap={row['gap_pct']:+.2f}%"
                         if "bks" in row else "")
                print(f"  {stem:<26} beam={args.beam}  obj={obj:.4f} (validated)"
                      f"{extra}  t={wall:.1f}s")

    fieldnames = ["method", "n", "layout", "frac", "instance", "ED", "beam",
                  "objective", "time_s"]
    if bks:
        fieldnames += ["bks", "gap_pct"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
