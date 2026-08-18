import argparse
import csv
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("rl/a280/gat")

from ttpd.hub import ensure_local, weights_dir as hub_weights, instance as hub_instance  # noqa: E402

ROOT = _ROOT
REPO = _ROOT
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("ABLATION_THREADS", "8")))

from env import TTPDEnv
from env.instance import load_a280, milp_instance
from policy.beam import beam_search
from policy.eas import eas_emb
from solution_csv import load_policy, _trace_rollout, replay_details, COLUMNS
from eval_bench import load_ckpt_adapted

A280 = hub_instance("a280", "a280_benchmark.txt")
# the deliverable checkpoint per size, see docs/weights.md
# (n5/n10 from the specialist runs, n15/n20 from the sampled runs)
CKPTS = {5: os.path.join("specialists", "n5.pt"),
         10: os.path.join("specialists", "n10.pt"),
         15: os.path.join("sampled", "n15", "best.pt"),
         20: os.path.join("sampled", "n20", "best.pt")}
# env-exact optima (= exact eval of MILP solution = GVNS best); proven optimal n<=15
REF = {5: -16725.35, 10: -27493.1662, 15: -36385.6806, 20: -40090.7533}
PROVEN_OPT = {5, 10, 15}

STOCHASTIC = {"sampling", "eas", "eas_beam"}

def _starts(env, inst):
    env.reset(options={"instance": inst})
    return [int(j) for j in np.flatnonzero(env.current_masks()["j"])
            if j not in (inst.source, inst.sink)]

def _greedy_over(policy, inst, env, augs, starts):
    best = (None, None)
    for aug in range(augs):
        ctx = policy.encode(inst, aug_idx=aug)
        for fj in starts:
            G, acts = _trace_rollout(policy, env, inst, ctx, fj, True)
            if G is not None and (best[0] is None or G > best[0]):
                best = (G, acts)
    return best

def _tuned_ctx(policy, inst, h):
    ctx = policy.encode(inst, aug_idx=0)
    try:
        ctx.h_nodes = h
    except Exception:
        pass
    return ctx

def run_method(name, policy, env, inst, seed, *, n_sample, beam_width,
               eas_iters, eas_k):
    if name == "greedy":
        ctx = policy.encode(inst, aug_idx=0)
        return _trace_rollout(policy, env, inst, ctx, None, True)
    if name == "pomo":
        return _greedy_over(policy, inst, env, 1, _starts(env, inst))
    if name == "pomo_aug":
        return _greedy_over(policy, inst, env, 8, _starts(env, inst))
    if name == "sampling":
        torch.manual_seed(seed)
        best = _greedy_over(policy, inst, env, 8, _starts(env, inst))
        for aug in range(8):
            ctx = policy.encode(inst, aug_idx=aug)
            for fj in _starts(env, inst):
                for _ in range(n_sample):
                    G, acts = _trace_rollout(policy, env, inst, ctx, fj, False)
                    if G is not None and (best[0] is None or G > best[0]):
                        best = (G, acts)
        return best
    if name == "beam":
        return beam_search(policy, env, inst, beam_width=beam_width, n_aug=8,
                           stratify_k0=True, return_actions=True)
    if name in ("eas", "eas_beam"):
        torch.manual_seed(seed)
        env.reset(options={"instance": inst})
        env.inst = inst
        _, h = eas_emb(policy, env, iters=eas_iters, k=eas_k, return_embeddings=True)
        ctx = _tuned_ctx(policy, inst, h)
        if name == "eas_beam":
            try:
                G, acts = beam_search(policy, env, inst, ctx_override=ctx,
                                      beam_width=beam_width, return_actions=True)
                if G is not None:
                    return G, acts
            except Exception:
                pass
        # eas: greedy + a few sampled traced rollouts on the tuned embeddings
        best = _trace_rollout(policy, env, inst, ctx, None, True)
        for _ in range(max(8, n_sample // 4)):
            G, acts = _trace_rollout(policy, env, inst, ctx, None, False)
            if G is not None and (best[0] is None or G > best[0]):
                best = (G, acts)
        return best
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="5,10,15,20")
    ap.add_argument("--methods",
                    default="greedy,pomo,pomo_aug,sampling,beam,eas,eas_beam")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--sample", type=int, default=128)
    ap.add_argument("--beam", type=int, default=512)
    ap.add_argument("--eas-iters", type=int, default=50)
    ap.add_argument("--eas-k", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sizes = [int(s) for s in args.sizes.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    a280 = load_a280(A280)
    print(f"device={device}  sizes={sizes}  methods={methods}  seeds={args.seeds}",
          flush=True)

    run_cols = ["n", "method", "seed"] + COLUMNS[3:]   # drop dup n/method/status pos
    run_cols = ["n", "method", "seed", "objective", "gap_vs_milp_%", "time_s",
                "n_items", "n_truck", "n_drone", "n_rendv", "W_final",
                "profit", "rental", "arrival",
                "truck_route_node_ids", "drone_arc_node_ids", "collected_node_ids",
                "truck_node_ids", "drone_node_ids", "rendezvous_node_ids",
                "W_sequence"]
    runs_path = os.path.join(args.outdir, "drl_ablation_runs.csv")
    runs_f = open(runs_path, "w", newline="")
    runs_w = csv.DictWriter(runs_f, fieldnames=run_cols)
    runs_w.writeheader()

    summary = []
    for n in sizes:
        policy = load_ckpt_adapted(os.path.join(hub_weights("a280", "gat"), CKPTS[n]), device)
        inst = milp_instance(a280, n, seed=n, scale_capacity=True)
        ref = REF[n]
        for method in methods:
            n_runs = args.seeds if method in STOCHASTIC else 1
            gaps, objs, times = [], [], []
            for seed in range(n_runs):
                env = TTPDEnv(a280_path=A280, n=n, scale_capacity=True)
                t0 = time.perf_counter()
                G, acts = run_method(method, policy, env, inst, seed,
                                     n_sample=args.sample, beam_width=args.beam,
                                     eas_iters=args.eas_iters, eas_k=args.eas_k)
                dt = time.perf_counter() - t0
                if G is None:
                    print(f"  n={n} {method} seed={seed} FAILED", flush=True)
                    continue
                gap = (ref - G) / abs(ref) * 100.0
                det = replay_details(inst, A280, acts) if acts is not None else {}
                row = {"n": n, "method": method, "seed": seed,
                       "objective": round(G, 4), "gap_vs_milp_%": round(gap, 4),
                       "time_s": round(dt, 3)}
                for c in run_cols:
                    if c in det and c not in row:
                        v = det[c]
                        row[c] = v if not isinstance(v, (list, tuple)) else str(v)
                runs_w.writerow({c: row.get(c, "") for c in run_cols})
                runs_f.flush()
                gaps.append(gap); objs.append(G); times.append(dt)
            if not objs:
                continue
            summary.append({
                "n": n, "method": method, "n_runs": len(objs),
                "obj_mean": round(statistics.mean(objs), 4),
                "obj_std": round(statistics.pstdev(objs), 4) if len(objs) > 1 else 0.0,
                "obj_best": round(max(objs), 4), "obj_worst": round(min(objs), 4),
                "gap_mean_%": round(statistics.mean(gaps), 4),
                "gap_best_%": round(min(gaps), 4),
                "time_mean_s": round(statistics.mean(times), 3),
                "ref_optimum": ref, "proven_optimal": n in PROVEN_OPT,
            })
            s = summary[-1]
            print(f"  n={n:<2} {method:<9} gap_mean={s['gap_mean_%']:+.3f}% "
                  f"gap_best={s['gap_best_%']:+.3f}%  obj={s['obj_mean']:+.1f}"
                  f"±{s['obj_std']:.1f}  t={s['time_mean_s']:.2f}s "
                  f"(x{s['n_runs']})", flush=True)
    runs_f.close()

    sum_path = os.path.join(args.outdir, "drl_ablation_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {runs_path}\nwrote {sum_path}", flush=True)

if __name__ == "__main__":
    main()
