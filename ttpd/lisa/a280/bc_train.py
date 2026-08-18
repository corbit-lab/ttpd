from __future__ import annotations

import argparse
import json
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
import time
from dataclasses import dataclass

import numpy as np
import torch

GAT = os.path.join(ROOT, "GAT")
A280 = hub_instance("a280", "a280_benchmark.txt")

from env import TTPDEnv                                   # noqa: E402
from env.instance import TTPDInstance                    # noqa: E402
from env.masking import NO_LAUNCH                         # noqa: E402
from policy.attention_policy import AttentionPolicy      # noqa: E402
from policy.decoder import DecoderConfig                  # noqa: E402
from policy.encoder import EncoderConfig                  # noqa: E402

ENV_EPS = 1e-3   # cross-env replay tolerance (solver-core solve vs GAT env)

def instance_from_dict(d: dict) -> TTPDInstance:
    return TTPDInstance(
        n=int(d["n"]),
        coords=np.asarray(d["coords"], dtype=float),
        profits=np.asarray(d["profits"], dtype=float),
        weights=np.asarray(d["weights"], dtype=float),
        dist=np.asarray(d["dist"], dtype=float),
        W=float(d["W"]), v_max=float(d["v_max"]), v_min=float(d["v_min"]),
        v_D=float(d["v_D"]), R=float(d["R"]), node_ids=list(d["node_ids"]),
    )

@dataclass
class StepRec:
    c_t: int
    U_mask: np.ndarray
    scalars: np.ndarray
    drone_slot_local: int
    a_rej: int
    a_zd: int
    a_zc: int
    a_k_local: int
    a_j: int
    m_rej: np.ndarray
    m_zd: np.ndarray
    m_zc: np.ndarray
    m_k_local: np.ndarray
    m_j: np.ndarray

def replay_to_steps(inst, actions) -> tuple[list[StepRec], float]:
    n = inst.n
    local_dummy = n + 2
    env = TTPDEnv(a280_path=A280, n=n, scale_capacity=False)
    obs, _ = env.reset(options={"instance": inst})

    steps: list[StepRec] = []
    g = 0.0
    for a in actions:
        rej = int(a.get("rejoin", 0))
        zd = int(a.get("z_drone", 0))
        zc = int(a.get("z_curr", 0))
        k = int(a.get("k", NO_LAUNCH))
        j = int(a.get("j", inst.sink))
        k_local = local_dummy if k == NO_LAUNCH else k
        in_flight = bool(obs["in_flight"])
        drone_slot = int(obs["drone_D"]) if in_flight else local_dummy

        m_rej = np.asarray(env.current_masks()["rejoin"], dtype=bool)
        m_zd = (np.asarray(env.mask_z_drone_now(), dtype=bool) if rej == 1
                else np.array([True, False], dtype=bool))
        m_zc = np.asarray(env.mask_z_curr_after(rej, zd), dtype=bool)
        m_k = np.asarray(env.masks_for_launch(rejoined=(rej == 1))["k_ext"], dtype=bool)
        m_j = np.asarray(env.mask_j_after_launch(k, rejoined=(rej == 1)), dtype=bool)

        steps.append(StepRec(
            c_t=int(obs["c_t"]), U_mask=np.asarray(obs["U_mask"], dtype=bool).copy(),
            scalars=np.array(
                [obs["W_norm"], obs["tau_norm"], obs["U_frac"], obs["R_norm"],
                 obs["drone_norm"], obs["drone_wait_norm"], obs["drone_elapsed_norm"]],
                dtype=np.float32),
            drone_slot_local=drone_slot,
            a_rej=rej, a_zd=zd, a_zc=zc, a_k_local=k_local, a_j=j,
            m_rej=m_rej, m_zd=m_zd, m_zc=m_zc, m_k_local=m_k, m_j=m_j,
        ))

        obs, r, term, trunc, _ = env.step(a)
        g += float(r)
        if term or trunc:
            break
    return steps, g

@dataclass
class BCBuffer:
    feats: torch.Tensor
    n_per_encoder: torch.Tensor
    encoder_idx: torch.Tensor
    c_t: torch.Tensor
    U_mask: torch.Tensor
    scalars: torch.Tensor
    drone_D_slot: torch.Tensor
    a_rej: torch.Tensor
    a_zd: torch.Tensor
    a_zc: torch.Tensor
    a_k_slot: torch.Tensor
    a_j: torch.Tensor
    m_rej: torch.Tensor
    m_zd: torch.Tensor
    m_zc: torch.Tensor
    m_k_ext: torch.Tensor
    m_j: torch.Tensor
    n_max: int

    def __len__(self):
        return int(self.c_t.shape[0])

def build_buffer(records, policy, device, n_aug: int) -> BCBuffer:
    feats_list, n_per_enc = [], []
    enc_idx = []
    c_t, U_mask_l, scal_l, drone_l = [], [], [], []
    a_rej_l, a_zd_l, a_zc_l, a_k_l, a_j_l = [], [], [], [], []
    m_rej_l, m_zd_l, m_zc_l, m_k_l, m_j_l = [], [], [], [], []
    next_enc = 0
    n_consistency_fail = 0

    for rec in records:
        inst = instance_from_dict(rec["instance"])
        steps, env_obj = replay_to_steps(inst, rec["actions"])
        if abs(env_obj - rec["objective"]) > ENV_EPS:
            n_consistency_fail += 1
            continue

        for aug in range(n_aug):
            feats = policy.build_encoder_inputs(inst, device, aug).squeeze(0).detach().cpu()
            feats_list.append(feats)
            n_per_enc.append(inst.n)
            eid = next_enc
            next_enc += 1
            for s in steps:
                enc_idx.append(eid)
                c_t.append(s.c_t); U_mask_l.append(s.U_mask); scal_l.append(s.scalars)
                drone_l.append(s.drone_slot_local)
                a_rej_l.append(s.a_rej); a_zd_l.append(s.a_zd); a_zc_l.append(s.a_zc)
                a_k_l.append(s.a_k_local); a_j_l.append(s.a_j)
                m_rej_l.append(s.m_rej); m_zd_l.append(s.m_zd); m_zc_l.append(s.m_zc)
                m_k_l.append(s.m_k_local); m_j_l.append(s.m_j)

    if n_consistency_fail:
        print(f"[bc] WARNING: {n_consistency_fail} records failed GAT-env replay "
              f"consistency and were skipped")
    if not enc_idx:
        raise RuntimeError("No usable training steps -- dataset empty or all skipped.")

    E = len(feats_list)
    N_max = max(f.shape[0] for f in feats_list)
    R = len(enc_idx)
    F = feats_list[0].shape[1]

    feats = torch.zeros(E, N_max, F)
    for e, f in enumerate(feats_list):
        feats[e, : f.shape[0]] = f

    U_mask = torch.zeros(R, N_max, dtype=torch.bool)
    m_j = torch.zeros(R, N_max, dtype=torch.bool)
    m_k_ext = torch.zeros(R, N_max + 1, dtype=torch.bool)
    for r in range(R):
        nrec = U_mask_l[r].shape[0]
        U_mask[r, :nrec] = torch.as_tensor(U_mask_l[r])
        m_j[r, :nrec] = torch.as_tensor(m_j_l[r])
        m_k_ext[r, :nrec] = torch.as_tensor(m_k_l[r][:nrec])
        m_k_ext[r, N_max] = bool(m_k_l[r][nrec])

    def remap(vals):
        out = []
        for r, v in enumerate(vals):
            nrec = U_mask_l[r].shape[0]
            out.append(N_max if int(v) == nrec else int(v))
        return out

    return BCBuffer(
        feats=feats,
        n_per_encoder=torch.tensor(n_per_enc, dtype=torch.long),
        encoder_idx=torch.tensor(enc_idx, dtype=torch.long),
        c_t=torch.tensor(c_t, dtype=torch.long),
        U_mask=U_mask,
        scalars=torch.tensor(np.stack(scal_l), dtype=torch.float32),
        drone_D_slot=torch.tensor(remap(drone_l), dtype=torch.long),
        a_rej=torch.tensor(a_rej_l, dtype=torch.long),
        a_zd=torch.tensor(a_zd_l, dtype=torch.long),
        a_zc=torch.tensor(a_zc_l, dtype=torch.long),
        a_k_slot=torch.tensor(remap(a_k_l), dtype=torch.long),
        a_j=torch.tensor(a_j_l, dtype=torch.long),
        m_rej=torch.tensor(np.stack(m_rej_l)),
        m_zd=torch.tensor(np.stack(m_zd_l)),
        m_zc=torch.tensor(np.stack(m_zc_l)),
        m_k_ext=m_k_ext, m_j=m_j, n_max=N_max,
    )

def _to_device(buf: BCBuffer, device):
    N_max = buf.n_max
    n_per_encoder = buf.n_per_encoder.to(device)
    arange = torch.arange(N_max, device=device).unsqueeze(0)
    node_mask_per_encoder = arange < (n_per_encoder.unsqueeze(1) + 2)
    return {
        "feats": buf.feats.to(device), "node_mask_per_encoder": node_mask_per_encoder,
        "encoder_idx": buf.encoder_idx.to(device),
        "c_t": buf.c_t.to(device), "U_mask": buf.U_mask.to(device),
        "scalars": buf.scalars.to(device), "drone_D_slot": buf.drone_D_slot.to(device),
        "m_rej": buf.m_rej.to(device), "m_zd": buf.m_zd.to(device),
        "m_zc": buf.m_zc.to(device), "m_k_ext": buf.m_k_ext.to(device),
        "m_j": buf.m_j.to(device),
        "a_rej": buf.a_rej.to(device), "a_zd": buf.a_zd.to(device),
        "a_zc": buf.a_zc.to(device), "a_k": buf.a_k_slot.to(device),
        "a_j": buf.a_j.to(device),
        "no_pin": torch.zeros(len(buf), dtype=torch.bool, device=device),
        "R": len(buf),
    }

def _forward_nll(policy, d, idx):
    eidx = d["encoder_idx"][idx]
    nm = d["node_mask_per_encoder"][eidx]
    h = policy.encoder(d["feats"][eidx], node_mask=nm)
    out = policy.evaluate_batch(
        h_nodes=h, c_t=d["c_t"][idx], scalars=d["scalars"][idx], U_mask=d["U_mask"][idx],
        drone_D_slot=d["drone_D_slot"][idx],
        mask_rejoin=d["m_rej"][idx], mask_zd=d["m_zd"][idx], mask_zc=d["m_zc"][idx],
        mask_k_ext=d["m_k_ext"][idx], mask_j=d["m_j"][idx],
        a_rejoin=d["a_rej"][idx], a_zd=d["a_zd"][idx], a_zc=d["a_zc"][idx],
        a_k_slot=d["a_k"][idx], a_j=d["a_j"][idx],
        j_forced=d["no_pin"][idx], k_forced=d["no_pin"][idx], node_mask=nm,
    )
    return -out["log_prob"].mean(), out["entropy"].mean()

@torch.no_grad()
def _val_nll(policy, d, batch_size):
    policy.eval()
    R = d["R"]
    tot = 0.0; nb = 0
    for start in range(0, R, batch_size):
        idx = torch.arange(start, min(start + batch_size, R), device=d["c_t"].device)
        nll, _ = _forward_nll(policy, d, idx)
        tot += float(nll); nb += 1
    return tot / max(nb, 1)

def _cosine_lr(step, total, base_lr, warmup, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    frac = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * min(frac, 1.0)))

def bc_train(buf: BCBuffer, policy, device, *, epochs, batch_size, lr, min_lr,
             warmup, weight_decay, entropy_coef, grad_clip, log_every,
             val_buf=None, patience=0, on_best=None):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    d = _to_device(buf, device)
    dv = _to_device(val_buf, device) if val_buf is not None else None
    R = d["R"]

    best_val = float("inf"); best_ep = -1; since_best = 0
    print(f"  {'epoch':>6} {'lr':>9} {'train_nll':>10} {'H':>7} "
          f"{'val_nll':>9} {'sec':>6}")
    for ep in range(epochs):
        cur_lr = _cosine_lr(ep, epochs, lr, warmup, min_lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        policy.train()
        t_ep = time.perf_counter()
        perm = torch.randperm(R, device=device)
        tot_loss = tot_ent = 0.0; nb = 0; gmax = 0.0
        for start in range(0, R, batch_size):
            idx = perm[start : start + batch_size]
            nll, ent = _forward_nll(policy, d, idx)
            loss = nll - entropy_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()
            tot_loss += float(nll.detach()); tot_ent += float(ent.detach())
            gmax = max(gmax, float(g)); nb += 1
        sec = time.perf_counter() - t_ep

        val = _val_nll(policy, dv, batch_size) if dv is not None else float("nan")
        if dv is not None and val < best_val - 1e-4:
            best_val = val; best_ep = ep; since_best = 0
            if on_best is not None:
                on_best(policy, ep, val)
        else:
            since_best += 1

        if ep % log_every == 0 or ep == epochs - 1 or (dv is not None and since_best == 0):
            tag = "  *best" if (dv is not None and since_best == 0) else ""
            print(f"  {ep:6d} {cur_lr:9.2e} {tot_loss/nb:10.4f} {tot_ent/nb:7.3f} "
                  f"{val:9.4f} {sec:6.1f}{tag}")

        if patience and dv is not None and since_best >= patience:
            print(f"  [early stop] no val improvement for {patience} epochs "
                  f"(best val_nll={best_val:.4f} @ ep{best_ep})")
            break
    return policy, best_val, best_ep

@torch.no_grad()
def greedy_objective(policy, inst) -> float | None:
    env = TTPDEnv(a280_path=A280, n=inst.n, scale_capacity=False)
    obs, _ = env.reset(options={"instance": inst})
    ctx = policy.encode(inst)
    g = 0.0
    for _ in range(2 * (inst.n + 2) + 2):
        sample = policy.act(obs, env, deterministic=True, ctx=ctx)
        obs, r, term, trunc, _ = env.step(sample.action)
        g += float(r)
        if term:
            return g
        if trunc:
            return None
    return None

def evaluate_clone(records, policy, device, max_eval=200):
    policy.eval()
    rows = []
    for rec in records[:max_eval]:
        inst = instance_from_dict(rec["instance"])
        gobj = greedy_objective(policy, inst)
        sa_obj = rec["objective"]
        if gobj is None:
            rows.append((rec["id"], rec["n"], sa_obj, None, None))
        else:
            gap = 100.0 * (sa_obj - gobj) / abs(sa_obj) if sa_obj != 0 else 0.0
            rows.append((rec["id"], rec["n"], sa_obj, gobj, gap))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, required=True, help="JSONL from gen_dataset.py")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512, help="steps per minibatch")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5, help="warmup epochs")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-aug", type=int, default=8, help="dihedral augmentations per instance")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out instance fraction")
    ap.add_argument("--patience", type=int, default=25, help="early-stop epochs (0=off)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, required=True, help="output checkpoint .pt")
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start policy weights from a prior BC checkpoint "
                         "(the size-progressive chain: n10 -> n15 -> n20 -> ...)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="greedy-decode eval after training")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    records = [json.loads(l) for l in open(args.data)]
    print(f"[bc] {len(records)} expert records from {args.data}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(records))
    n_val = int(round(args.val_frac * len(records))) if args.val_frac > 0 else 0
    val_ids = set(perm[:n_val].tolist())
    train_recs = [r for i, r in enumerate(records) if i not in val_ids]
    val_recs = [r for i, r in enumerate(records) if i in val_ids]
    by_size = {}
    for r in records:
        by_size.setdefault(r["n"], 0)
        by_size[r["n"]] += 1
    print(f"[bc] sizes: {dict(sorted(by_size.items()))}  | "
          f"train={len(train_recs)} val={len(val_recs)} instances")

    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    if args.init_from:
        payload = torch.load(ensure_local(args.init_from), map_location=device, weights_only=False)
        policy.load_state_dict(payload["policy"])
        print(f"[bc] warm-started from {args.init_from} "
              f"(size-progressive chain)")

    t0 = time.perf_counter()
    buf = build_buffer(train_recs, policy, device, n_aug=args.n_aug)
    val_buf = build_buffer(val_recs, policy, device, n_aug=1) if val_recs else None
    print(f"[bc] train buffer: {len(buf)} steps / {int(buf.encoder_idx.max())+1} "
          f"encoders (n_aug={args.n_aug}), N_max={buf.n_max}"
          + (f"  | val: {len(val_buf)} steps" if val_buf else "")
          + f"  ({time.perf_counter()-t0:.1f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def save(pol, tag=""):
        torch.save({"policy": pol.state_dict(),
                    "encoder_cfg": vars(enc_cfg), "decoder_cfg": vars(dec_cfg),
                    "source": "bc", "data": os.path.abspath(args.data)}, args.out)

    best_state = {"val": float("inf")}
    def on_best(pol, ep, val):
        best_state["val"] = val
        save(pol)

    t_train = time.perf_counter()
    policy, best_val, best_ep = bc_train(
        buf, policy, device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, min_lr=args.min_lr, warmup=args.warmup,
        weight_decay=args.weight_decay, entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip, log_every=args.log_every,
        val_buf=val_buf, patience=args.patience, on_best=on_best)
    train_secs = time.perf_counter() - t_train

    if val_buf is None:
        save(policy)
        print(f"[bc] saved final -> {args.out}  ({train_secs:.1f}s)")
    else:
        # restore the best-val checkpoint saved by on_best
        policy.load_state_dict(torch.load(ensure_local(args.out), map_location=device,
                                          weights_only=False)["policy"])
        print(f"[bc] best val_nll={best_val:.4f} @ ep{best_ep}; "
              f"saved best -> {args.out}  ({train_secs:.1f}s total, "
              f"{train_secs/max(best_ep+1,1):.2f}s/epoch)")

    if args.eval:
        print("\n[bc] greedy-decode eval on HELD-OUT val instances (clone vs SA):")
        eval_recs = val_recs if val_recs else records
        rows = evaluate_clone(eval_recs, policy, device)
        per_size: dict[int, list] = {}
        n_fail = 0
        for _rid, n, _sa, g_o, gap in rows:
            if gap is None:
                n_fail += 1
                continue
            per_size.setdefault(n, []).append(gap)
        print(f"  {'n':>5} {'instances':>10} {'mean_gap%':>10} {'max_gap%':>9} "
              f"{'<=1%':>6} {'decode_fail':>12}")
        all_gaps = []
        for n in sorted(per_size):
            gs = per_size[n]
            all_gaps.extend(gs)
            frac_good = 100.0 * sum(g <= 1.0 for g in gs) / len(gs)
            print(f"  {n:>5} {len(gs):>10} {np.mean(gs):>10.2f} {max(gs):>9.2f} "
                  f"{frac_good:>5.0f}% {'':>12}")
        if all_gaps:
            print(f"  {'ALL':>5} {len(all_gaps):>10} {np.mean(all_gaps):>10.2f} "
                  f"{max(all_gaps):>9.2f} "
                  f"{100.0*sum(g<=1.0 for g in all_gaps)/len(all_gaps):>5.0f}% "
                  f"{n_fail:>12}")

if __name__ == "__main__":
    main()
