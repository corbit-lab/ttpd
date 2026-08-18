#a
# Beam-search decoding for the TTP-D policy
# Expands the top-W cumulative-log-prob action prefixes at each step
from dataclasses import dataclass

import numpy as np
import torch

from env.masking import NO_LAUNCH
from policy.attention_policy import candidate_features
from policy.decoder import _gather_node, _masked_log_softmax


@dataclass
class _Item:
    env: object
    obs: dict
    logp: float
    ret: float
    acts: tuple = ()   # action history (kept only when tracing)


def _clone_env(env):
    e = object.__new__(type(env))
    e.__dict__ = dict(env.__dict__)
    e.V_t = env.V_t.copy()
    return e


def _head_logps(policy, ctx, obs, env, device, force_collect=False, k0=None):
    
    h_nodes = ctx.h_nodes
    dummy = ctx.dummy_slot

    c_t = torch.tensor([obs["c_t"]], dtype=torch.long, device=device)
    U_mask = torch.as_tensor(obs["U_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    scalars = policy._scalars_from_obs(obs, device)
    ctx_vec = policy.decoder.build_context(h_nodes, c_t, U_mask, scalars)
    h_c = _gather_node(h_nodes, c_t)
    slot = int(obs["drone_D"]) if bool(obs["in_flight"]) else dummy
    h_D = _gather_node(ctx.h_with_dummy, torch.tensor([slot], dtype=torch.long, device=device))
    fj_np, fk_np = candidate_features(obs, env.inst)
    feat_j = torch.as_tensor(fj_np, device=device).unsqueeze(0)
    feat_k = torch.as_tensor(fk_np, device=device).unsqueeze(0)

    masks = env.current_masks()
    m_rej = np.asarray(masks["rejoin"], dtype=bool)
    m_zc = np.asarray(masks["z_curr"], dtype=bool)
    if force_collect and m_zc[1]:
        m_zc = np.array([False, True])

    rej_t = torch.as_tensor(m_rej, dtype=torch.bool, device=device).unsqueeze(0)
    rej_lp = _masked_log_softmax(
        policy.decoder.head_rejoin_forward(ctx_vec, h_D, rej_t), rej_t)[0].cpu().numpy()

    zc_t = torch.as_tensor(m_zc, dtype=torch.bool, device=device).unsqueeze(0)
    zc_lp = _masked_log_softmax(
        policy.decoder.head_z_curr_forward(ctx_vec, h_c, zc_t), zc_t)[0].cpu().numpy()
    zc_feasible = {}
    for rj in (0, 1):
        for zd_v in (0, 1):
            mzc = np.asarray(env.mask_z_curr_after(rj, zd_v), dtype=bool)
            if force_collect and mzc[1]:
                mzc = np.array([False, True])
            zc_feasible[(rj, zd_v)] = mzc

    # z_drone and launch masks depend on the rejoin choice
    zd_lp, mk, k_lp = {}, {}, {}
    zd_raw = policy.decoder.head_z_drone(torch.cat([ctx_vec, h_D], dim=-1))
    for rej in (0, 1):
        if not m_rej[rej]:
            continue
        m_zd = (np.asarray(env.mask_z_drone_now(), dtype=bool) if rej == 1
                else np.array([True, False]))
        if force_collect and m_zd[1]:
            m_zd = np.array([False, True])
        zd_t = torch.as_tensor(m_zd, dtype=torch.bool, device=device).unsqueeze(0)
        zd_lp[rej] = (_masked_log_softmax(zd_raw.masked_fill(~zd_t, -1e30), zd_t)[0]
                      .cpu().numpy(), m_zd)
        m_k = np.asarray(env.masks_for_launch(rejoined=(rej == 1))["k_ext"], dtype=bool)
        if k0 is not None and m_k[k0]:
            m_k = np.zeros_like(m_k)
            m_k[k0] = True
        kt = torch.as_tensor(m_k, dtype=torch.bool, device=device).unsqueeze(0)
        # re-score the pointer under the actual mask (mask shapes the glimpse too)
        k_scores = policy.decoder.head_k(ctx_vec, ctx.h_with_dummy, kt, feats=feat_k)
        k_lp[rej] = _masked_log_softmax(k_scores, kt)[0].cpu().numpy()
        mk[rej] = m_k

    # j depends on (rejoin, k): batch the j-head over all feasible (rej, k) pairs
    pairs = []
    for rej in k_lp:
        for k_slot in np.flatnonzero(mk[rej]):
            k_action = NO_LAUNCH if int(k_slot) == dummy else int(k_slot)
            m_j = env.mask_j_after_launch(k_action, rejoined=(rej == 1))
            pairs.append((rej, int(k_slot), k_action, m_j))
    if pairs:
        P = len(pairs)
        ctx_rep = ctx_vec.expand(P, -1)
        h_nodes_rep = h_nodes.expand(P, -1, -1)
        h_wd_rep = ctx.h_with_dummy.expand(P, -1, -1)
        kslots = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        mj_t = torch.as_tensor(np.stack([p[3] for p in pairs]), dtype=torch.bool, device=device)
        j_scores = policy.decoder.head_j_forward(h_nodes_rep, h_wd_rep, ctx_rep, kslots, mj_t,
                                                 feat_j=feat_j.expand(P, -1, -1))
        j_lp_all = _masked_log_softmax(j_scores, mj_t).cpu().numpy()
    else:
        j_lp_all = np.zeros((0, 0))

    return {
        "m_rej": m_rej, "rej_lp": rej_lp,
        "m_zc": m_zc, "zc_lp": zc_lp, "zc_feasible": zc_feasible,
        "zd": zd_lp, "m_k": mk, "k_lp": k_lp,
        "pairs": pairs, "j_lp": j_lp_all,
        "dummy": dummy,
    }


@torch.no_grad()
def beam_search(policy, env, inst, *, beam_width: int = 256, n_aug: int = 8,
                ctx_override=None, force_collect: bool = False,
                stratify_k0: bool = False, return_actions: bool = False):
    device = policy.device
    best_ret = None
    best_acts = None
    max_depth = 2 * (inst.n + 2) + 2

    # first-step launch strata: every customer slot + the dummy (no-launch)
    if stratify_k0:
        probe = _clone_env(env)
        probe.reset(options={"instance": inst})
        m_k0 = np.asarray(probe.masks_for_launch(rejoined=False)["k_ext"], dtype=bool)
        strata = [int(s) for s in np.flatnonzero(m_k0)]
    else:
        strata = [None]

    for aug in range(1 if ctx_override is not None else max(1, n_aug)):
      ctx = ctx_override if ctx_override is not None else policy.encode(inst, aug_idx=aug)
      for k0 in strata:
        e0 = _clone_env(env)
        obs0, _ = e0.reset(options={"instance": inst})
        beams = [_Item(env=e0, obs=obs0, logp=0.0, ret=0.0)]

        for _depth in range(max_depth):
            if not beams:
                break
            candidates = []  # (total_logp, item_idx, action)
            for ii, it in enumerate(beams):
                H = _head_logps(policy, ctx, it.obs, it.env, device,
                                force_collect=force_collect,
                                k0=k0 if _depth == 0 else None)
                for pi, (rej, k_slot, k_action, m_j) in enumerate(H["pairs"]):
                    base = H["rej_lp"][rej] + H["k_lp"][rej][k_slot]
                    zd_lp_arr, m_zd = H["zd"][rej]
                    for zd in np.flatnonzero(m_zd):
                        # z_curr feasibility depends on the post-rejoin load (rej, zd)
                        m_zc_now = H["zc_feasible"][(rej, int(zd))]
                        for zc in np.flatnonzero(m_zc_now):
                            partial = base + zd_lp_arr[zd] + H["zc_lp"][zc]
                            for j in np.flatnonzero(m_j):
                                candidates.append((
                                    it.logp + partial + H["j_lp"][pi][j], ii,
                                    {"j": int(j), "z_curr": int(zc), "rejoin": int(rej),
                                     "z_drone": int(zd), "k": int(k_action)},
                                ))
            if not candidates:
                break
            candidates.sort(key=lambda c: c[0], reverse=True)
            next_beams = []
            for total_lp, ii, action in candidates[:beam_width]:
                child = _clone_env(beams[ii].env)
                obs, r, term, trunc, _ = child.step(action)
                ret = beams[ii].ret + float(r)
                acts = beams[ii].acts + (action,) if return_actions else ()
                if term or trunc:
                    if best_ret is None or ret > best_ret:
                        best_ret = ret
                        best_acts = acts
                else:
                    next_beams.append(_Item(env=child, obs=obs, logp=float(total_lp),
                                            ret=ret, acts=acts))
            beams = next_beams

    if return_actions:
        return best_ret, best_acts
    return best_ret
