#a
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
from env import TTPDEnv
from env.constants import tour_time_ub
from env.masking import NO_LAUNCH
from policy.attention_policy import AttentionPolicy, RolloutContext


@dataclass
class RolloutBuffer:
    feats_per_encoder: torch.Tensor
    n_per_encoder: torch.Tensor
    encoder_idx: torch.Tensor
    c_t: torch.Tensor
    U_mask: torch.Tensor
    scalars: torch.Tensor
    drone_D_slot: torch.Tensor
    action_rejoin: torch.Tensor
    action_zd: torch.Tensor
    action_zc: torch.Tensor
    action_k_slot: torch.Tensor
    action_j: torch.Tensor
    mask_rejoin: torch.Tensor
    mask_zd: torch.Tensor
    mask_zc: torch.Tensor
    mask_k_ext: torch.Tensor
    mask_j: torch.Tensor
    j_forced: torch.Tensor          # True where the truck move was pinned (POMO start)
    k_forced: torch.Tensor          # True where the launch was pinned (POMO launch strata)
    log_prob_old: torch.Tensor
    advantage: torch.Tensor
    raw_episode_returns: torch.Tensor
    raw_episode_truncated: torch.Tensor
    n_max: int = 0

    def __len__(self) -> int:
        return int(self.log_prob_old.shape[0])


def _enumerate_first_nodes(env, rng: np.random.Generator) -> list[int]:
    feasible_j = np.flatnonzero(env.current_masks()["j"])
    nodes = [int(j) for j in feasible_j]
    if nodes:
        nodes = [nodes[i] for i in rng.permutation(len(nodes))]
    return nodes


def _enumerate_first_launches(env, rng: np.random.Generator) -> list[int]:
    m_k = np.asarray(env.masks_for_launch(rejoined=False)["k_ext"], dtype=bool)
    slots = [int(s) for s in np.flatnonzero(m_k)]
    if slots:
        slots = [slots[i] for i in rng.permutation(len(slots))]
    return slots


def _sample_size(rng, n_lo, n_hi, n_sample_power, anchor_sizes, anchor_frac) -> int:
    anchors = [a for a in (anchor_sizes or ()) if n_lo <= a <= n_hi]
    if anchors and rng.random() < anchor_frac:
        return int(anchors[rng.integers(0, len(anchors))])
    if n_sample_power and n_hi > n_lo:
        sizes = np.arange(n_lo, n_hi + 1, dtype=np.float64)
        w = sizes ** float(n_sample_power)
        return int(rng.choice(sizes.astype(np.int64), p=w / w.sum()))
    return int(rng.integers(n_lo, n_hi + 1))


def collect_rollouts(
    policy: AttentionPolicy,
    env_factory: Callable[[int], TTPDEnv],
    n_instances: int,
    pomo_size: int,
    n_lo: int,
    n_hi: int,
    rng: np.random.Generator,
    device: torch.device,
    n_sample_power: float = 0.0,
    anchor_sizes: tuple = (),
    anchor_frac: float = 0.0,
) -> RolloutBuffer:
    policy.eval()

    feats_enc, n_enc = [], []
    rec_encoder_idx, rec_c_t = [], []
    rec_U_mask, rec_scalars, rec_drone_slot = [], [], []
    rec_a_rej, rec_a_zd, rec_a_zc, rec_a_k, rec_a_j = [], [], [], [], []
    rec_m_rej, rec_m_zd, rec_m_zc, rec_m_k, rec_m_j = [], [], [], [], []
    rec_j_forced, rec_k_forced, rec_logp = [], [], []
    ep_total_return: list[float] = []
    ep_truncated: list[bool] = []
    ep_ranges: list[list[int]] = []   # per-episode flat step indices (interleaved batching)
    group_ranges: list[tuple[int, int, float]] = []  # [ep_start, ep_end) per instance + scale
    next_encoder_id = 0

    for _ in range(n_instances):
        n_inst = _sample_size(rng, n_lo, n_hi, n_sample_power, anchor_sizes, anchor_frac)
        base_env = env_factory(n_inst)
        base_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        inst = base_env.inst
        # Stationary per-instance reward scale: the magnitude of the dominant
        # time-cost term. A function of the instance only, so it never collapses.
        inst_scale = max(float(inst.R) * tour_time_ub(inst.d_max, inst.n), 1.0)
        first_nodes = _enumerate_first_nodes(base_env, rng)
        first_launches = _enumerate_first_launches(base_env, rng)
        local_dummy = inst.n + 2
        group_ep_start = len(ep_total_return)

        # one encode per augmentation (8), shared by the episodes that use it
        n_aug = min(8, pomo_size)
        ctxs: list[RolloutContext] = [policy.encode(inst, aug_idx=a) for a in range(n_aug)]
        feats_by_aug = [policy.build_encoder_inputs(inst, policy.device, a).squeeze(0).detach().cpu()
                        for a in range(n_aug)]
        h_all = torch.cat([ctxs[p % n_aug].h_nodes for p in range(pomo_size)], dim=0)
        h_wd_all = torch.cat([ctxs[p % n_aug].h_with_dummy for p in range(pomo_size)], dim=0)

        # spawn the episode envs (shared parsed dataset; no file re-reads)
        envs = []
        obs_list = []
        for p in range(pomo_size):
            e = TTPDEnv(a280_data=base_env._a280, n=n_inst,
                        scale_capacity=base_env._scale_capacity)
            o, _ = e.reset(options={"instance": inst})
            envs.append(e)
            obs_list.append(o)

        # per-episode bookkeeping
        for p in range(pomo_size):
            feats_enc.append(feats_by_aug[p % n_aug])
            n_enc.append(inst.n)
        enc_id = [next_encoder_id + p for p in range(pomo_size)]
        next_encoder_id += pomo_size
        step_count = [0] * pomo_size
        ep_reward = [0.0] * pomo_size
        ep_trunc = [False] * pomo_size
        ep_steps: list[list[int]] = [[] for _ in range(pomo_size)]  # flat indices per episode

        active = list(range(pomo_size))
        while active:
            sub_envs = [envs[i] for i in active]
            sub_obs = [obs_list[i] for i in active]
            fj = [int(first_nodes[i % len(first_nodes)])
                  if (first_nodes and step_count[i] == 0) else None
                  for i in active]
            # launch strata: stride decoupled from the first-node cycle so
            # (first node, first launch) pairs vary across rollouts; if a
            # stratum is infeasible the launch head is left free
            fk = [int(first_launches[(i // max(len(first_nodes), 1) + i)
                                     % len(first_launches)])
                  if (first_launches and step_count[i] == 0) else None
                  for i in active]
            out = policy.act_batch(sub_obs, sub_envs, h_all[active], h_wd_all[active],
                                   forced_j=fj, forced_k=fk)

            still_active = []
            for bi, i in enumerate(active):
                obs = obs_list[i]
                action = out["actions"][bi]
                pin_j = fj[bi] is not None
                pin_k = bool(out["k_pinned"][bi])
                in_flight = bool(obs["in_flight"])
                drone_D = int(obs["drone_D"])
                k_slot = local_dummy if action["k"] == NO_LAUNCH else int(action["k"])
                d_slot = drone_D if in_flight else local_dummy
                hlp = out["head_lp"]
                logp = (hlp["rejoin"][bi] + hlp["z_drone"][bi] + hlp["z_curr"][bi]
                        + (0.0 if pin_k else hlp["k"][bi])
                        + (0.0 if pin_j else hlp["j"][bi]))

                next_obs, r, term, trunc, _ = envs[i].step(action)

                rec_encoder_idx.append(enc_id[i])
                rec_c_t.append(int(obs["c_t"]))
                rec_U_mask.append(obs["U_mask"].copy())
                rec_scalars.append(np.array(
                    [obs["W_norm"], obs["tau_norm"], obs["U_frac"], obs["R_norm"], obs["drone_norm"],
                     obs["drone_wait_norm"], obs["drone_elapsed_norm"],
                     obs["ED_norm"], obs["endurance_slack_norm"]],
                    dtype=np.float32,
                ))
                rec_drone_slot.append(int(d_slot))
                rec_a_rej.append(int(action["rejoin"]))
                rec_a_zd.append(int(action["z_drone"]))
                rec_a_zc.append(int(action["z_curr"]))
                rec_a_k.append(int(k_slot))
                rec_a_j.append(int(action["j"]))
                rec_m_rej.append(out["m_rej"][bi])
                rec_m_zd.append(out["m_zd"][bi])
                rec_m_zc.append(out["m_zc"][bi])
                rec_m_k.append(out["m_k"][bi])
                rec_m_j.append(out["m_j"][bi])
                rec_j_forced.append(pin_j)
                rec_k_forced.append(pin_k)
                rec_logp.append(float(logp))
                ep_steps[i].append(len(rec_logp) - 1)

                ep_reward[i] += r
                obs_list[i] = next_obs
                step_count[i] += 1
                if term or trunc:
                    ep_trunc[i] = bool(trunc)
                else:
                    still_active.append(i)
            active = still_active

        for p in range(pomo_size):
            ep_total_return.append(ep_reward[p])
            ep_truncated.append(ep_trunc[p])
            ep_ranges.append(ep_steps[p])
        group_ranges.append((group_ep_start, len(ep_total_return), inst_scale))

    # POMO shared baseline: episode advantage = (G - mean of its instance group)/scale
    R = len(rec_logp)
    returns_arr = np.asarray(ep_total_return, dtype=np.float64)
    adv = np.zeros(R, dtype=np.float64)
    for (g0, g1, scale) in group_ranges:
        baseline = returns_arr[g0:g1].mean()
        for e in range(g0, g1):
            idxs = np.asarray(ep_ranges[e], dtype=np.int64)
            adv[idxs] = (returns_arr[e] - baseline) / scale

    # pack to padded tensors
    E = len(feats_enc)
    N_max = max(f.shape[0] for f in feats_enc)
    N_slots_max = N_max + 1
    F_node = feats_enc[0].shape[1]

    feats_padded = torch.zeros(E, N_max, F_node)
    for e, f in enumerate(feats_enc):
        feats_padded[e, : f.shape[0]] = f

    U_mask_pad = torch.zeros(R, N_max, dtype=torch.bool)
    mj_pad = torch.zeros(R, N_max, dtype=torch.bool)
    mk_ext_pad = torch.zeros(R, N_slots_max, dtype=torch.bool)
    for r_i in range(R):
        n_rec = rec_U_mask[r_i].shape[0]                # inst.n + 2
        U_mask_pad[r_i, :n_rec] = torch.as_tensor(rec_U_mask[r_i])
        mj_pad[r_i, :n_rec] = torch.as_tensor(rec_m_j[r_i])
        mk_ext_pad[r_i, :n_rec] = torch.as_tensor(rec_m_k[r_i][:n_rec])
        mk_ext_pad[r_i, N_max] = bool(rec_m_k[r_i][n_rec])   # local dummy -> global dummy slot

    # remap local-dummy slot indices (== inst.n+2) to the global dummy slot N_max
    def _remap(slot_list):
        out = []
        for r_i, s in enumerate(slot_list):
            n_rec = rec_U_mask[r_i].shape[0]
            out.append(N_max if int(s) == n_rec else int(s))
        return out

    return RolloutBuffer(
        feats_per_encoder=feats_padded,
        n_per_encoder=torch.tensor(n_enc, dtype=torch.long),
        encoder_idx=torch.tensor(rec_encoder_idx, dtype=torch.long),
        c_t=torch.tensor(rec_c_t, dtype=torch.long),
        U_mask=U_mask_pad,
        scalars=torch.tensor(np.stack(rec_scalars), dtype=torch.float32),
        drone_D_slot=torch.tensor(_remap(rec_drone_slot), dtype=torch.long),
        action_rejoin=torch.tensor(rec_a_rej, dtype=torch.long),
        action_zd=torch.tensor(rec_a_zd, dtype=torch.long),
        action_zc=torch.tensor(rec_a_zc, dtype=torch.long),
        action_k_slot=torch.tensor(_remap(rec_a_k), dtype=torch.long),
        action_j=torch.tensor(rec_a_j, dtype=torch.long),
        mask_rejoin=torch.tensor(np.stack(rec_m_rej)),
        mask_zd=torch.tensor(np.stack(rec_m_zd)),
        mask_zc=torch.tensor(np.stack(rec_m_zc)),
        mask_k_ext=mk_ext_pad,
        mask_j=mj_pad,
        j_forced=torch.tensor(rec_j_forced, dtype=torch.bool),
        k_forced=torch.tensor(rec_k_forced, dtype=torch.bool),
        log_prob_old=torch.tensor(rec_logp, dtype=torch.float32),
        advantage=torch.tensor(adv, dtype=torch.float32),
        raw_episode_returns=torch.tensor(ep_total_return, dtype=torch.float32),
        raw_episode_truncated=torch.tensor(ep_truncated, dtype=torch.bool),
        n_max=N_max,
    )
