import torch

from env.masking import NO_LAUNCH
from env.simulator import TTPDEnv
from policy.attention_policy import AttentionPolicy, _sample_or_argmax
from policy.decoder import _gather_node, _masked_log_softmax


def _rollout_with_grad(policy: AttentionPolicy, env: TTPDEnv, h_nodes: torch.Tensor):
    device = policy.device
    dummy = h_nodes.shape[1]                      # n_nodes index == dummy slot
    h_with_dummy = policy.decoder.build_h_with_dummy(h_nodes)
    obs, _ = env.reset(options={"instance": env.inst})
    total_reward = 0.0
    logp_sum = torch.zeros((), device=device)
    while True:
        c_t = torch.tensor([obs["c_t"]], dtype=torch.long, device=device)
        U_mask = torch.as_tensor(obs["U_mask"], dtype=torch.bool, device=device).unsqueeze(0)
        scalars = policy._scalars_from_obs(obs, device)
        ctx_vec = policy.decoder.build_context(h_nodes, c_t, U_mask, scalars)
        h_c = _gather_node(h_nodes, c_t)
        slot = int(obs["drone_D"]) if bool(obs["in_flight"]) else dummy
        h_D = _gather_node(h_with_dummy, torch.tensor([slot], dtype=torch.long, device=device))
        masks = env.current_masks()

        m_rej = torch.as_tensor(masks["rejoin"], dtype=torch.bool, device=device).unsqueeze(0)
        rej_logp = _masked_log_softmax(policy.decoder.head_rejoin_forward(ctx_vec, h_D, m_rej), m_rej)
        rej = _sample_or_argmax(rej_logp[0], False).item()

        if rej == 1:
            m_zd = torch.as_tensor(env.mask_z_drone_now(), dtype=torch.bool, device=device).unsqueeze(0)
        else:
            m_zd = torch.tensor([[True, False]], dtype=torch.bool, device=device)
        zd_logp = _masked_log_softmax(policy.decoder.head_z_drone_forward(ctx_vec, h_D, m_zd), m_zd)
        zd = _sample_or_argmax(zd_logp[0], False).item()

        m_zc = torch.as_tensor(masks["z_curr"], dtype=torch.bool, device=device).unsqueeze(0)
        zc_logp = _masked_log_softmax(policy.decoder.head_z_curr_forward(ctx_vec, h_c, m_zc), m_zc)
        zc = _sample_or_argmax(zc_logp[0], False).item()

        li = env.masks_for_launch(rejoined=(rej == 1))
        m_k = torch.as_tensor(li["k_ext"], dtype=torch.bool, device=device).unsqueeze(0)
        k_logp = _masked_log_softmax(
            policy.decoder.head_k_forward(h_with_dummy, ctx_vec, m_k), m_k)
        k_slot = _sample_or_argmax(k_logp[0], False).item()
        k_action = NO_LAUNCH if k_slot == dummy else k_slot

        m_j_np = env.mask_j_after_launch(k_action, rejoined=(rej == 1))
        m_j = torch.as_tensor(m_j_np, dtype=torch.bool, device=device).unsqueeze(0)
        k_slot_t = torch.tensor([k_slot], dtype=torch.long, device=device)
        j_logp = _masked_log_softmax(
            policy.decoder.head_j_forward(h_nodes, h_with_dummy, ctx_vec, k_slot_t, m_j), m_j)
        j = _sample_or_argmax(j_logp[0], False).item()

        logp_sum = (logp_sum + rej_logp[0, rej] + zd_logp[0, zd] + zc_logp[0, zc]
                    + k_logp[0, k_slot] + j_logp[0, j])
        action = {"j": int(j), "z_curr": int(zc), "rejoin": int(rej),
                  "z_drone": int(zd), "k": int(k_action)}
        obs, r, term, trunc, _ = env.step(action)
        total_reward += float(r)
        if term or trunc:
            return total_reward, logp_sum


def eas_emb(policy: AttentionPolicy, env: TTPDEnv, *, iters: int, k: int,
            lr: float = 1e-2, return_embeddings: bool = False):
    device = policy.device
    # Encode once to get the starting embeddings, then optimize a copy of them.
    with torch.no_grad():
        ctx = policy.encode(env.inst, aug_idx=0)
    h = ctx.h_nodes.detach().clone().requires_grad_(True)

    # Freeze the policy; only h is optimized.
    saved_req = [(p, p.requires_grad) for p in policy.parameters()]
    for p in policy.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([h], lr=lr)

    best_obj = -float("inf")
    try:
        for _ in range(iters):
            rewards = torch.empty(k, device=device)
            logps = []
            for b in range(k):
                rew, logp = _rollout_with_grad(policy, env, h)
                rewards[b] = rew
                logps.append(logp)
                if rew > best_obj:
                    best_obj = rew
            logp_stack = torch.stack(logps)
            # batch-mean baseline (same POMO baseline used in training)
            baseline = rewards.mean()
            adv = rewards - baseline
            loss = -(adv.detach() * logp_stack).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    finally:
        for p, req in saved_req:
            p.requires_grad_(req)

    if return_embeddings:
        return best_obj, h.detach()
    return best_obj
