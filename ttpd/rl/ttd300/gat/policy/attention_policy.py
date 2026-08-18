#a
from dataclasses import dataclass
import numpy as np
import torch
from env.masking import NO_LAUNCH
from policy.base import ActionSample
from policy.decoder import (
    DecoderConfig,
    StructuredDecoder,
    _gather_node,
    _masked_entropy,
    _masked_log_softmax,
)
from policy.encoder import GATEncoder, EncoderConfig


@dataclass
class RolloutContext:
    h_nodes: torch.Tensor
    h_with_dummy: torch.Tensor
    n_nodes: int
    dummy_slot: int
    source_idx: int


def candidate_features(obs: dict, inst) -> tuple[np.ndarray, np.ndarray]:
    dist = inst.dist
    n_nodes = inst.n + 2
    sink = inst.sink
    c = int(obs["c_t"])
    d_ref = max(float(inst.d_max), 1e-9)
    p_max = max(float(inst.profits.max()), 1.0)
    W_left = max(float(inst.W) - float(obs["W_t"]), 1e-6)

    d_c = dist[c] / d_ref
    d_sink = dist[:, sink] / d_ref
    prof = inst.profits / p_max
    w_fit = np.clip(inst.weights / W_left, 0.0, 2.0)
    if bool(obs["in_flight"]) and int(obs["drone_D"]) >= 0:
        d_drone = dist[int(obs["drone_D"])] / d_ref
    else:
        d_drone = np.zeros(n_nodes, dtype=np.float64)
    feat_j = np.stack([d_c, d_sink, prof, w_fit, d_drone], axis=1).astype(np.float32)

    avail = np.asarray(obs["U_mask"], dtype=bool)
    r_idx = np.flatnonzero(avail).tolist() + [sink]
    M = dist[:, r_idx].copy()
    for col, r in enumerate(r_idx):
        M[r, col] = np.inf            
    min_rejoin = M.min(axis=1)
    ED_ref = float(inst.ED) if inst.ED is not None else 2.0 * d_ref
    sortie = np.clip((dist[c] + min_rejoin) / max(ED_ref, 1e-9), 0.0, 3.0)
    feat_k = np.zeros((n_nodes + 1, 5), dtype=np.float32)
    feat_k[:n_nodes] = np.stack([d_c, sortie, d_sink, prof, w_fit], axis=1)
    return feat_j, feat_k


class AttentionPolicy(torch.nn.Module):
    def __init__(
        self,
        encoder_cfg: EncoderConfig | None = None,
        decoder_cfg: DecoderConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.encoder = GATEncoder(encoder_cfg)
        self.decoder = StructuredDecoder(decoder_cfg)
        if self.encoder.cfg.d_model != self.decoder.cfg.d_model:
            raise ValueError("Encoder and decoder d_model must match.")
        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # dihedral coordinate augmentation (8 views) for POMO
    @staticmethod
    def _augment_coords(coords_norm: np.ndarray, aug_idx: int) -> np.ndarray:
        x = coords_norm[:, 0]
        y = coords_norm[:, 1]
        if aug_idx >= 4:
            y = 1.0 - y
        rot = aug_idx % 4
        if rot == 0:
            xn, yn = x, y
        elif rot == 1:
            xn, yn = y, 1.0 - x
        elif rot == 2:
            xn, yn = 1.0 - x, 1.0 - y
        else:
            xn, yn = 1.0 - y, x
        return np.stack([xn, yn], axis=1)

    @staticmethod
    def build_encoder_inputs(inst, device, aug_idx: int = 0) -> torch.Tensor:
        coord_min = inst.coords.min(axis=0)
        coord_span = inst.coords.max(axis=0) - coord_min
        coord_span = np.where(coord_span > 1e-9, coord_span, 1.0)
        coords_norm = (inst.coords - coord_min) / coord_span
        coords_norm = AttentionPolicy._augment_coords(coords_norm, aug_idx)
        p_max = max(float(inst.profits.max()), 1.0)
        feats = np.stack(
            [
                coords_norm[:, 0],
                coords_norm[:, 1],
                inst.profits / p_max,
                inst.weights / inst.W,
            ],
            axis=1,
        )
        return torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

    def encode(self, inst, aug_idx: int = 0) -> RolloutContext:
        feats_t = self.build_encoder_inputs(inst, self.device, aug_idx)
        h_nodes = self.encoder(feats_t)
        with torch.no_grad():
            h_with_dummy = self.decoder.build_h_with_dummy(h_nodes)
        N = inst.n + 2
        return RolloutContext(
            h_nodes=h_nodes, h_with_dummy=h_with_dummy,
            n_nodes=N, dummy_slot=N, source_idx=inst.source,
        )

    @staticmethod
    def _scalars_from_obs(obs: dict, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [obs["W_norm"], obs["tau_norm"], obs["U_frac"], obs["R_norm"], obs["drone_norm"],
             obs["drone_wait_norm"], obs["drone_elapsed_norm"],
             obs["ED_norm"], obs["endurance_slack_norm"]],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

    def act(self, obs, env, *, deterministic: bool = False, ctx: RolloutContext | None = None,
            temperature: float = 1.0) -> ActionSample:
        if ctx is None:
            ctx = self.encode(env.inst)
        with torch.no_grad():
            return self._decode(obs, env, ctx, deterministic=deterministic, forced=None,
                                temperature=temperature)

    # POMO multi-start: pin some heads (e.g. the first truck node) and sample the rest.
    def act_force(self, obs, env, force: dict, *, ctx: RolloutContext,
                  deterministic: bool = False) -> ActionSample:
        with torch.no_grad():
            return self._decode(obs, env, ctx, deterministic=deterministic, forced=force)

    def _drone_target_emb(self, obs, ctx):
        # embedding of the in-flight drone target (dummy embedding when idle)
        dummy = ctx.dummy_slot
        slot = int(obs["drone_D"]) if bool(obs["in_flight"]) else dummy
        return _gather_node(ctx.h_with_dummy, torch.tensor([slot], dtype=torch.long, device=self.device)), slot

    # decisions in order: rejoin -> z_drone -> z_curr -> launch k -> truck move j
    def _decode(self, obs, env, ctx, deterministic, forced, temperature: float = 1.0):
        device = self.device
        h_nodes = ctx.h_nodes
        dummy = ctx.dummy_slot

        c_t = torch.tensor([obs["c_t"]], dtype=torch.long, device=device)
        U_mask = torch.as_tensor(obs["U_mask"], dtype=torch.bool, device=device).unsqueeze(0)
        scalars = self._scalars_from_obs(obs, device)
        ctx_vec = self.decoder.build_context(h_nodes, c_t, U_mask, scalars)
        h_c = _gather_node(h_nodes, c_t)
        h_D, _ = self._drone_target_emb(obs, ctx)
        fj_np, fk_np = candidate_features(obs, env.inst)
        feat_j = torch.as_tensor(fj_np, device=device).unsqueeze(0)
        feat_k = torch.as_tensor(fk_np, device=device).unsqueeze(0)

        masks = env.current_masks()

        def pick(logp, key):
            if forced is not None and key in forced:
                return int(forced[key])
            return _sample_or_argmax(logp[0], deterministic, temperature).item()

        # 1. rejoin
        m_rej = torch.as_tensor(masks["rejoin"], dtype=torch.bool, device=device).unsqueeze(0)
        rej_logp = _masked_log_softmax(self.decoder.head_rejoin_forward(ctx_vec, h_D, m_rej), m_rej)
        rej = pick(rej_logp, "rejoin")
        rej_lp = rej_logp[0, rej]

        # 2. z_drone (collect the landed item); free only when landing
        if rej == 1:
            m_zd = torch.as_tensor(env.mask_z_drone_now(), dtype=torch.bool, device=device).unsqueeze(0)
        else:
            m_zd = torch.tensor([[True, False]], dtype=torch.bool, device=device)
        zd_logp = _masked_log_softmax(self.decoder.head_z_drone_forward(ctx_vec, h_D, m_zd), m_zd)
        zd = pick(zd_logp, "z_drone")
        zd_lp = zd_logp[0, zd]

        # 3. z_curr (truck pickup at c_t)
        m_zc = torch.as_tensor(masks["z_curr"], dtype=torch.bool, device=device).unsqueeze(0)
        zc_logp = _masked_log_softmax(self.decoder.head_z_curr_forward(ctx_vec, h_c, m_zc), m_zc)
        zc = pick(zc_logp, "z_curr")
        zc_lp = zc_logp[0, zc]

        # 4. launch_k (drone launch target or no-launch dummy). If the truck move j is
        # pinned (POMO start) but k is free, the pinned node cannot be a launch target.
        launch_info = env.masks_for_launch(rejoined=(rej == 1))
        m_k_np = np.asarray(launch_info["k_ext"], dtype=bool)
        if forced is not None and "j" in forced and "k" not in forced:
            m_k_np = m_k_np.copy()
            m_k_np[int(forced["j"])] = False
        m_k = torch.as_tensor(m_k_np, dtype=torch.bool, device=device).unsqueeze(0)
        k_logp = _masked_log_softmax(
            self.decoder.head_k_forward(ctx.h_with_dummy, ctx_vec, m_k, feat_k=feat_k), m_k
        )
        if forced is not None and "k" in forced:
            k_slot = dummy if int(forced["k"]) == NO_LAUNCH else int(forced["k"])
        else:
            k_slot = _sample_or_argmax(k_logp[0], deterministic, temperature).item()
        k_lp = k_logp[0, k_slot]
        k_action = NO_LAUNCH if k_slot == dummy else k_slot

        # 5. j (truck next node), masked by the launch commitment
        m_j_np = env.mask_j_after_launch(k_action, rejoined=(rej == 1))
        m_j = torch.as_tensor(m_j_np, dtype=torch.bool, device=device).unsqueeze(0)
        k_slot_t = torch.tensor([k_slot], dtype=torch.long, device=device)
        j_logp = _masked_log_softmax(
            self.decoder.head_j_forward(h_nodes, ctx.h_with_dummy, ctx_vec, k_slot_t, m_j,
                                        feat_j=feat_j), m_j
        )
        j = pick(j_logp, "j")
        j_lp = j_logp[0, j]

        joint_lp = rej_lp + zd_lp + zc_lp + k_lp + j_lp
        joint_ent = (_masked_entropy(rej_logp)[0] + _masked_entropy(zd_logp)[0]
                     + _masked_entropy(zc_logp)[0] + _masked_entropy(k_logp)[0]
                     + _masked_entropy(j_logp)[0])
        return ActionSample(
            action={"j": int(j), "z_curr": int(zc), "rejoin": int(rej),
                    "z_drone": int(zd), "k": int(k_action)},
            log_prob=float(joint_lp), entropy=float(joint_ent),
            head_log_probs={"rejoin": float(rej_lp), "z_drone": float(zd_lp),
                            "j": float(j_lp), "z_curr": float(zc_lp), "k": float(k_lp)},
        )

    def evaluate(self, obs, env, action) -> ActionSample:
        with torch.no_grad():
            ctx = self.encode(env.inst)
            return self._decode(obs, env, ctx, deterministic=True, forced=action)

    # Batched decode for rollout collection: all B episodes share the SAME instance
    # (so node counts match; no padding) but may use different augmentations
    # (h_nodes is per-episode). One NN forward per head per step for the whole
    # group instead of B sequential decodes; this is what keeps the GPU busy.
    @torch.no_grad()
    def act_batch(
        self,
        obs_list: list[dict],
        envs: list,
        h_nodes: torch.Tensor,        # (B, N, D)
        h_with_dummy: torch.Tensor,   # (B, N+1, D)
        forced_j: list[int | None],
        forced_k: list[int | None] | None = None,
    ) -> dict:
        device = self.device
        B, N, _ = h_nodes.shape
        dummy = N

        c_t = torch.tensor([o["c_t"] for o in obs_list], dtype=torch.long, device=device)
        U_mask = torch.as_tensor(np.stack([o["U_mask"] for o in obs_list]),
                                 dtype=torch.bool, device=device)
        scalars = torch.as_tensor(np.stack([
            [o["W_norm"], o["tau_norm"], o["U_frac"], o["R_norm"], o["drone_norm"],
             o["drone_wait_norm"], o["drone_elapsed_norm"],
             o["ED_norm"], o["endurance_slack_norm"]] for o in obs_list
        ]), dtype=torch.float32, device=device)
        ctx_vec = self.decoder.build_context(h_nodes, c_t, U_mask, scalars)
        h_c = _gather_node(h_nodes, c_t)
        d_slot = torch.tensor(
            [int(o["drone_D"]) if bool(o["in_flight"]) else dummy for o in obs_list],
            dtype=torch.long, device=device)
        h_D = _gather_node(h_with_dummy, d_slot)

        cand = [candidate_features(o, envs[i].inst) for i, o in enumerate(obs_list)]
        feat_j_np = np.stack([f[0] for f in cand])
        feat_k_np = np.stack([f[1] for f in cand])
        feat_j = torch.as_tensor(feat_j_np, device=device)
        feat_k = torch.as_tensor(feat_k_np, device=device)

        masks_now = [env.current_masks() for env in envs]

        # 1. rejoin
        m_rej_np = np.stack([m["rejoin"] for m in masks_now])
        m_rej = torch.as_tensor(m_rej_np, dtype=torch.bool, device=device)
        rej_logp = _masked_log_softmax(self.decoder.head_rejoin_forward(ctx_vec, h_D, m_rej), m_rej)
        rej = _sample_batch(rej_logp)

        # 2. z_drone; mask depends on each env's sampled rejoin
        m_zd_np = np.stack([
            envs[i].mask_z_drone_now() if rej[i] == 1 else np.array([True, False])
            for i in range(B)
        ])
        m_zd = torch.as_tensor(m_zd_np, dtype=torch.bool, device=device)
        zd_logp = _masked_log_softmax(self.decoder.head_z_drone_forward(ctx_vec, h_D, m_zd), m_zd)
        zd = _sample_batch(zd_logp)

        # 3. z_curr -- masked against the POST-rejoin truck load (step() validates it
        # after a drone landing transfers weight; matters only near capacity / large n)
        m_zc_np = np.stack([
            envs[i].mask_z_curr_after(int(rej[i]), int(zd[i])) for i in range(B)
        ])
        m_zc = torch.as_tensor(m_zc_np, dtype=torch.bool, device=device)
        zc_logp = _masked_log_softmax(self.decoder.head_z_curr_forward(ctx_vec, h_c, m_zc), m_zc)
        zc = _sample_batch(zc_logp)

        # 4. launch k (pinned truck moves are excluded as launch targets)
        m_k_np = np.stack([
            np.asarray(envs[i].masks_for_launch(rejoined=(rej[i] == 1))["k_ext"], dtype=bool)
            for i in range(B)
        ])
        for i in range(B):
            if forced_j[i] is not None:
                m_k_np[i, int(forced_j[i])] = False
        m_k = torch.as_tensor(m_k_np, dtype=torch.bool, device=device)
        k_logp = _masked_log_softmax(
            self.decoder.head_k_forward(h_with_dummy, ctx_vec, m_k, feat_k=feat_k), m_k)
        k_slot = _sample_batch(k_logp)
        # POMO launch strata: pin the first launch decision (same convention
        # as forced_j; the pinned head is excluded from the trained log-prob)
        k_pinned = [False] * B
        if forced_k is not None:
            for i in range(B):
                k0 = forced_k[i]
                if k0 is not None and bool(m_k_np[i, int(k0)]):
                    k_slot[i] = int(k0)
                    k_pinned[i] = True
        k_action = [NO_LAUNCH if int(k_slot[i]) == dummy else int(k_slot[i]) for i in range(B)]

        # 5. truck move j, conditioned on the launch commitment
        m_j_np = np.stack([
            envs[i].mask_j_after_launch(k_action[i], rejoined=(rej[i] == 1))
            for i in range(B)
        ])
        m_j = torch.as_tensor(m_j_np, dtype=torch.bool, device=device)
        k_slot_t = torch.as_tensor(k_slot, dtype=torch.long, device=device)
        j_logp = _masked_log_softmax(
            self.decoder.head_j_forward(h_nodes, h_with_dummy, ctx_vec, k_slot_t, m_j,
                                        feat_j=feat_j), m_j
        )
        j = _sample_batch(j_logp)
        for i in range(B):
            if forced_j[i] is not None:
                j[i] = int(forced_j[i])

        idx = torch.arange(B, device=device)
        rej_t = torch.as_tensor(rej, dtype=torch.long, device=device)
        zd_t = torch.as_tensor(zd, dtype=torch.long, device=device)
        zc_t = torch.as_tensor(zc, dtype=torch.long, device=device)
        j_t = torch.as_tensor(j, dtype=torch.long, device=device)
        head_lp = {
            "rejoin": rej_logp[idx, rej_t].cpu().numpy(),
            "z_drone": zd_logp[idx, zd_t].cpu().numpy(),
            "z_curr": zc_logp[idx, zc_t].cpu().numpy(),
            "k": k_logp[idx, k_slot_t].cpu().numpy(),
            "j": j_logp[idx, j_t].cpu().numpy(),
        }
        actions = [
            {"j": int(j[i]), "z_curr": int(zc[i]), "rejoin": int(rej[i]),
             "z_drone": int(zd[i]), "k": int(k_action[i])}
            for i in range(B)
        ]
        return {
            "actions": actions, "head_lp": head_lp,
            "m_rej": m_rej_np, "m_zd": m_zd_np, "m_zc": m_zc_np,
            "m_k": m_k_np, "m_j": m_j_np, "k_pinned": k_pinned,
            "feat_j": feat_j_np, "feat_k": feat_k_np,
        }

    def evaluate_batch(
        self,
        h_nodes, c_t, scalars, U_mask, drone_D_slot,
        mask_rejoin, mask_zd, mask_zc, mask_k_ext, mask_j,
        a_rejoin, a_zd, a_zc, a_k_slot, a_j,
        j_forced=None,
        k_forced=None,
        node_mask=None,
        feat_j=None,
        feat_k=None,
    ) -> dict:
        ctx_vec = self.decoder.build_context(h_nodes, c_t, U_mask, scalars, node_mask=node_mask)
        h_c = _gather_node(h_nodes, c_t)
        h_with_dummy = self.decoder.build_h_with_dummy(h_nodes)
        h_D = _gather_node(h_with_dummy, drone_D_slot)

        rej_logp = _masked_log_softmax(self.decoder.head_rejoin_forward(ctx_vec, h_D, mask_rejoin), mask_rejoin)
        zd_logp = _masked_log_softmax(self.decoder.head_z_drone_forward(ctx_vec, h_D, mask_zd), mask_zd)
        zc_logp = _masked_log_softmax(self.decoder.head_z_curr_forward(ctx_vec, h_c, mask_zc), mask_zc)
        k_logp = _masked_log_softmax(
            self.decoder.head_k_forward(h_with_dummy, ctx_vec, mask_k_ext, feat_k=feat_k),
            mask_k_ext
        )
        j_logp = _masked_log_softmax(
            self.decoder.head_j_forward(h_nodes, h_with_dummy, ctx_vec, a_k_slot, mask_j,
                                        feat_j=feat_j), mask_j
        )

        rej_lp = rej_logp.gather(1, a_rejoin.unsqueeze(-1)).squeeze(-1)
        zd_lp = zd_logp.gather(1, a_zd.unsqueeze(-1)).squeeze(-1)
        zc_lp = zc_logp.gather(1, a_zc.unsqueeze(-1)).squeeze(-1)
        k_lp = k_logp.gather(1, a_k_slot.unsqueeze(-1)).squeeze(-1)
        j_lp = j_logp.gather(1, a_j.unsqueeze(-1)).squeeze(-1)
        if j_forced is not None:
            j_lp = j_lp * (~j_forced).float()
        if k_forced is not None:
            k_lp = k_lp * (~k_forced).float()

        ent_rejoin = _masked_entropy(rej_logp); ent_zd = _masked_entropy(zd_logp)
        ent_zc = _masked_entropy(zc_logp); ent_k = _masked_entropy(k_logp)
        ent_j = _masked_entropy(j_logp)
        joint_log_prob = rej_lp + zd_lp + zc_lp + k_lp + j_lp
        joint_entropy = ent_rejoin + ent_zd + ent_zc + ent_k + ent_j
        return {"log_prob": joint_log_prob, "entropy": joint_entropy,
                "head_entropy": {"rejoin": ent_rejoin, "z_drone": ent_zd, "j": ent_j,
                                 "z_curr": ent_zc, "k": ent_k}}


def _sample_or_argmax(log_probs: torch.Tensor, deterministic: bool,
                      temperature: float = 1.0) -> torch.Tensor:
    if deterministic:
        return log_probs.argmax()
    # temper the masked log-probs: probs proportional to exp(logp / T).
    # T>1 flattens the distribution, T<1 sharpens it; masked entries stay
    # near -1e30 so the mask survives any T>0.
    lp = log_probs if temperature == 1.0 else log_probs / max(temperature, 1e-6)
    probs = lp.exp()
    s = probs.sum()
    if not torch.isfinite(s) or s <= 0:
        raise RuntimeError("Cannot sample from a fully-masked categorical; check masks.")
    probs = probs / s
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _sample_batch(log_probs: torch.Tensor) -> list[int]:
    # (B, A) masked log-probs -> one sampled action per row
    probs = log_probs.exp()
    s = probs.sum(dim=-1, keepdim=True)
    if not torch.isfinite(s).all() or (s <= 0).any():
        raise RuntimeError("Cannot sample from a fully-masked categorical; check masks.")
    return torch.multinomial(probs / s, num_samples=1).squeeze(-1).tolist()
