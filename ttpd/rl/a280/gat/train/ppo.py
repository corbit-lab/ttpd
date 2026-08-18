#a
from dataclasses import dataclass
import torch
from policy.attention_policy import AttentionPolicy
from train.config import PPOConfig
from train.rollout import RolloutBuffer


@dataclass
class PPOStats:
    loss_total: float = 0.0
    loss_policy: float = 0.0
    loss_entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    grad_norm: float = 0.0          
    n_updates: int = 0
    ent_rejoin: float = 0.0
    ent_zd: float = 0.0
    ent_j: float = 0.0
    ent_zc: float = 0.0
    ent_k: float = 0.0
    grad_norm_preclip_max: float = 0.0
    ratio_max: float = 0.0

    def reduce(self) -> "PPOStats":
        if self.n_updates == 0:
            return self
        n = self.n_updates
        return PPOStats(
            loss_total=self.loss_total / n, loss_policy=self.loss_policy / n,
            loss_entropy=self.loss_entropy / n,
            approx_kl=self.approx_kl / n, clip_fraction=self.clip_fraction / n,
            grad_norm=self.grad_norm / n, n_updates=n,
            ent_rejoin=self.ent_rejoin / n, ent_zd=self.ent_zd / n, ent_j=self.ent_j / n,
            ent_zc=self.ent_zc / n, ent_k=self.ent_k / n,
            grad_norm_preclip_max=self.grad_norm_preclip_max, ratio_max=self.ratio_max,
        )


class PPOTrainer:
    """PPO-clip over POMO group-baseline advantages. No critic: the baseline is the
    mean return of the instance's multi-start group (computed in collect_rollouts)."""

    def __init__(self, policy: AttentionPolicy, cfg: PPOConfig):
        self.policy = policy
        self.cfg = cfg
        self._entropy_coef = cfg.entropy_coef
        self._target_kl = cfg.target_kl
        self.optimiser = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg.curriculum[0].lr,
            weight_decay=cfg.weight_decay,
        )

    def set_lr(self, lr: float) -> None:
        for pg in self.optimiser.param_groups:
            pg["lr"] = lr

    def set_entropy_coef(self, coef: float) -> None:
        self._entropy_coef = float(coef)

    @staticmethod
    def _normalise(adv: torch.Tensor) -> torch.Tensor:
        return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    def update(self, buf: RolloutBuffer) -> PPOStats:
        cfg = self.cfg
        device = next(self.policy.parameters()).device
        R = len(buf)

        adv = buf.advantage.clone()
        if cfg.normalise_advantage and R > 1:
            adv = self._normalise(adv)

        feats_enc = buf.feats_per_encoder.to(device)
        n_per_encoder = buf.n_per_encoder.to(device)

        # per-encoder node_mask: True for real slots 0..n+1, False for padding.
        N_max = int(buf.n_max)
        arange = torch.arange(N_max, device=device).unsqueeze(0)
        node_mask_per_encoder = arange < (n_per_encoder.unsqueeze(1) + 2)
        c_t = buf.c_t.to(device)
        U_mask = buf.U_mask.to(device)
        scalars = buf.scalars.to(device)
        drone_D_slot = buf.drone_D_slot.to(device)
        mask_rejoin = buf.mask_rejoin.to(device)
        mask_zd = buf.mask_zd.to(device)
        mask_zc = buf.mask_zc.to(device)
        mask_k_ext = buf.mask_k_ext.to(device)
        mask_j = buf.mask_j.to(device)
        a_rejoin = buf.action_rejoin.to(device)
        a_zd = buf.action_zd.to(device)
        a_zc = buf.action_zc.to(device)
        a_ks = buf.action_k_slot.to(device)
        a_j = buf.action_j.to(device)
        j_forced = buf.j_forced.to(device)
        k_forced = buf.k_forced.to(device)
        logp_old = buf.log_prob_old.to(device)
        adv = adv.to(device)
        encoder_idx = buf.encoder_idx.to(device)

        stats = PPOStats()
        self.policy.train()
        stop = False
        for _ in range(cfg.n_update_epochs):
            if stop:
                break
            perm = torch.randperm(R, device=device)
            for start in range(0, R, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                mb_kl = self._update_minibatch(
                    feats_enc=feats_enc,
                    node_mask_per_encoder=node_mask_per_encoder, encoder_idx=encoder_idx,
                    idx=idx,
                    c_t=c_t, U_mask=U_mask, scalars=scalars, drone_D_slot=drone_D_slot,
                    mask_rejoin=mask_rejoin, mask_zd=mask_zd, mask_zc=mask_zc,
                    mask_k_ext=mask_k_ext, mask_j=mask_j,
                    a_rejoin=a_rejoin, a_zd=a_zd, a_zc=a_zc, a_ks=a_ks, a_j=a_j,
                    j_forced=j_forced, k_forced=k_forced,
                    logp_old=logp_old, adv=adv,
                    stats=stats,
                )
                if self._target_kl is not None and mb_kl > 1.5 * self._target_kl:
                    stop = True
                    break

        return stats.reduce()

    def _update_minibatch(self, *, feats_enc, node_mask_per_encoder,
                           encoder_idx, idx,
                           c_t, U_mask, scalars, drone_D_slot,
                           mask_rejoin, mask_zd, mask_zc, mask_k_ext, mask_j,
                           a_rejoin, a_zd, a_zc, a_ks, a_j, j_forced, k_forced,
                           logp_old, adv, stats):
        cfg = self.cfg

        eidx_mb = encoder_idx[idx]
        node_mask_mb = node_mask_per_encoder[eidx_mb]
        # re-encode under autograd so gradients flow through the encoder
        h_mb = self.policy.encoder(feats_enc[eidx_mb], node_mask=node_mask_mb)

        out = self.policy.evaluate_batch(
            h_nodes=h_mb,
            c_t=c_t[idx], scalars=scalars[idx], U_mask=U_mask[idx],
            drone_D_slot=drone_D_slot[idx],
            mask_rejoin=mask_rejoin[idx], mask_zd=mask_zd[idx], mask_zc=mask_zc[idx],
            mask_k_ext=mask_k_ext[idx], mask_j=mask_j[idx],
            a_rejoin=a_rejoin[idx], a_zd=a_zd[idx], a_zc=a_zc[idx],
            a_k_slot=a_ks[idx], a_j=a_j[idx],
            j_forced=j_forced[idx], k_forced=k_forced[idx],
            node_mask=node_mask_mb,
        )
        logp_new = out["log_prob"]
        hent = out["head_entropy"]

        ratio = (logp_new - logp_old[idx]).exp()
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - (logp_new - logp_old[idx])).mean()

        a = adv[idx]
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a
        policy_loss = -torch.min(ratio * a, clipped).mean()
        entropy_loss = -out["entropy"].mean()

        loss = policy_loss + self._entropy_coef * entropy_loss

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimiser.step()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

        stats.loss_total += float(loss.detach())
        stats.loss_policy += float(policy_loss.detach())
        stats.loss_entropy += float(entropy_loss.detach())
        stats.approx_kl += float(approx_kl)
        stats.clip_fraction += float(clip_frac)
        stats.grad_norm += float(g)
        stats.ent_rejoin += float(hent["rejoin"].mean().detach())
        stats.ent_zd += float(hent["z_drone"].mean().detach())
        stats.ent_j += float(hent["j"].mean().detach())
        stats.ent_zc += float(hent["z_curr"].mean().detach())
        stats.ent_k += float(hent["k"].mean().detach())
        stats.grad_norm_preclip_max = max(stats.grad_norm_preclip_max, float(g))
        stats.ratio_max = max(stats.ratio_max, float(ratio.max().detach()))
        stats.n_updates += 1
        return float(approx_kl.detach())
