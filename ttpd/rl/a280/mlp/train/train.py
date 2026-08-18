#a
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from env import TTPDEnv
from policy.attention_policy import AttentionPolicy
from policy.decoder import DecoderConfig
from policy.encoder import EncoderConfig
from train.checkpoint import (
    TrainState, find_latest_checkpoint, load_checkpoint,
    promote_to_best, save_checkpoint,
)
from train.config import PPOConfig
from train.eval import EvalResult, build_eval_set, evaluate
from train.logger import JSONLLogger
from train.ppo import PPOStats, PPOTrainer
from train.rollout import collect_rollouts
from train.scheduler import WarmupCosine

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_HUB_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while (_HUB_ROOT != _os.path.dirname(_HUB_ROOT)
       and not _os.path.isdir(_os.path.join(_HUB_ROOT, "ttpd"))):
    _HUB_ROOT = _os.path.dirname(_HUB_ROOT)
if _HUB_ROOT not in _sys.path:
    _sys.path.insert(0, _HUB_ROOT)
from ttpd.hub import ensure_local  # noqa: E402


@dataclass
class RunConfig:
    run_dir: str = "runs/default"
    checkpoint_every: int = 25         # epochs
    eval_every: int = 5                # epochs (0 disables)
    eval_n_instances: int = 40
    eval_n_lo: int = 5
    eval_n_hi: int = 20
    eval_seed: int = 99
    resume: bool = True
    scale_capacity: bool = True        # W_n = W*n/280; must match the MILP baseline
    milp_dir: str | None = None        # dir with n15.csv/n20.csv/ttpd_results.csv for live refs
    milp_sizes: tuple = (5, 10, 15, 20)
    init_from: str | None = None


@dataclass
class EpochLog:
    stage_idx: int
    epoch: int          # within stage
    global_epoch: int   # across all stages
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    stats: PPOStats
    wall_seconds: float
    lr: float
    n_instances: int
    n_lo: int
    n_hi: int
    entropy_coef: float = 0.0
    frac_truncated: float = 0.0    # episodes that hit the step cap instead of terminating
    mean_ep_len: float = 0.0
    frac_drone_launch: float = 0.0 # fraction of steps that launch the drone
    frac_collect: float = 0.0      # fraction of steps that collect an item (z_curr or z_drone)
    eval: EvalResult | None = None


@dataclass
class TrainHistory:
    epochs: list[EpochLog] = field(default_factory=list)


def default_log(log: EpochLog) -> None:
    s = log.stats
    eval_str = ""
    if log.eval is not None:
        eval_str = (f"  EVAL R={log.eval.mean_return:+9.1f}"
                     f"±{log.eval.std_return:7.1f} sel={log.eval.select_score:+.3f}")
        if log.eval.milp_gap:
            # TRUE gap: same instance the MILP solved, best-known objective as reference.
            parts = []
            for n, d in sorted(log.eval.milp_gap.items()):
                if d["return"] is None:
                    parts.append(f"n{n}=FAIL")
                else:
                    star = "" if d["proven_optimal"] else "~"   # ~ = vs incumbent, not optimum
                    parts.append(f"n{n} gap{star}{d['gap_pct']:+.1f}%")
            eval_str += f" MILP[{' '.join(parts)}]"
    print(
        f"ep#{log.global_epoch:4d} st{log.stage_idx} "
        f"n=[{log.n_lo},{log.n_hi}] "
        f"R={log.mean_return:+9.1f}±{log.std_return:7.1f} "
        f"loss={s.loss_total:+.3f}(p={s.loss_policy:+.4f}) "
        f"kl={s.approx_kl:+.4f} clip={s.clip_fraction:.2f} "
        f"|g|={s.grad_norm:.1f}(max{s.grad_norm_preclip_max:.0f}) rmax={s.ratio_max:.1f} "
        f"Hk={s.ent_k:.2f} drone={log.frac_drone_launch:.2f} collect={log.frac_collect:.2f} trunc={log.frac_truncated:.2f} "
        f"lr={log.lr:.1e} {log.wall_seconds:.1f}s{eval_str}"
    )


def run(
    cfg: PPOConfig,
    a280_path: str,
    *,
    run_cfg: RunConfig | None = None,
    on_epoch: Callable[[EpochLog], None] | None = None,
    history: TrainHistory | None = None,
    max_epochs: int | None = None,
) -> TrainHistory:
    if run_cfg is None:
        run_cfg = RunConfig()
    if history is None:
        history = TrainHistory()

    run_dir = Path(run_cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    trainer = PPOTrainer(policy, cfg)
    # single global warmup-cosine over the whole run (no per-stage reset)
    total_epochs = sum(
        (s.epochs if max_epochs is None else min(s.epochs, max_epochs))
        for s in cfg.curriculum
    )
    scheduler = WarmupCosine(
        lr=cfg.curriculum[0].lr,
        warmup_steps=cfg.curriculum[0].warmup_steps,
        total_steps=total_epochs,
        min_lr=cfg.min_lr,
    )

    last_ckpt = find_latest_checkpoint(run_dir) if run_cfg.resume else None
    if last_ckpt is not None:
        train_state = load_checkpoint(
            last_ckpt,
            policy=policy, optimiser=trainer.optimiser,
            scheduler=scheduler, strict_cfg=cfg, map_location=device,
        )
        print(f"[resume] continuing from {last_ckpt}: "
              f"stage={train_state.stage_idx} epoch={train_state.epoch_in_stage} "
              f"global_epoch={train_state.global_epoch} "
              f"best_eval={train_state.best_eval_return:+.1f}")
        # Extension: the curriculum grew beyond the saved schedule (a finished run is
        # being continued). Re-warm the LR over the remaining epochs instead of
        # sitting at min_lr for the whole extension.
        if total_epochs > scheduler._total and train_state.global_epoch >= scheduler._total:
            remaining = max(1, total_epochs - train_state.global_epoch)
            scheduler.reset(lr=cfg.extend_peak_lr, warmup_steps=min(5, remaining),
                            total_steps=remaining, min_lr=cfg.min_lr)
            print(f"[resume] extension detected: re-warming LR to {cfg.extend_peak_lr:.1e} "
                  f"over the remaining {remaining} epochs")
    else:
        train_state = TrainState(
            stage_idx=0, epoch_in_stage=0,
            global_epoch=0, best_eval_return=float("-inf"),
        )
        if run_cfg.init_from:
            payload = torch.load(ensure_local(run_cfg.init_from), map_location=device, weights_only=False)
            policy.load_state_dict(payload["policy"])
            print(f"[init] policy weights warm-started from {run_cfg.init_from} "
                  f"(fresh optimiser/schedule)")

    logger = JSONLLogger(run_dir)
    if last_ckpt is None:
        logger.write_header(cfg=cfg, extra={"run_cfg": run_cfg})

    eval_set = build_eval_set(
        a280_path=a280_path,
        n_instances=run_cfg.eval_n_instances,
        n_lo=run_cfg.eval_n_lo, n_hi=run_cfg.eval_n_hi,
        seed=run_cfg.eval_seed,
        scale_capacity=run_cfg.scale_capacity,
    ) if run_cfg.eval_every > 0 else None

    def env_factory(n_inst: int) -> TTPDEnv:
        return TTPDEnv(a280_path=a280_path, n=n_inst,
                       scale_capacity=run_cfg.scale_capacity)

    # (total_epochs computed above; also used for the global entropy anneal)

    for stage_idx in range(train_state.stage_idx, len(cfg.curriculum)):
        stage = cfg.curriculum[stage_idx]
        # NO per-stage scheduler reset: the LR decays smoothly across stage boundaries.

        start_epoch = train_state.epoch_in_stage if stage_idx == train_state.stage_idx else 0
        end_epoch = stage.epochs if max_epochs is None else min(stage.epochs, max_epochs)

        for epoch in range(start_epoch, end_epoch):
            t0 = time.perf_counter()
            buf = collect_rollouts(
                policy=policy,
                env_factory=env_factory,
                n_instances=stage.instances_per_epoch,
                pomo_size=cfg.pomo_size,
                n_lo=stage.n_lo, n_hi=stage.n_hi,
                rng=rng,
                device=device,
                n_sample_power=cfg.n_sample_power,
                anchor_sizes=cfg.anchor_sizes,
                anchor_frac=cfg.anchor_frac,
            )
            scheduler.step()
            trainer.set_lr(scheduler.lr)

            frac = train_state.global_epoch / (total_epochs - 1) if total_epochs > 1 else 1.0
            cos = 0.5 * (1.0 + np.cos(np.pi * min(frac, 1.0)))
            ent_coef = cfg.entropy_coef_final + (cfg.entropy_coef - cfg.entropy_coef_final) * cos
            trainer.set_entropy_coef(ent_coef)
            stats = trainer.update(buf)
            dt = time.perf_counter() - t0

            ep_returns = buf.raw_episode_returns.numpy()
            n_eps = max(int(buf.raw_episode_returns.shape[0]), 1)
            frac_trunc = float(buf.raw_episode_truncated.float().mean()) if buf.raw_episode_truncated.numel() else 0.0
            mean_ep_len = float(len(buf)) / n_eps
            frac_drone = float((buf.action_k_slot != int(buf.n_max)).float().mean()) if len(buf) else 0.0
            frac_collect = float(((buf.action_zc == 1) | (buf.action_zd == 1)).float().mean()) if len(buf) else 0.0

            eval_result = None
            if eval_set is not None and (train_state.global_epoch + 1) % run_cfg.eval_every == 0:
                eval_result = evaluate(
                    policy, a280_path=a280_path, eval_set=eval_set,
                    deterministic=True, scale_capacity=run_cfg.scale_capacity,
                    milp_dir=run_cfg.milp_dir, milp_sizes=run_cfg.milp_sizes,
                )

            log = EpochLog(
                stage_idx=stage_idx, epoch=epoch,
                global_epoch=train_state.global_epoch,
                mean_return=float(ep_returns.mean()) if ep_returns.size else 0.0,
                std_return=float(ep_returns.std(ddof=1)) if ep_returns.size > 1 else 0.0,
                min_return=float(ep_returns.min()) if ep_returns.size else 0.0,
                max_return=float(ep_returns.max()) if ep_returns.size else 0.0,
                stats=stats, wall_seconds=dt, lr=scheduler.lr,
                n_instances=stage.instances_per_epoch,
                n_lo=stage.n_lo, n_hi=stage.n_hi,
                entropy_coef=float(ent_coef),
                frac_truncated=frac_trunc, mean_ep_len=mean_ep_len, frac_drone_launch=frac_drone,
                frac_collect=frac_collect,
                eval=eval_result,
            )
            history.epochs.append(log)
            logger.write_epoch(log)
            if on_epoch is not None:
                on_epoch(log)

            train_state = TrainState(
                stage_idx=stage_idx,
                epoch_in_stage=epoch + 1,
                global_epoch=train_state.global_epoch + 1,
                best_eval_return=train_state.best_eval_return,
                best_milp_gap=train_state.best_milp_gap,
            )

            should_save = (
                train_state.global_epoch % run_cfg.checkpoint_every == 0
                or epoch + 1 == end_epoch
            )
            if should_save:
                last_path = ckpt_dir / "last.pt"
                save_checkpoint(
                    last_path, policy=policy, optimiser=trainer.optimiser,
                    scheduler=scheduler, train_state=train_state, cfg=cfg,
                )
                if train_state.global_epoch % run_cfg.checkpoint_every == 0:
                    save_checkpoint(
                        ckpt_dir / f"epoch_{train_state.global_epoch}.pt",
                        policy=policy, optimiser=trainer.optimiser,
                        scheduler=scheduler, train_state=train_state, cfg=cfg,
                    )

            if eval_result is not None and eval_result.select_score > train_state.best_eval_return:
                train_state = TrainState(
                    stage_idx=train_state.stage_idx,
                    epoch_in_stage=train_state.epoch_in_stage,
                    global_epoch=train_state.global_epoch,
                    best_eval_return=eval_result.select_score,
                    best_milp_gap=train_state.best_milp_gap,
                )
                last_path = ckpt_dir / "last.pt"
                save_checkpoint(
                    last_path, policy=policy, optimiser=trainer.optimiser,
                    scheduler=scheduler, train_state=train_state, cfg=cfg,
                )
                promote_to_best(last_path, ckpt_dir / "best.pt")
                logger.write({"type": "best_updated",
                               "global_epoch": train_state.global_epoch,
                               "select_score": eval_result.select_score,
                               "eval_return": eval_result.mean_return,
                               "per_size": eval_result.per_size})

            if eval_result is not None and eval_result.milp_gap:
                gaps = [d["gap_pct"] for d in eval_result.milp_gap.values()
                        if d["return"] is not None and np.isfinite(d["gap_pct"])]
                mean_gap = float(np.mean(gaps)) if len(gaps) == len(eval_result.milp_gap) \
                    else float("inf")
                if mean_gap < train_state.best_milp_gap:
                    train_state = TrainState(
                        stage_idx=train_state.stage_idx,
                        epoch_in_stage=train_state.epoch_in_stage,
                        global_epoch=train_state.global_epoch,
                        best_eval_return=train_state.best_eval_return,
                        best_milp_gap=mean_gap,
                    )
                    last_path = ckpt_dir / "last.pt"
                    save_checkpoint(
                        last_path, policy=policy, optimiser=trainer.optimiser,
                        scheduler=scheduler, train_state=train_state, cfg=cfg,
                    )
                    promote_to_best(last_path, ckpt_dir / "best_milp.pt")
                    logger.write({"type": "best_milp_updated",
                                   "global_epoch": train_state.global_epoch,
                                   "mean_gap_pct": mean_gap,
                                   "milp_gap": eval_result.milp_gap})

    logger.close()
    return history
