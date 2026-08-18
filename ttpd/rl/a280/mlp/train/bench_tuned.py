import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from env import TTPDEnv
from env.instance import load_a280, sample_instance
from policy.attention_policy import AttentionPolicy
from policy.beam import beam_search
from policy.decoder import DecoderConfig
from policy.encoder import EncoderConfig
from train.checkpoint import (
    TrainState, find_latest_checkpoint, load_checkpoint,
    promote_to_best, save_checkpoint,
)
from train.config import CurriculumStage, PPOConfig
from train.logger import JSONLLogger
from train.ppo import PPOTrainer
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
class BenchTunedConfig:
    run_dir: str
    bench_files: list[str]            # the 5 bench_<n>_*.txt paths
    n: int                            # 50 or 100 (full instance size of each file)
    tag: str = "GAT"                  # "GAT" or "MLP" -- prefixes every log line
    init_from: str | None = None      # warm-start policy weights (n<=22 best.pt); None = from scratch
    instance_reps: int = 1            # times each bench instance appears per epoch
    total_epochs: int = 300           # whole run; every epoch sees ALL bench files
    lr: float = 6e-5                  # fine-tune peak LR (lower than from-scratch 1e-4)
    eval_every: int = 10              # epochs between cheap beam evals (0 disables)
    checkpoint_every: int = 25
    eval_beam_width: int = 16         # cheap periodic-eval beam (the promotion metric)
    eval_beam_n_aug: int = 2
    final_beam_width: int = 128       # full beam for the end-of-run eval
    final_beam_n_aug: int = 8
    beam_stratify: bool = False       # POMO launch-stratified beam (slow at large n)
    scale_capacity: bool = False
    resume: bool = True


def _full_instance(a280_data: dict, n: int, scale_capacity: bool):
    rng = np.random.default_rng(0)
    return sample_instance(a280_data, n=n, rng=rng, scale_capacity=scale_capacity)


def _log(bench_cfg: BenchTunedConfig, msg: str) -> None:
    print(f"[{bench_cfg.tag} n={bench_cfg.n}] {msg}", flush=True)


def _beam_eval(policy, bench_envs, bench_insts, bench_cfg: BenchTunedConfig,
               beam_width: int, beam_n_aug: int) -> dict:
    policy.eval()
    per_file = {}
    for path, env, inst in zip(bench_cfg.bench_files, bench_envs, bench_insts):
        G = beam_search(policy, env, inst, beam_width=beam_width,
                        n_aug=beam_n_aug, stratify_k0=bench_cfg.beam_stratify)
        per_file[Path(path).stem] = None if G is None else float(G)
    vals = [v for v in per_file.values() if v is not None]
    mean = float(np.mean(vals)) if vals else float("-inf")
    return {"mean": mean, "per_file": per_file}


def _save_and_promote(ckpt_dir, policy, trainer, scheduler, train_state, cfg):
    save_checkpoint(ckpt_dir / "last.pt", policy=policy,
                    optimiser=trainer.optimiser, scheduler=scheduler,
                    train_state=train_state, cfg=cfg)
    promote_to_best(ckpt_dir / "last.pt", ckpt_dir / "best.pt")


def run_bench_tuned(cfg: PPOConfig, bench_cfg: BenchTunedConfig) -> None:
    run_dir = Path(bench_cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    enc_cfg = EncoderConfig()
    dec_cfg = DecoderConfig(d_model=enc_cfg.d_model)
    policy = AttentionPolicy(enc_cfg, dec_cfg, device=device)
    trainer = PPOTrainer(policy, cfg)

    scheduler = WarmupCosine(
        lr=cfg.curriculum[0].lr,
        warmup_steps=cfg.curriculum[0].warmup_steps,
        total_steps=bench_cfg.total_epochs,
        min_lr=cfg.min_lr,
    )

    # one parsed dataset + one fixed full instance + one env per bench file
    bench_data = [load_a280(p) for p in bench_cfg.bench_files]
    bench_insts = [_full_instance(d, bench_cfg.n, bench_cfg.scale_capacity) for d in bench_data]
    bench_envs = [TTPDEnv(a280_data=d, n=bench_cfg.n, scale_capacity=bench_cfg.scale_capacity)
                  for d in bench_data]
    n_files = len(bench_cfg.bench_files)
    _log(bench_cfg, f"capacity check: scale_capacity={bench_cfg.scale_capacity}  "
                f"W(per file)={[round(float(i.W)) for i in bench_insts]}")

    last_ckpt = find_latest_checkpoint(run_dir) if bench_cfg.resume else None
    if last_ckpt is not None:
        train_state = load_checkpoint(
            last_ckpt, policy=policy, optimiser=trainer.optimiser,
            scheduler=scheduler, strict_cfg=cfg, map_location=device,
        )
        _log(bench_cfg, f"[resume] from {last_ckpt}: global_epoch={train_state.global_epoch} "
                    f"best={train_state.best_eval_return:+.1f}")
    else:
        train_state = TrainState(stage_idx=0, epoch_in_stage=0,
                                 global_epoch=0, best_eval_return=float("-inf"))
        if bench_cfg.init_from:
            payload = torch.load(ensure_local(bench_cfg.init_from), map_location=device, weights_only=False)
            policy.load_state_dict(payload["policy"])
            _log(bench_cfg, f"[init] warm-started policy from {bench_cfg.init_from} "
                        f"(fresh optimiser/schedule, lr={bench_cfg.lr:.1e})")
        else:
            _log(bench_cfg, "[init] FROM SCRATCH (no init_from) -- expect slow convergence at n>=50")

    logger = JSONLLogger(run_dir)
    if last_ckpt is None:
        logger.write_header(cfg=cfg, extra={"benchmark-tuned": bench_cfg.__dict__})

    n_inst_per_epoch = n_files * max(1, bench_cfg.instance_reps)
    _log(bench_cfg, f"start: {n_files} bench files x{bench_cfg.instance_reps}/epoch "
                f"({n_inst_per_epoch} POMO groups), total {bench_cfg.total_epochs} ep, "
                f"eval beam={bench_cfg.eval_beam_width}/{bench_cfg.eval_beam_n_aug}, device={cfg.device}")

    # round-robin env factory: collect_rollouts() pulls the next bench file on each
    # call, so n_instances = n_files * reps covers every instance reps times/epoch.
    rr = {"i": 0}

    def env_factory(_n):
        d = bench_data[rr["i"] % n_files]
        rr["i"] += 1
        return TTPDEnv(a280_data=d, n=bench_cfg.n, scale_capacity=bench_cfg.scale_capacity)

    if bench_cfg.init_from and train_state.global_epoch == 0 and bench_cfg.eval_every > 0:
        ev0 = _beam_eval(policy, bench_envs, bench_insts, bench_cfg,
                         bench_cfg.eval_beam_width, bench_cfg.eval_beam_n_aug)
        train_state = TrainState(stage_idx=0, epoch_in_stage=0, global_epoch=0,
                                 best_eval_return=ev0["mean"])
        _save_and_promote(ckpt_dir, policy, trainer, scheduler, train_state, cfg)
        _log(bench_cfg, f"[init-eval] warm-start beam mean={ev0['mean']:+.1f} -> seeded best.pt "
                    f"(stage best.pt can only improve from here)")
        logger.write({"type": "init_eval", "global_epoch": 0, **ev0})

    for gep in range(train_state.global_epoch, bench_cfg.total_epochs):
        rr["i"] = 0   # re-align to a file boundary each epoch (covers all files evenly)
        t0 = time.perf_counter()
        buf = collect_rollouts(
            policy=policy,
            env_factory=env_factory,
            n_instances=n_inst_per_epoch,   # ALL bench files (x reps), every epoch
            pomo_size=cfg.pomo_size,
            n_lo=bench_cfg.n, n_hi=bench_cfg.n,
            rng=rng,
            device=device,
            n_sample_power=0.0,
            anchor_sizes=(),
            anchor_frac=0.0,
        )
        scheduler.step()
        trainer.set_lr(scheduler.lr)

        frac = gep / (bench_cfg.total_epochs - 1) if bench_cfg.total_epochs > 1 else 1.0
        cos = 0.5 * (1.0 + np.cos(np.pi * min(frac, 1.0)))
        ent_coef = cfg.entropy_coef_final + (cfg.entropy_coef - cfg.entropy_coef_final) * cos
        trainer.set_entropy_coef(ent_coef)
        stats = trainer.update(buf)
        dt = time.perf_counter() - t0

        ep_returns = buf.raw_episode_returns.numpy()
        mean_R = float(ep_returns.mean()) if ep_returns.size else 0.0
        std_R = float(ep_returns.std(ddof=1)) if ep_returns.size > 1 else 0.0
        max_R = float(ep_returns.max()) if ep_returns.size else 0.0
        frac_trunc = float(buf.raw_episode_truncated.float().mean()) if buf.raw_episode_truncated.numel() else 0.0
        frac_drone = float((buf.action_k_slot != int(buf.n_max)).float().mean()) if len(buf) else 0.0
        frac_collect = float(((buf.action_zc == 1) | (buf.action_zd == 1)).float().mean()) if len(buf) else 0.0

        train_state = TrainState(
            stage_idx=0, epoch_in_stage=0,
            global_epoch=gep + 1, best_eval_return=train_state.best_eval_return,
        )

        # ---- training-metrics-only log line (no gap) ----
        _log(bench_cfg,
             f"ep#{gep:4d} files=all{n_files}x{bench_cfg.instance_reps} "
             f"R={mean_R:+9.1f}±{std_R:7.1f} max={max_R:+9.1f} "
             f"loss={stats.loss_total:+.3f}(p={stats.loss_policy:+.4f}) "
             f"kl={stats.approx_kl:+.4f} clip={stats.clip_fraction:.2f} "
             f"|g|={stats.grad_norm:.1f} Hk={stats.ent_k:.2f} "
             f"drone={frac_drone:.2f} collect={frac_collect:.2f} trunc={frac_trunc:.2f} "
             f"lr={scheduler.lr:.1e} {dt:.1f}s")

        logger.write_epoch({
            "global_epoch": gep, "n_instances": n_inst_per_epoch,
            "mean_return": mean_R, "std_return": std_R, "max_return": max_R,
            "loss_total": stats.loss_total, "loss_policy": stats.loss_policy,
            "approx_kl": stats.approx_kl, "clip_fraction": stats.clip_fraction,
            "grad_norm": stats.grad_norm, "ent_k": stats.ent_k,
            "frac_drone_launch": frac_drone, "frac_collect": frac_collect,
            "frac_truncated": frac_trunc, "lr": scheduler.lr, "wall_seconds": dt,
            "entropy_coef": float(ent_coef),
        })

        # ---- periodic (cheap) beam eval + best.pt promotion ----
        is_eval = bench_cfg.eval_every > 0 and (gep + 1) % bench_cfg.eval_every == 0
        if is_eval:
            ev = _beam_eval(policy, bench_envs, bench_insts, bench_cfg,
                            bench_cfg.eval_beam_width, bench_cfg.eval_beam_n_aug)
            per = " ".join(f"{k}={('NA' if v is None else f'{v:+.0f}')}"
                           for k, v in ev["per_file"].items())
            _log(bench_cfg, f"EVAL(beam{bench_cfg.eval_beam_width}) mean={ev['mean']:+.1f}  [{per}]")
            logger.write({"type": "beam_eval", "global_epoch": gep, **ev})

        should_save = (gep + 1) % bench_cfg.checkpoint_every == 0 or gep + 1 == bench_cfg.total_epochs
        if should_save:
            save_checkpoint(ckpt_dir / "last.pt", policy=policy,
                            optimiser=trainer.optimiser, scheduler=scheduler,
                            train_state=train_state, cfg=cfg)

        if is_eval and ev["mean"] > train_state.best_eval_return:
            train_state = TrainState(
                stage_idx=0, epoch_in_stage=0,
                global_epoch=train_state.global_epoch,
                best_eval_return=ev["mean"],
            )
            _save_and_promote(ckpt_dir, policy, trainer, scheduler, train_state, cfg)
            _log(bench_cfg, f"** new best (beam mean {ev['mean']:+.1f}) -> best.pt **")
            logger.write({"type": "best_updated", "global_epoch": gep,
                          "beam_mean": ev["mean"], "per_file": ev["per_file"]})

    # always persist final weights first, so nothing below can lose them
    save_checkpoint(ckpt_dir / "last.pt", policy=policy,
                    optimiser=trainer.optimiser, scheduler=scheduler,
                    train_state=train_state, cfg=cfg)
    if not (ckpt_dir / "best.pt").exists():
        _save_and_promote(ckpt_dir, policy, trainer, scheduler, train_state, cfg)

    try:
        ev = _beam_eval(policy, bench_envs, bench_insts, bench_cfg,
                        bench_cfg.final_beam_width, bench_cfg.final_beam_n_aug)
        per = " ".join(f"{k}={('NA' if v is None else f'{v:+.0f}')}"
                       for k, v in ev["per_file"].items())
        _log(bench_cfg, f"FINAL EVAL(beam{bench_cfg.final_beam_width}) mean={ev['mean']:+.1f}  [{per}]")
        logger.write({"type": "final_beam_eval", "global_epoch": bench_cfg.total_epochs, **ev})
        if ev["mean"] > train_state.best_eval_return:
            train_state = TrainState(stage_idx=0, epoch_in_stage=0,
                                     global_epoch=train_state.global_epoch,
                                     best_eval_return=ev["mean"])
            _save_and_promote(ckpt_dir, policy, trainer, scheduler, train_state, cfg)
            _log(bench_cfg, f"** final beam is best (mean {ev['mean']:+.1f}) -> best.pt **")
    except Exception as e:
        _log(bench_cfg, f"[warn] final beam eval failed ({type(e).__name__}: {e}); "
                    f"keeping periodic best.pt (mean {train_state.best_eval_return:+.1f})")

    logger.close()
    _log(bench_cfg, f"done. best beam mean={train_state.best_eval_return:+.1f}  "
                f"weights -> {ckpt_dir/'best.pt'}")


def make_curriculum(bench_cfg: BenchTunedConfig, cfg: PPOConfig) -> None:
    cfg.curriculum = [CurriculumStage(
        n_lo=bench_cfg.n, n_hi=bench_cfg.n, epochs=bench_cfg.total_epochs,
        instances_per_epoch=len(bench_cfg.bench_files) * max(1, bench_cfg.instance_reps),
        lr=bench_cfg.lr, warmup_steps=5,
    )]
