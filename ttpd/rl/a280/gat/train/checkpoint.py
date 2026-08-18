#a
import dataclasses
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from policy.attention_policy import AttentionPolicy
from train.config import PPOConfig
from train.scheduler import WarmupCosine


def warm_start_policy(policy, payload_state_dict) -> tuple[int, list]:
    model_sd = policy.state_dict()
    compat, skipped = {}, []
    for k, v in payload_state_dict.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        tgt_shape = model_sd[k].shape
        if tgt_shape == v.shape:
            compat[k] = v
        elif (v.dim() == 2 and len(tgt_shape) == 2 and tgt_shape[0] == v.shape[0]
              and tgt_shape[1] > v.shape[1]):
            # input width grew (appended features): copy old columns, zero new ones
            w = torch.zeros(tgt_shape, dtype=v.dtype)
            w[:, : v.shape[1]] = v
            compat[k] = w
        else:
            skipped.append(k)
    skipped += [k for k in model_sd if k not in payload_state_dict]
    model_sd.update(compat)
    policy.load_state_dict(model_sd)
    return len(compat), sorted(set(skipped))


@dataclass
class TrainState:
    stage_idx: int
    epoch_in_stage: int     
    global_epoch: int        
    best_eval_return: float 
    best_milp_gap: float = float("inf")  


def _atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_checkpoint(
    path: str | os.PathLike,
    *,
    policy: AttentionPolicy,
    optimiser: torch.optim.Optimizer,
    scheduler: WarmupCosine,
    train_state: TrainState,
    cfg: PPOConfig,
) -> None:
    payload = {
        "policy": policy.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": {
            "peak_lr": scheduler._peak_lr,
            "warmup": scheduler._warmup,
            "total": scheduler._total,
            "step": scheduler._step,
            "lr": scheduler.lr,
        },
        "train_state": dataclasses.asdict(train_state),
        "cfg": dataclasses.asdict(cfg),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
    }
    _atomic_save(payload, Path(path))


def load_checkpoint(
    path: str | os.PathLike,
    *,
    policy: AttentionPolicy,
    optimiser: torch.optim.Optimizer,
    scheduler: WarmupCosine,
    strict_cfg: PPOConfig | None = None,
    map_location: str | torch.device | None = None,
) -> TrainState:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)

    if strict_cfg is not None:
        saved_cfg = payload.get("cfg", {})
        # only the fields that would silently corrupt training if changed
        critical = ["pomo_size"]
        live = dataclasses.asdict(strict_cfg)
        for k in critical:
            if saved_cfg.get(k) != live.get(k):
                raise RuntimeError(
                    f"Refusing to resume: cfg.{k} changed "
                    f"({saved_cfg.get(k)} -> {live.get(k)}). "
                    f"Start a fresh run or restore the previous value."
                )

    policy.load_state_dict(payload["policy"])
    optimiser.load_state_dict(payload["optimiser"])

    sd = payload["scheduler"]
    scheduler._peak_lr = sd["peak_lr"]
    scheduler._warmup = sd["warmup"]
    scheduler._total = sd["total"]
    scheduler._step = sd["step"]
    scheduler.lr = sd["lr"]

    # RNG restore is best-effort: a serialisation mismatch must not crash a resume.
    try:
        rng = payload["torch_rng"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.cpu().to(torch.uint8)
        torch.set_rng_state(rng)
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        np.random.set_state(payload["numpy_rng"])
    except Exception as e:
        print(f"[resume] warning: could not restore RNG state ({e}); continuing with fresh RNG")

    return TrainState(**payload["train_state"])


def find_latest_checkpoint(run_dir: str | os.PathLike) -> Path | None:
    p = Path(run_dir) / "checkpoints" / "last.pt"
    return p if p.exists() else None


def promote_to_best(last_path: str | os.PathLike, best_path: str | os.PathLike) -> None:
    last_path = Path(last_path)
    best_path = Path(best_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=best_path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        shutil.copyfile(last_path, tmp)
        os.replace(tmp, best_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
