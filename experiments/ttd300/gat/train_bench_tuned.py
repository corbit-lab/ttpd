import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
while (_ROOT != os.path.dirname(_ROOT)
       and not os.path.isdir(os.path.join(_ROOT, "ttpd"))):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttpd import _paths  # noqa: E402
_paths.use("rl/ttd300/gat")

from ttpd.hub import ensure_local, weights_dir as hub_weights  # noqa: E402

ROOT = _ROOT

import torch

from train.bench_tuned import BenchTunedConfig, make_curriculum, run_bench_tuned
from train.config import PPOConfig

DATA_DIR = os.path.abspath(os.path.join(ROOT, "..", "Data"))
SIZES = [10, 20, 30, 40, 50, 75, 100]

def layout_files(data_dir: str, n: int) -> list[str]:
    """The 5 base layout files (no endurance header) -- the training pool."""
    files = sorted(f for f in glob.glob(os.path.join(data_dir, f"ttd300_n{n}_L*.txt"))
                   if "_f" not in os.path.basename(f))
    if not files:
        raise FileNotFoundError(f"no ttd300_n{n}_L*.txt in {data_dir}")
    return files

def endurance_files(data_dir: str, n: int, fracs: list[str]) -> list[str]:
    files = []
    for tag in fracs:
        files += sorted(glob.glob(os.path.join(data_dir, f"ttd300_n{n}_L*_{tag}.txt")))
    if not files:
        raise FileNotFoundError(f"no ttd300_n{n}_L*_f*.txt in {data_dir}")
    return files

def default_init_from(tag: str) -> str | None:
    p = os.path.join(hub_weights("ttd300", "gat", "sampled"), "n20", "best.pt")
    return p if os.path.exists(p) else None

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, choices=SIZES)
    p.add_argument("--tag", default="GAT", help="label prefixed to every log line")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--device", default=None, help="cuda / cpu / mps")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--init-from", default=None,
                   help="warm-start checkpoint (default: weights/new/<tag>_n20 best.pt; "
                        "pass '' to train from scratch)")
    p.add_argument("--instance-reps", type=int, default=1,
                   help="times each of the 5 layout instances appears per epoch")
    p.add_argument("--total-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=6e-5, help="fine-tune peak LR")
    p.add_argument("--min-lr-frac", type=float, default=0.5,
                   help="LR floor as a fraction of peak (cosine never decays below this)")
    p.add_argument("--entropy", type=float, default=0.003, help="peak entropy coef")
    p.add_argument("--entropy-final", type=float, default=0.0003, help="entropy coef floor")
    p.add_argument("--no-entropy-norm", action="store_true",
                   help="legacy raw-entropy bonus (n-dependent pressure)")
    p.add_argument("--target-kl", type=float, default=0.015,
                   help="early-stop the update epoch when mean KL exceeds 1.5x this")
    p.add_argument("--pomo", type=int, default=64,
                   help="POMO multi-start group size = per-step forward batch")
    p.add_argument("--minibatch", type=int, default=2048,
                   help="PPO minibatch (transitions) -- bigger keeps the GPU busy")
    p.add_argument("--update-epochs", type=int, default=4,
                   help="PPO passes per epoch over the rollout buffer")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--early-stop-evals", type=int, default=4,
                   help="stop the stage after this many consecutive non-improving "
                        "periodic evals (0 = run every epoch regardless)")
    p.add_argument("--eval-beam-width", type=int, default=16)
    p.add_argument("--eval-beam-n-aug", type=int, default=2)
    p.add_argument("--eval-fracs", default="f025,f050,f075,f100",
                   help="promotion-grid fracs (comma list); trim (e.g. f025,f100) to "
                        "speed up the periodic eval at large n")
    p.add_argument("--final-beam-width", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--beam-stratify", action="store_true",
                   help="POMO launch-stratified beam in eval (slow at n=50/100)")
    p.add_argument("--ed-frac-range", default="0.25,1.0",
                   help="per-episode training endurance frac range lo,hi (x d_max); "
                        "'none' = unbounded")
    p.add_argument("--ed-frac-power", type=float, default=1.0,
                   help="skew the frac draw toward the LOW (hard) end: "
                        "f = lo+(hi-lo)*u^p; 1.0 uniform, 2.0 ~2/3 below midpoint")
    p.add_argument("--synth-frac", type=float, default=0.0,
                   help="fraction of POMO groups drawn as fresh synthetic instances "
                        "instead of the 5 fixed layouts (regularises the tiny pool)")
    p.add_argument("--synth-n-range", default="",
                   help="synthetic instance size range lo,hi (default: n,n)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    init_from = default_init_from(args.tag) if args.init_from is None else (args.init_from or None)

    cfg = PPOConfig()
    cfg.seed = args.seed
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.pomo_size = args.pomo
    cfg.minibatch_size = args.minibatch
    cfg.n_update_epochs = args.update_epochs
    cfg.entropy_coef = args.entropy
    cfg.entropy_coef_final = args.entropy_final
    cfg.normalise_entropy = not args.no_entropy_norm
    cfg.target_kl = args.target_kl
    cfg.min_lr = args.lr * args.min_lr_frac   # floor scales with peak (no dead tail)

    if args.ed_frac_range.strip().lower() == "none":
        ed_frac_range = None
    else:
        lo, hi = (float(x) for x in args.ed_frac_range.split(","))
        ed_frac_range = (lo, hi)

    if args.synth_n_range.strip():
        s_lo, s_hi = (int(x) for x in args.synth_n_range.split(","))
    else:
        s_lo = s_hi = None

    bench_cfg = BenchTunedConfig(
        run_dir=args.run_dir,
        bench_files=layout_files(args.data_dir, args.n),
        eval_files=endurance_files(args.data_dir, args.n,
                                   [t.strip() for t in args.eval_fracs.split(",")]),
        n=args.n,
        tag=args.tag,
        init_from=init_from,
        instance_reps=args.instance_reps,
        total_epochs=args.total_epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_beam_width=args.eval_beam_width,
        eval_beam_n_aug=args.eval_beam_n_aug,
        final_beam_width=args.final_beam_width,
        checkpoint_every=args.checkpoint_every,
        beam_stratify=args.beam_stratify,
        scale_capacity=False,   # ttd300 files carry the per-size W; True would rescale
        resume=not args.no_resume,
        ed_frac_range=ed_frac_range,
        ed_frac_power=args.ed_frac_power,
        synth_frac=args.synth_frac,
        synth_n_lo=s_lo,
        synth_n_hi=s_hi,
        early_stop_evals=args.early_stop_evals,
    )
    make_curriculum(bench_cfg, cfg)
    run_bench_tuned(cfg, bench_cfg)

if __name__ == "__main__":
    main()
