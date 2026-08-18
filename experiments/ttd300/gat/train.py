import argparse
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

from ttpd.hub import ensure_local  # noqa: E402

import torch

from train.config import PPOConfig
from train.train import RunConfig, default_log, run

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="cap epochs per stage (smoke tests)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads")
    p.add_argument("--curriculum",
                   choices=["default", "small", "small10", "small20",
                            "pure10", "pure20"],
                   default="default",
                   help="training curriculum: 'small' = n4-11 specialist, "
                        "'small10' = n8-12, 'small20' = n18-22, "
                        "'pureN' = size-N instances only")
    p.add_argument("--init-from", default=None,
                   help="warm-start policy weights from this checkpoint (fresh run)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the curriculum's epoch count (single-stage curricula)")
    p.add_argument("--ed-frac-range", default="0.25,1.0",
                   help="per-episode endurance frac range lo,hi (x d_max); "
                        "'none' = unbounded")
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = PPOConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # ttd300 anchor sizes are the small benchmark sizes
    cfg.anchor_sizes = (10, 20)
    if args.curriculum == "small":
        # small-n specialist: n4-11 with anchor sampling on the benchmark sizes
        # and a sustained entropy floor to avoid premature mode collapse.
        from train.config import CurriculumStage
        cfg.curriculum = [CurriculumStage(n_lo=4, n_hi=11, epochs=250,
                                          instances_per_epoch=128, lr=7e-5, warmup_steps=5)]
        cfg.anchor_sizes = (10,)
        cfg.anchor_frac = 0.55
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.03
    elif args.curriculum == "small10":
        # n=10 specialist: same recipe as 'small', centred on n=10.
        from train.config import CurriculumStage
        cfg.curriculum = [CurriculumStage(n_lo=8, n_hi=12, epochs=250,
                                          instances_per_epoch=128, lr=7e-5, warmup_steps=5)]
        cfg.anchor_sizes = (10,)
        cfg.anchor_frac = 0.6
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.03
    elif args.curriculum == "small20":
        # n=20 specialist: same recipe as 'small', centred on n=20.
        from train.config import CurriculumStage
        cfg.curriculum = [CurriculumStage(n_lo=18, n_hi=22, epochs=250,
                                          instances_per_epoch=128, lr=7e-5, warmup_steps=5)]
        cfg.anchor_sizes = (20,)
        cfg.anchor_frac = 0.6
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.03
    elif args.curriculum.startswith("pure"):
        from train.config import CurriculumStage
        N = int(args.curriculum[4:])
        ep = 600 if N <= 12 else 350
        cfg.curriculum = [CurriculumStage(n_lo=N, n_hi=N, epochs=ep,
                                          instances_per_epoch=128, lr=1e-4, warmup_steps=5)]
        cfg.anchor_frac = 0.0
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.02

    if args.epochs is not None:
        for st in cfg.curriculum:
            st.epochs = args.epochs

    if args.ed_frac_range.strip().lower() == "none":
        ed_frac_range = None
    else:
        lo, hi = (float(x) for x in args.ed_frac_range.split(","))
        ed_frac_range = (lo, hi)

    # specialists grade the eval on their target size only
    eval_sizes = {"small10": (10,), "small20": (20,), "small": (10,)}.get(
        args.curriculum, (10, 20))
    if args.curriculum.startswith("pure"):
        eval_sizes = (int(args.curriculum[4:]),)

    run_cfg = RunConfig(run_dir=args.run_dir, resume=not args.no_resume,
                        init_from=args.init_from, eval_sizes=eval_sizes,
                        eval_n_lo=min(eval_sizes), eval_n_hi=max(max(eval_sizes), 22),
                        ed_frac_range=ed_frac_range)
    run(cfg, run_cfg=run_cfg, on_epoch=default_log, max_epochs=args.max_epochs)

if __name__ == "__main__":
    main()
