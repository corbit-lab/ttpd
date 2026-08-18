#a
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
_paths.use("rl/a280/mlp")

from ttpd.hub import ensure_local, instance as hub_instance  # noqa: E402

ROOT = _ROOT

import torch

from train.config import PPOConfig
from train.train import RunConfig, default_log, run

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--a280", default=hub_instance("a280", "a280_benchmark.txt"))
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="cap epochs per stage (smoke tests)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads (keep low when a MILP solve shares the box)")
    p.add_argument("--milp-dir", default=os.path.join(ROOT, "..", "MILP"),
                   help="dir with milp_results.csv (live gap refs)")
    p.add_argument("--curriculum",
                   choices=["default", "small", "small10", "small20", "bench",
                            "pure5", "pure10", "pure15", "pure20"],
                   default="default",
                   help="training curriculum: 'small' = n4-11 specialist, "
                        "'small10' = n8-12, 'small20' = n18-22, 'bench' = mixed "
                        "benchmark sizes, 'pureN' = size-N instances only")
    p.add_argument("--init-from", default=None,
                   help="warm-start policy weights from this checkpoint (fresh run)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the curriculum's epoch count (single-stage curricula)")
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    cfg = PPOConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.curriculum == "small":

        from train.config import CurriculumStage
        cfg.curriculum = [CurriculumStage(n_lo=4, n_hi=11, epochs=250,
                                          instances_per_epoch=128, lr=7e-5, warmup_steps=5)]
        cfg.anchor_sizes = (5, 10)
        cfg.anchor_frac = 0.55
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.03
    elif args.curriculum == "small10":

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
        # one model per problem size: every training instance is a fresh random
        # size-N instance; the benchmark instance is held out by construction.
        from train.config import CurriculumStage
        N = int(args.curriculum[4:])
        ep = 600 if N <= 12 else 350
        cfg.curriculum = [CurriculumStage(n_lo=N, n_hi=N, epochs=ep,
                                          instances_per_epoch=128, lr=1e-4, warmup_steps=5)]
        cfg.anchor_frac = 0.0
        cfg.entropy_coef = 0.06
        cfg.entropy_coef_final = 0.02
    elif args.curriculum == "bench":
        # single model for all four benchmark sizes: 80% of training instances
        # are drawn from {5,10,15,20}; low entropy/LR to preserve learnt modes.
        from train.config import CurriculumStage
        cfg.curriculum = [CurriculumStage(n_lo=4, n_hi=22, epochs=200,
                                          instances_per_epoch=128, lr=5e-5, warmup_steps=5)]
        cfg.anchor_sizes = (5, 10, 15, 20)
        cfg.anchor_frac = 0.8
        cfg.entropy_coef = 0.03
        cfg.entropy_coef_final = 0.01

    if args.epochs is not None:
        for st in cfg.curriculum:
            st.epochs = args.epochs

    # specialists track the benchmark gap on their target size only
    milp_sizes = {"small10": (10,), "small20": (20,)}.get(args.curriculum, (5, 10, 15, 20))
    if args.curriculum.startswith("pure"):
        milp_sizes = (int(args.curriculum[4:]),)

    milp_dir = args.milp_dir if os.path.isdir(args.milp_dir) else None
    run_cfg = RunConfig(run_dir=args.run_dir, resume=not args.no_resume,
                        milp_dir=milp_dir, init_from=args.init_from,
                        milp_sizes=milp_sizes)
    run(cfg, a280_path=args.a280, run_cfg=run_cfg, on_epoch=default_log,
        max_epochs=args.max_epochs)

if __name__ == "__main__":
    main()
