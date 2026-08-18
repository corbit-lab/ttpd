#a
from dataclasses import dataclass, field


@dataclass
class CurriculumStage:
    n_lo: int
    n_hi: int
    epochs: int
    instances_per_epoch: int
    lr: float
    warmup_steps: int


@dataclass
class PPOConfig:
    pomo_size: int = 32
    n_sample_power: float = 1.0   # weight n-sampling within a stage toward larger n
    anchor_sizes: tuple = (5, 10, 15, 20)
    anchor_frac: float = 0.34
    seed: int = 2026

    # PPO-clip (no critic; advantages come from the POMO group baseline)
    clip_eps: float = 0.2
    entropy_coef: float = 0.02        # peak entropy bonus, annealed over the run
    entropy_coef_final: float = 0.002 # floor
    min_lr: float = 2e-5
    extend_peak_lr: float = 6e-5
    target_kl: float = 0.02          

    # optimisation
    n_update_epochs: int = 3
    minibatch_size: int = 1024
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4

    normalise_advantage: bool = True

    curriculum: list[CurriculumStage] = field(
        default_factory=lambda: [
            CurriculumStage(n_lo=3, n_hi=10, epochs=30, instances_per_epoch=128, lr=1e-4, warmup_steps=5),
            CurriculumStage(n_lo=3, n_hi=15, epochs=30, instances_per_epoch=128, lr=1e-4, warmup_steps=5),
            CurriculumStage(n_lo=5, n_hi=22, epochs=60, instances_per_epoch=128, lr=1e-4, warmup_steps=5),
            # extension stage: resumes on top of the first 120 epochs (scheduler is
            # re-warmed to extend_peak_lr); anchor sampling keeps {5,10,15,20} fed.
            CurriculumStage(n_lo=4, n_hi=22, epochs=280, instances_per_epoch=128, lr=1e-4, warmup_steps=5),
        ]
    )

    log_every: int = 1
    checkpoint_every: int = 50

    device: str = "cuda"
