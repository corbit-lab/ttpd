#a
# linear warmup, cosine decay to min_lr; reset per curriculum stage

import math

class WarmupCosine:
    def __init__(self, lr: float, warmup_steps: int, total_steps: int,
                 min_lr: float = 0.0):
        self.reset(lr, warmup_steps, total_steps, min_lr)

    def reset(self, lr: float, warmup_steps: int, total_steps: int,
              min_lr: float = 0.0) -> None:
        if warmup_steps < 0 or total_steps < 0:
            raise ValueError("warmup_steps and total_steps must be non-negative.")
        if warmup_steps > total_steps:
            warmup_steps = total_steps
        self._peak_lr = float(lr)
        self._min_lr = float(min_lr)
        self._warmup = int(warmup_steps)
        self._total = int(total_steps)
        self._step = 0
        self.lr = self._min_lr if warmup_steps > 0 else float(lr)

    def step(self) -> float:
        self._step += 1
        if self._step <= self._warmup:
            self.lr = self._peak_lr * (self._step / max(1, self._warmup))
        else:
            decay_steps = max(1, self._total - self._warmup)
            t = (self._step - self._warmup) / decay_steps
            t = min(t, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            self.lr = self._min_lr + (self._peak_lr - self._min_lr) * cos
        return self.lr
