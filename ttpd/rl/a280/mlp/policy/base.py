#a
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

@dataclass
class ActionSample:
    action: dict
    log_prob: float          # sum of the five head log-probs
    entropy: float           # sum of the five head entropies
    head_log_probs: dict

@runtime_checkable
class Policy(Protocol):
    def act(self, obs: dict[str, Any], env: Any, *, deterministic: bool = False) -> ActionSample:
        ...
    def evaluate(self, obs: dict[str, Any], env: Any, action: dict[str, int]) -> ActionSample:
        ...

