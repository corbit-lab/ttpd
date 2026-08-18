"""Put the TTP-D stacks a script needs onto sys.path.

The code keeps the flat-module import style it was written and validated with
(``from core import evaluate``, ``from env import TTPDEnv``, ``from sa import sa``).
Rather than rewrite every import -- which would risk changing behaviour right
before submission -- this module is the single place that knows where each stack
lives, replacing the per-script ``HERE/ROOT/../..`` walking of the old layout.

    from ttpd import _paths
    _paths.use("core", "heuristics", "rl/a280/gat")

Stacks are directories under ``ttpd/``:

    core             constants, core, instance, masking, simulator, ttpd_common
                     -- one copy, verified byte-identical to the seven it replaced
    heuristics       sa, vns, greedy
    exact            Gurobi MILP models
    rl/a280/gat      env/, policy/, train/ -- fixed-endurance GAT
    rl/a280/mlp      the MLP-encoder ablation of the above
    rl/ttd300/gat    the endurance-conditioned stack
    lisa/a280        behaviour-cloning + hybrid pipeline
    lisa/ttd300

The three rl stacks are deliberately NOT merged: they differ in the endurance
conditioning and the encoder under ablation, so sharing them would change
published numbers.
"""

from __future__ import annotations

import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PKG)


def use(*stacks: str) -> None:
    """Prepend each named stack to sys.path (idempotent, order-preserving)."""
    for stack in reversed(stacks):
        p = os.path.join(PKG, *stack.split("/"))
        if not os.path.isdir(p):
            raise ValueError(
                f"unknown stack {stack!r}: no such directory {p}. "
                f"Available: {', '.join(sorted(available()))}")
        if p not in sys.path:
            sys.path.insert(0, p)
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)


def available() -> list[str]:
    """Every stack name `use()` accepts.

    A stack is a directory holding importable modules directly (``core``) or one
    whose children are packages to be imported by name (``rl/a280/gat``, which
    supplies ``env``/``policy``/``train``).
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "_internal")]
        rel = os.path.relpath(dirpath, PKG).replace(os.sep, "/")
        if rel == ".":
            continue
        holds_modules = any(f.endswith(".py") for f in filenames)
        holds_packages = any(
            os.path.exists(os.path.join(dirpath, d, "__init__.py"))
            for d in dirnames)
        if holds_modules or holds_packages:
            out.append(rel)
    return out


if __name__ == "__main__":
    print("\n".join(sorted(available())))
