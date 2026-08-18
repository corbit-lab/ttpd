import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None

def _to_jsonable(obj: Any) -> Any:
    # recursively convert dataclasses, numpy scalars, etc. to json friendly types
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


class JSONLLogger:
    # it appends logger writing on json object per line 
    def __init__(self, run_dir: str | os.PathLike):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.jsonl"
        self._fh = open(self.log_path, "a", buffering=1)  # line-buffered

    # header 
    def write_header(self, *, cfg: Any, extra: dict | None = None) -> None:
        rec = {
            "type": "header",
            "wall_time": time.time(),
            "git_hash": _git_hash(),
            "cfg": _to_jsonable(cfg),
            "extra": _to_jsonable(extra or {}),
        }
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    # epoch summary 
    def write_epoch(self, log: Any) -> None:
        rec = {"type": "epoch", "wall_time": time.time(),
                **_to_jsonable(log)}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    # arbitrary event
    def write(self, rec: dict) -> None:
        rec = {"type": rec.get("type", "event"),
                "wall_time": time.time(), **_to_jsonable(rec)}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# results would be like:
# {"epoch": 1, "reward": 42000}
# {"epoch": 2, "reward": 43500}
# {"epoch": 3, "reward": 44100}