from __future__ import annotations

import sys as _sys

from ttpd._internal import hub
from ttpd._internal import paths as _paths

_sys.modules.setdefault(__name__ + ".hub", hub)
_sys.modules.setdefault(__name__ + "._paths", _paths)

__all__ = ["_paths", "hub"]
