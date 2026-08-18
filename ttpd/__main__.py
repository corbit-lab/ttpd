from __future__ import annotations
import sys
from ttpd._internal.hub import main

def _run() -> None:
    if "--stacks" in sys.argv:
        from ttpd._internal import paths
        print("\n".join(sorted(paths.available())))
        return
    main()

if __name__ == "__main__":
    _run()
