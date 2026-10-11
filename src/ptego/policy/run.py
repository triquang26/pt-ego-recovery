from __future__ import annotations

import runpy
import sys

from ptego.policy.configs import REPO, register

SCRIPTS = {
    "train": "train.py",
    "norm-stats": "compute_norm_stats.py",
    "serve": "serve_policy.py",
}


def main() -> None:
    command, *args = sys.argv[1:]
    register()
    script = REPO / "third_party" / "openpi" / "scripts" / SCRIPTS[command]
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
