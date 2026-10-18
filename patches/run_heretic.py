"""Launch Heretic with the MPS svd_lowrank patch applied.

Use exactly like the `heretic` command (same flags; config.toml is read from the CWD):

    .venv/bin/python patches/run_heretic.py [heretic flags...]
"""

import sys

import mps_svd_lowrank  # noqa: F401  (applies the patch on import)
from heretic.main import main

if __name__ == "__main__":
    sys.argv[0] = "heretic"
    sys.exit(main())
