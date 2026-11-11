"""Rebuild transactions, train both VAEs and fit forecasts from raw inputs."""

from datetime import datetime
from pathlib import Path
import os
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent


def has_option(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


if not has_option("--experiment-profile"):
    sys.argv += ["--experiment-profile", str(ROOT / "config/experiments/reported_from_raw.json")]
if not has_option("--output-root"):
    sys.argv += ["--output-root", str(ROOT / "outputs" / ("from_raw_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))]
if not has_option("--rebuild-raw"):
    sys.argv += ["--rebuild-raw"]

from functions.cli import main

if __name__ == "__main__":
    main("all")
