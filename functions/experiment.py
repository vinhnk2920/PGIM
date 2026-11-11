"""Explicit methodology/configuration for a fresh, reported-method run."""

from functools import lru_cache
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile(settings):
    path = getattr(settings, "experiment_profile", None)
    return _load(str(path)) if path is not None else None


def reported(settings):
    config = profile(settings)
    return config is not None and config["methodology"] == "reported_aug2026"


def record_run(settings, stage):
    """Capture the exact configuration and code used by each freshly run stage."""
    config = profile(settings)
    if config is None:
        return
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "functions").rglob("*.py"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    result = {
        "stage": stage, "source_root": str(settings.source_root),
        "output_root": str(settings.output_root), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "profile": config, "code_sha256": hashes,
        "training_policy": "Fresh VAE and LightGBM fits using reported scalar hyperparameters",
    }
    (settings.folder("audit") / f"{stage}_run_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


def rental_buckets(area, q=5):
    """Original rank-first quantiles, including its tie-order dependence."""
    valid = area.dropna()
    result = pd.Series(index=area.index, dtype=object)
    q_use = min(q, valid.nunique())
    if q_use < 2:
        result.loc[valid.index] = "SZ1"
    else:
        labels = [f"SZ{i+1}" for i in range(q_use)]
        try:
            values = pd.qcut(valid.rank(method="first"), q=q_use, labels=labels, duplicates="drop")
        except ValueError:
            values = pd.cut(valid, bins=q_use, labels=labels, include_lowest=True, duplicates="drop")
        result.loc[valid.index] = values.astype(str)
    return result.fillna("SZ1")


def bucket_ranges(area, buckets):
    data = pd.DataFrame({"Area_Cleaned": area, "size_bucket": buckets}).dropna()
    ranges = data.groupby("size_bucket", as_index=False)["Area_Cleaned"].agg(lower_bound="min", upper_bound="max")
    return ranges.sort_values("size_bucket").reset_index(drop=True)


def sales_buckets(area, ranges):
    bins = [-np.inf] + ranges["upper_bound"].iloc[:-1].tolist() + [np.inf]
    return pd.cut(area, bins=bins, labels=ranges["size_bucket"].tolist(), include_lowest=True).astype(object).fillna("SZ1")


def record_panel(settings, market, full, mask, projects, months):
    config = profile(settings)
    if config is None:
        return
    observed = mask.astype(bool)
    result = {"market": market, "P": len(projects), "T": len(months),
              "observed_cells": int(observed.sum()), "first_month": months[0], "last_month": months[-1],
              "observed_price_sha256": hashlib.sha256(full[observed].tobytes()).hexdigest(),
              "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
              "series_sha256": hashlib.sha256("\n".join(projects).encode()).hexdigest()}
    result["expected"] = config["expected_panels"][market]
    result["passed"] = all(result[k] == v for k, v in result["expected"].items())
    folder = settings.folder("audit")
    (folder / f"{market}_observed_panel.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise ValueError(f"{market}: freshly built observed panel differs from reported panel: {result}")
