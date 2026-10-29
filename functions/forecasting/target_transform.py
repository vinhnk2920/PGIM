"""
target_transform.py
-------------------
Optional transform of rent PSF before lag features and model targets.

Rolling / static notebooks can set ``TARGET_TRANSFORM`` to:

- ``'none'`` — model and lags on SGD/sqft (current behaviour).
- ``'log'``  — ``np.log(rent_psf)`` for ``rent_psf_imp`` / ``rent_psf_obs`` and all
  lag medians built from those columns; metrics are computed on **level** scale
  (``np.exp``) via :func:`to_level`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RENT_PSF_COLS = ("rent_psf_imp", "rent_psf_obs")
TARGET_TRANSFORM_CHOICES = ("none", "log", "log_growth")


def normalize_target_transform(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in TARGET_TRANSFORM_CHOICES:
        raise ValueError(
            f"target_transform must be one of {TARGET_TRANSFORM_CHOICES!r}, got {mode!r}",
        )
    return m


def to_model_scale(values, mode: str = "none") -> np.ndarray:
    """Map level PSF → model scale (identity or log)."""
    mode = normalize_target_transform(mode)
    x = np.asarray(values, dtype=float)
    if mode in ("none", "log_growth"):
        return x
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.log(x[ok])
    return out


def to_level(values, mode: str = "none") -> np.ndarray:
    """Map model-scale predictions/targets → SGD/sqft for reporting.

    For ``'log_growth'`` the rolling helpers invert predictions to level PSF
    *before* storing them in the output frame, so this function returns the
    values unchanged (identity) for that mode.
    """
    mode = normalize_target_transform(mode)
    x = np.asarray(values, dtype=float)
    if mode in ("none", "log_growth"):
        return x
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.exp(x[ok])
    return out


def apply_rent_psf_transform(df: pd.DataFrame, mode: str = "none") -> None:
    """
    In-place: transform ``rent_psf_imp`` and ``rent_psf_obs`` when ``mode='log'``.

    For ``'log_growth'`` this is a no-op: the original level PSF is needed as
    the origin price for the change-target inversion
    ``exp(log(price_t) + pred_change)``.

    Call **after** ``fill_missing`` and **before** ``build_lag_features``.
    """
    mode = normalize_target_transform(mode)
    if mode in ("none", "log_growth"):
        return
    for col in RENT_PSF_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        pos = s.notna() & (s > 0)
        df.loc[pos, col] = np.log(s.loc[pos].astype(float))
    n_bad = 0
    for col in RENT_PSF_COLS:
        if col in df.columns:
            n_bad += int((df[col].notna() & (df[col] <= 0)).sum())
    if n_bad:
        print(
            f"  Warning: {n_bad:,} non-positive {RENT_PSF_COLS} rows left NaN under log transform",
        )


def metric_label_suffix(mode: str = "none") -> str:
    """Short tag for evaluate() titles."""
    mode = normalize_target_transform(mode)
    return "" if mode == "none" else " [metrics on level PSF]"


def predictions_frame_to_level(df: pd.DataFrame, mode: str = "none") -> pd.DataFrame:
    """
    Copy prediction export frame with targets/preds/residuals on SGD/sqft.

    Columns converted when present: ``target``, ``y_pred_imputed``, ``target_obs``,
    ``y_pred_obs_only``; residuals recomputed on level.

    For ``'log_growth'`` this is a no-op: ``run_rolling_imputed`` /
    ``run_rolling_obs_only`` already store level-scale predictions and targets
    in the output frame.
    """
    mode = normalize_target_transform(mode)
    if mode in ("none", "log_growth") or df is None or len(df) == 0:
        return df

    out = df.copy()

    # Convert all model-scale columns back to SGD/sqft
    _convert_cols = (
        "target",
        "y_pred_imputed",
        "target_obs",
        "y_pred_obs_only",
        "rent_psf_imp",  # included in rw_df META_COLS; was log-transformed with df
    )
    for col in _convert_cols:
        if col in out.columns:
            out[col] = to_level(out[col].values, mode)

    # Recompute residuals on SGD/sqft level after conversion
    if "target" in out.columns and "y_pred_imputed" in out.columns:
        out["residual_imputed"] = out["target"] - out["y_pred_imputed"]
    if "target_obs" in out.columns and "y_pred_obs_only" in out.columns:
        out["residual_obs_only"] = np.nan
        m = out["y_pred_obs_only"].notna() & out["target_obs"].notna()
        if m.any():
            out.loc[m, "residual_obs_only"] = (
                out.loc[m, "target_obs"] - out.loc[m, "y_pred_obs_only"]
            )
    return out
