"""Compatibility helpers for the retained evaluation-reporting API."""

import numpy as np
import pandas as pd
from functions.common import add_months
from functions.forecasting.target_transform import normalize_target_transform

MERGE_KEYS = ["horizon", "window_idx", "project_idx", "period"]


def effective_origin_train_end(test_start_yyyymm: int, horizon_months: int) -> int:
    """
    Last origin month *p* such that ``add_months(p, horizon_months) < test_start``.

    Matches the train masks in ``run_rolling_imputed`` / ``run_rolling_obs_only``.
    For ``test_start = T`` and horizon *h*, this is ``add_months(T, -(h + 1))``.
    """
    return add_months(test_start_yyyymm, -(horizon_months + 1))


def merge_rolling_horizon(
    imp_df: pd.DataFrame, obs_df: pd.DataFrame, *, target_transform: str = "none"
) -> pd.DataFrame:
    """Combine imputed and ObsOnly rolling outputs for one horizon.

    ``target``, ``y_pred_*``, and ``target_obs`` are stored on **model scale**
    (log or level, matching ``target_transform``).  Residuals are likewise on
    model scale so the invariant ``residual = target - y_pred`` always holds in
    the returned frame.  Call ``predictions_frame_to_level`` before saving CSVs
    to convert everything to SGD/sqft.
    """
    normalize_target_transform(target_transform)
    if imp_df is None or len(imp_df) == 0:
        return pd.DataFrame()
    out = imp_df.copy()
    if obs_df is not None and len(obs_df):
        out = out.merge(
            obs_df[MERGE_KEYS + ["target_obs", "y_pred_obs_only"]], on=MERGE_KEYS, how="left"
        )
    else:
        out["target_obs"] = np.nan
        out["y_pred_obs_only"] = np.nan
    out["residual_imputed"] = out["target"] - out["y_pred_imputed"]
    out["residual_obs_only"] = np.nan
    m_obs = out["y_pred_obs_only"].notna() & out["target_obs"].notna()
    if m_obs.any():
        out.loc[m_obs, "residual_obs_only"] = (
            out.loc[m_obs, "target_obs"] - out.loc[m_obs, "y_pred_obs_only"]
        )
    return out
