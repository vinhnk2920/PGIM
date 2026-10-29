"""
utils.py
--------
Calendar helpers and cross-validation splitter shared by all notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def month_cv_split(X_df, y_ser, period_series, periods_arr, tscv):
    """
    Yield (X_tr, y_tr, X_va, y_va) tuples for each TimeSeriesSplit fold.
    Splitting is done at the monthly level so all rows belonging to the same
    calendar month stay in the same fold.
    """
    period_col = period_series.loc[X_df.index]
    for tr_idx, va_idx in tscv.split(periods_arr):
        tr_mask = period_col.isin(set(periods_arr[tr_idx]))
        va_mask = period_col.isin(set(periods_arr[va_idx]))
        yield (X_df.loc[tr_mask], y_ser.loc[tr_mask], X_df.loc[va_mask], y_ser.loc[va_mask])


def month_cv_index_split(period_series, periods_arr, tscv):
    """
    Yield (tr_row_indices, va_row_indices) — integer positions — for each fold.
    Useful when you need positional indices rather than boolean masks.
    """
    period_series = pd.Series(period_series).reset_index(drop=True)
    periods_arr = np.asarray(periods_arr)
    for tr_idx, va_idx in tscv.split(periods_arr):
        tr_months = set(periods_arr[tr_idx])
        va_months = set(periods_arr[va_idx])
        tr_rows = np.where(period_series.isin(tr_months).values)[0]
        va_rows = np.where(period_series.isin(va_months).values)[0]
        yield (tr_rows, va_rows)


def fair_keys_from_obs_test(df_obs_test: pd.DataFrame) -> set:
    """(project_idx, period) keys from the ObsOnly model's test frame."""
    return {
        (int(pid), int(per))
        for pid, per in zip(df_obs_test["project_idx"].values, df_obs_test["period"].values)
    }


def forward_target_observed_mask(
    project_idx, periods, horizon_months: int, obs_lookup
) -> np.ndarray:
    """True where the forward target period has at least one observed bucket."""
    return np.array(
        [
            obs_lookup.get((int(pid), add_months(int(per), horizon_months)), False)
            for pid, per in zip(project_idx, periods)
        ],
        dtype=bool,
    )


def fair_eval_mask(
    project_idx, periods, fair_keys: set, horizon_months: int, obs_lookup
) -> np.ndarray:
    """
    Fair comparison rows: forward target is observed AND origin is in ObsOnly test population.
      Used for both imputed and ObsOnly metrics so both models are scored on the same rows.
    """
    fwd = forward_target_observed_mask(project_idx, periods, horizon_months, obs_lookup)
    in_keys = np.array(
        [(int(pid), int(per)) in fair_keys for pid, per in zip(project_idx, periods)], dtype=bool
    )
    return fwd & in_keys


def save_current_figure(filename: str, output_root: Path) -> None:
    """Save the current matplotlib figure into output_root (figure still displays)."""
    out_path = output_root / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved figure → {out_path}")


from functions.common import add_months
