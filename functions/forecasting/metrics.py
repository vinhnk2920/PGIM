"""
metrics.py
----------
Regression evaluation helpers used by all prediction notebooks.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from functions.forecasting.target_transform import (
    metric_label_suffix,
    normalize_target_transform,
    to_level,
)


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, ignoring zero-valued actuals."""
    yt, yp = np.array(y_true), np.array(y_pred)
    mask = yt != 0
    return 100.0 * np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]))


def evaluate(
    name: str,
    y_true,
    y_pred,
    store_list=None,
    *,
    target_transform: str = "none",
) -> float | None:
    """
    Print RMSE / MAE / MAPE / R² and optionally append a dict to store_list.

    When ``target_transform='log'``, *y_true* / *y_pred* are assumed to be on log
    scale and are converted with ``exp`` before metrics (SGD/sqft).

    Returns RMSE (float) or None if there are no samples.
    """
    if len(y_true) == 0:
        print(f"[{name}] — no samples to evaluate")
        return None

    mode = normalize_target_transform(target_transform)
    yt = to_level(y_true, mode)
    yp = to_level(y_pred, mode)
    suffix = metric_label_suffix(mode)
    display_name = f"{name}{suffix}" if suffix and suffix not in name else name

    rmse_ = np.sqrt(mean_squared_error(yt, yp))
    mae_ = mean_absolute_error(yt, yp)
    mape_ = mape(yt, yp)
    r2_ = r2_score(yt, yp)

    print("=" * 60)
    print(f"  {display_name}  (n={len(yt):,})")
    print(f"  RMSE : {rmse_:.4f} SGD/sqft")
    print(f"  MAE  : {mae_:.4f} SGD/sqft")
    print(f"  MAPE : {mape_:.2f} %")
    print(f"  R2   : {r2_:.4f}")

    if store_list is not None:
        store_list.append(
            {
                "Model": name,
                "n": len(y_true),
                "RMSE": rmse_,
                "MAE": mae_,
                "MAPE(%)": mape_,
                "R2": r2_,
            }
        )
    return rmse_
