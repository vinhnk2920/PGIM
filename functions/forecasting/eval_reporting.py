"""
eval_reporting.py
-----------------
Shared evaluation tables, per-window breakdowns, and leaderboard plots
for static and rolling LightGBM rental notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from functions.forecasting.metrics import evaluate
from functions.forecasting.rolling_helpers import effective_origin_train_end
from functions.forecasting.target_transform import normalize_target_transform, to_level


def fair_subset(
    rw_df: pd.DataFrame,
    horizon: int,
    *,
    target_col: str = "target",
) -> pd.DataFrame:
    """Rows used for fair imputed-model metrics (known imputed / evaluation target)."""
    if rw_df is None or len(rw_df) == 0:
        return rw_df
    if target_col not in rw_df.columns:
        raise KeyError(f"fair_subset: missing column {target_col!r}")
    return rw_df[
        (rw_df["horizon"] == horizon) & rw_df["fair_eval_row"] & rw_df[target_col].notna()
    ].copy()


def merge_row_stats(merged_h: pd.DataFrame, horizon: int) -> dict[str, int]:
    """Counts for §8.final-style logging."""
    n_all = len(merged_h)
    fair = fair_subset(merged_h, horizon)
    n_fair = len(fair)
    n_obs = int(fair["y_pred_obs_only"].notna().sum()) if n_fair else 0
    return {
        "horizon": horizon,
        "n_all": n_all,
        "n_fair": n_fair,
        "n_fair_obs_pred": n_obs,
    }


def print_merge_stats(merged_h: pd.DataFrame, horizon: int) -> None:
    s = merge_row_stats(merged_h, horizon)
    print(
        f"h={horizon}: merged {s['n_all']:,} prediction rows "
        f"(fair eval: {s['n_fair']:,}; ObsOnly preds on fair: {s['n_fair_obs_pred']:,})"
    )


def evaluate_imputed_block(
    imp_df: pd.DataFrame,
    horizon: int,
    results_list: list | None = None,
    *,
    tag_prefix: str = "Rolling",
    target_transform: str = "none",
) -> pd.DataFrame:
    """Fair metrics for imputed panel only (after imputed tuning block)."""
    fair = fair_subset(imp_df, horizon)
    print(f"\n--- Post-tuning eval (imputed block only) h={horizon} ---")
    if len(fair) == 0:
        print("  (no fair rows)")
        return fair
    evaluate(
        f"{tag_prefix} h={horizon} Imputed only (fair rows)",
        fair["target"].values,
        fair["y_pred_imputed"].values,
        results_list,
        target_transform=target_transform,
    )
    return fair


def evaluate_merged_horizon(
    imp_df: pd.DataFrame,
    obs_df: pd.DataFrame | None,
    horizon: int,
    results_list: list | None = None,
    *,
    test_windows: list | None = None,
    tag_prefix: str = "Rolling",
    show_per_window: bool = True,
    target_transform: str = "none",
) -> pd.DataFrame:
    """Fair metrics after imputed + ObsOnly blocks are merged for one horizon."""
    from functions.forecasting.rolling_helpers import merge_rolling_horizon

    merged = merge_rolling_horizon(imp_df, obs_df, target_transform=target_transform)
    print(f"\n--- Post-tuning eval (merged) h={horizon} ---")
    print_merge_stats(merged, horizon)

    fair = fair_subset(merged, horizon)
    if len(fair) == 0:
        print("  (no fair rows)")
        return merged

    evaluate(
        f"{tag_prefix} h={horizon} Imputed (fair vs ObsOnly)",
        fair["target"].values,
        fair["y_pred_imputed"].values,
        results_list,
        target_transform=target_transform,
    )
    fair_obs = fair[fair["y_pred_obs_only"].notna()]
    if len(fair_obs):
        evaluate(
            f"{tag_prefix} h={horizon} ObsOnly (fair rows)",
            fair_obs["target_obs"].values,
            fair_obs["y_pred_obs_only"].values,
            results_list,
            target_transform=target_transform,
        )

    if show_per_window and test_windows is not None:
        print_per_window_breakdown(
            merged,
            horizon,
            test_windows,
            target_transform=target_transform,
        )

    return merged


def evaluate_pooled_horizons(
    rw_df: pd.DataFrame,
    results_list: list | None = None,
    *,
    horizons: tuple[int, ...] = (1, 12, 24),
    test_windows: list | None = None,
    tag_prefix: str = "Rolling",
    show_per_window: bool = True,
    target_transform: str = "none",
) -> None:
    """§9-style pooled fair evaluation across all windows."""
    print("Aggregate metrics (fair_eval_row only, pooled global R²/RMSE):\n")
    for h in horizons:
        fair = fair_subset(rw_df, h)
        if len(fair) == 0:
            print(f"  h={h}: no fair rows")
            continue
        evaluate(
            f"{tag_prefix} h={h} Imputed (fair vs ObsOnly)",
            fair["target"].values,
            fair["y_pred_imputed"].values,
            results_list,
            target_transform=target_transform,
        )
        fair_obs = fair[fair["y_pred_obs_only"].notna()]
        if len(fair_obs):
            evaluate(
                f"{tag_prefix} h={h} ObsOnly (fair rows)",
                fair_obs["target_obs"].values,
                fair_obs["y_pred_obs_only"].values,
                results_list,
                target_transform=target_transform,
            )
        if show_per_window and test_windows is not None:
            sub = rw_df[rw_df["horizon"] == h]
            print_per_window_breakdown(
                sub,
                h,
                test_windows,
                target_transform=target_transform,
            )


def print_per_window_breakdown(
    rw_h: pd.DataFrame,
    horizon: int,
    test_windows: list,
    *,
    target_transform: str = "none",
) -> None:
    """Per-window RMSE / MAE / R² on fair rows (macro view by quarter)."""
    fair = fair_subset(rw_h, horizon)
    if len(fair) == 0:
        return
    mode = normalize_target_transform(target_transform)
    print(f"\nPer-window breakdown (fair rows) — h={horizon}:")
    print(
        f"{'Win':>4}  {'Test period':>18}  {'n':>6}  "
        f"{'RMSE_imp':>9}  {'RMSE_obs':>9}  {'MAE_imp':>8}  {'MAE_obs':>8}  "
        f"{'R2_imp':>8}  {'R2_obs':>8}"
    )
    print("-" * 98)
    for w, grp in fair.groupby("window_idx"):
        win = test_windows[int(w) - 1]
        ps = f"{win['test_start']}-{win['test_end']}"
        y = to_level(grp["target"].values, mode)
        p_i = to_level(grp["y_pred_imputed"].values, mode)
        r_i = float(np.sqrt(mean_squared_error(y, p_i)))
        m_i = float(mean_absolute_error(y, p_i))
        r2_i = float(r2_score(y, p_i)) if len(y) > 1 else float("nan")
        g_o = grp[grp["y_pred_obs_only"].notna()]
        if len(g_o) > 0:
            yo = to_level(g_o["target_obs"].values, mode)
            po = to_level(g_o["y_pred_obs_only"].values, mode)
            r_o = float(np.sqrt(mean_squared_error(yo, po)))
            m_o = float(mean_absolute_error(yo, po))
            r2_o = float(r2_score(yo, po)) if len(yo) > 1 else float("nan")
        else:
            r_o, m_o, r2_o = float("nan"), float("nan"), float("nan")
        print(
            f"{w:>4}  {ps:>18}  {len(grp):>6,}  {r_i:>9.4f}  {r_o:>9.4f}  "
            f"{m_i:>8.4f}  {m_o:>8.4f}  {r2_i:>8.4f}  {r2_o:>8.4f}"
        )


def get_result(results_list: list, key_fragment: str) -> dict[str, Any]:
    for r in results_list:
        if key_fragment in r.get("Model", ""):
            return r
    return {
        "Model": "",
        "n": 0,
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "MAPE(%)": float("nan"),
        "R2": float("nan"),
    }


def print_leaderboard_table(results_list: list, *, title: str | None = None) -> list[dict]:
    """Text leaderboard + return records for CSV."""
    if title is None:
        title = "LEADERBOARD — Rolling window (fair eval: observed forward targets)"

    rows_lb = [
        ("h=1  │ Imputed   │ pooled fair", get_result(results_list, "h=1 Imputed")),
        ("h=1  │ No-Impute │ pooled fair", get_result(results_list, "h=1 ObsOnly")),
        None,
        ("h=12 │ Imputed   │ pooled fair", get_result(results_list, "h=12 Imputed")),
        ("h=12 │ No-Impute │ pooled fair", get_result(results_list, "h=12 ObsOnly")),
        None,
        ("h=24 │ Imputed   │ pooled fair", get_result(results_list, "h=24 Imputed")),
        ("h=24 │ No-Impute │ pooled fair", get_result(results_list, "h=24 ObsOnly")),
    ]

    sep = "=" * 90
    dash = "-" * 90
    print(sep)
    print(title)
    print(sep)
    print(f"{'Task':<42} {'n':>7} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'R2':>8}")
    print(dash)
    records = []
    for row in rows_lb:
        if row is None:
            print(dash)
            continue
        label, r = row
        print(
            f"{label:<42} {r['n']:>7,} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} "
            f"{r['MAPE(%)']:>7.2f} {r['R2']:>8.4f}"
        )
        records.append({"task": label, **{k: r[k] for k in ("n", "RMSE", "MAE", "MAPE(%)", "R2")}})
    print(sep)
    return records


def plot_leaderboard_bars(
    results_list: list,
    output_root: Path,
    save_figure_fn,
    *,
    filename: str = "leaderboard_bar_metrics.png",
    suptitle: str | None = None,
) -> None:
    """2×2 bar chart (RMSE, MAE, MAPE, R²) — same layout as static notebook."""
    import matplotlib.pyplot as plt

    res_specs = [
        (get_result(results_list, "h=1 Imputed"), "h=1 Imputed"),
        (get_result(results_list, "h=1 ObsOnly"), "h=1 No-Imp"),
        (get_result(results_list, "h=12 Imputed"), "h=12 Imputed"),
        (get_result(results_list, "h=12 ObsOnly"), "h=12 No-Imp"),
        (get_result(results_list, "h=24 Imputed"), "h=24 Imputed"),
        (get_result(results_list, "h=24 ObsOnly"), "h=24 No-Imp"),
    ]

    metrics = ["RMSE", "MAE", "MAPE(%)", "R2"]
    titles = ["RMSE (SGD/sqft) ↓", "MAE (SGD/sqft) ↓", "MAPE (%) ↓", "R² ↑"]
    fmts = [".4f", ".4f", ".2f", ".4f"]

    color_imp = "#1976D2"
    color_obs = "#D32F2F"
    x_pos = [0, 1, 2.5, 3.5, 5.0, 6.0]
    x_labels = [s[1] for s in res_specs]
    colors = [color_imp if "Imputed" in lbl else color_obs for _, lbl in res_specs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric, title, fmt in zip(axes.flatten(), metrics, titles, fmts):
        vals = [float(s[0].get(metric, float("nan"))) for s in res_specs]
        bars = ax.bar(x_pos, vals, 0.6, color=colors, alpha=0.88, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015 if bar.get_height() >= 0 else bar.get_height() * 0.985,
                    f"{v:{fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=12)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    if suptitle is None:
        suptitle = (
            "Rolling window — VAE-Imputed vs No-Imputation (pooled global metrics, fair rows)"
        )
    fig.legend(
        handles=[
            Patch(facecolor=color_imp, label="Imputed (VAE)"),
            Patch(facecolor=color_obs, label="No-Imputation baseline"),
        ],
        loc="upper center",
        ncol=2,
        fontsize=14,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
    )
    fig.suptitle(suptitle, fontsize=16, y=1.04, fontweight="bold")
    plt.tight_layout()
    save_figure_fn(filename, output_root)
    plt.show()


def rolling_window_summary_df(
    rw_df: pd.DataFrame,
    test_windows: list,
    horizons: tuple[int, ...] = (1, 12, 24),
    *,
    target_col: str = "target",
    target_transform: str = "none",
) -> pd.DataFrame:
    """Per-window fair metrics for CSV export.

    ``train_end`` is the last month in the fixed-width rolling **calendar pool** (month
    before ``test_start``). The last **origin month that can enter the model** for horizon
    *h* is ``train_origin_end`` = ``add_months(test_start, -(h+1))`` (requires
    ``add_months(origin, h) < test_start``).
    """
    mode = normalize_target_transform(target_transform)
    rows = []
    for h in horizons:
        fair = fair_subset(rw_df, h, target_col=target_col)
        for w, grp in fair.groupby("window_idx"):
            win = test_windows[int(w) - 1]
            g_o = grp[grp["y_pred_obs_only"].notna()]
            y = to_level(grp[target_col].values, mode)
            p_i = to_level(grp["y_pred_imputed"].values, mode)
            row = {
                "horizon": h,
                "window": w,
                "test_start": win["test_start"],
                "test_end": win["test_end"],
                "train_start": win["train_start"],
                "train_end": win["train_end"],
                "train_origin_end": effective_origin_train_end(win["test_start"], h),
                "n_fair": len(grp),
                "RMSE_imputed": round(float(np.sqrt(mean_squared_error(y, p_i))), 4),
                "MAE_imputed": round(float(mean_absolute_error(y, p_i)), 4),
                "R2_imputed": round(float(r2_score(y, p_i)), 4) if len(y) > 1 else None,
            }
            if len(g_o):
                yo = to_level(g_o["target_obs"].values, mode)
                po = to_level(g_o["y_pred_obs_only"].values, mode)
                row["RMSE_obs_only"] = round(float(np.sqrt(mean_squared_error(yo, po))), 4)
                row["MAE_obs_only"] = round(float(mean_absolute_error(yo, po)), 4)
                row["R2_obs_only"] = round(float(r2_score(yo, po)), 4) if len(yo) > 1 else None
            else:
                row["RMSE_obs_only"] = None
                row["MAE_obs_only"] = None
                row["R2_obs_only"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def _imputed_eval_column(rw_df: pd.DataFrame, imputed_target_col: str | None) -> str:
    """Column holding the target used for imputed-model scores (`target` vs `y_true`)."""
    if imputed_target_col is not None:
        if imputed_target_col not in rw_df.columns:
            raise KeyError(imputed_target_col)
        return imputed_target_col
    if "target" in rw_df.columns:
        return "target"
    if "y_true" in rw_df.columns:
        return "y_true"
    raise KeyError("Need column target or y_true for imputed-model evaluation")


def _obs_eval_column(rw_df: pd.DataFrame, obs_target_col: str | None) -> str | None:
    """Observed forward rent column for ObsOnly metrics, if present."""
    if obs_target_col is not None:
        return obs_target_col if obs_target_col in rw_df.columns else None
    return "target_obs" if "target_obs" in rw_df.columns else None


def print_yearly_r2_obs_counts_audit(
    rw_df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 12, 24),
    *,
    fair_only: bool = True,
    imputed_target_col: str | None = None,
    obs_target_col: str | None = None,
    label: str = "",
    target_transform: str = "none",
) -> dict[int, dict[str, int]]:
    """
    Print how many rows enter each horizon's yearly R² breakdown.

    For **imputed** yearly R²: all fair-eval rows with a known eval target (same pool as
    :func:`yearly_r2_table`). ``sum(n_imputed by year)`` must equal that total.

    For **ObsOnly** yearly R²: fair-eval subset where ``y_pred_obs_only`` and observed
    target are non-null. ``sum(n_obs_only by year)`` must equal that total.

    R² within a single calendar year can still be NaN if that year has ``n < 2`` or zero
    variance in ``y_true`` — the printed counts are row counts before that filter.

    Parameters
    ----------
    label : str
        Printed prefix (e.g. ``'Rolling'`` / ``'Static'``) when comparing notebooks.
    """
    itc = _imputed_eval_column(rw_df, imputed_target_col)
    otc = _obs_eval_column(rw_df, obs_target_col)

    title = "Yearly R² — observation count audit"
    if label:
        title = f"[{label}] {title}"
    print(f"\n{title}")
    print(f"  eval target column (imputed): {itc!r}", end="")
    if otc:
        print(f"  |  ObsOnly target column: {otc!r}")
    else:
        print("  |  ObsOnly target column: (none — ObsOnly yearly R² skipped)")

    out: dict[int, dict[str, int]] = {}
    for h in horizons:
        sub = rw_df[rw_df["horizon"] == h].copy()
        if fair_only:
            pool = fair_subset(sub, h, target_col=itc)
        else:
            pool = sub[sub[itc].notna()].copy()

        n_fair_imp = len(pool)
        tbl = yearly_r2_table(
            rw_df,
            h,
            fair_only=fair_only,
            imputed_target_col=imputed_target_col,
            obs_target_col=obs_target_col,
            target_transform=target_transform,
        )
        sum_y_imp = int(tbl["n_imputed"].sum()) if len(tbl) else 0
        match_imp = "OK" if n_fair_imp == sum_y_imp else "MISMATCH"

        if otc and otc in pool.columns:
            m_obs = pool["y_pred_obs_only"].notna() & pool[otc].notna()
            n_obs = int(m_obs.sum())
            sum_y_obs = int(tbl["n_obs_only"].sum()) if len(tbl) else 0
            match_obs = "OK" if n_obs == sum_y_obs else "MISMATCH"
        else:
            n_obs = 0
            sum_y_obs = 0
            match_obs = "n/a"

        print(
            f"  h={h:2d}  Imputed R² rows: {n_fair_imp:,}  "
            f"(sum of yearly n_imputed: {sum_y_imp:,})  {match_imp}",
        )
        print(
            f"         ObsOnly R² rows: {n_obs:,}  "
            f"(sum of yearly n_obs_only: {sum_y_obs:,})  {match_obs}",
        )
        out[h] = {
            "n_imputed_r2_rows": n_fair_imp,
            "n_obs_only_r2_rows": n_obs,
        }
    print(
        "  Rolling vs static: totals match only if the **same origin months** enter the "
        "fair-eval pool (same TEST_CUTOFF / windows, features, and masks).\n",
    )
    return out


def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² when defined; NaN if too few points or zero variance in y_true."""
    if len(y_true) < 2:
        return float("nan")
    if float(np.var(y_true)) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def yearly_r2_table(
    rw_df: pd.DataFrame,
    horizon: int,
    *,
    fair_only: bool = True,
    imputed_target_col: str | None = None,
    obs_target_col: str | None = None,
    target_transform: str = "none",
) -> pd.DataFrame:
    """
    R² by calendar year of origin month (``period // 100``).

    When ``fair_only=True`` (default), uses ``fair_eval_row`` and rows with a known
    imputed / evaluation target (column ``target`` in rolling outputs, ``y_true`` in static export).

    Imputed column compares the imputed evaluation target vs ``y_pred_imputed``.
    ObsOnly column compares ``target_obs`` vs ``y_pred_obs_only`` when ``target_obs`` exists.
    """
    itc = _imputed_eval_column(rw_df, imputed_target_col)
    otc = _obs_eval_column(rw_df, obs_target_col)

    sub = rw_df[rw_df["horizon"] == horizon].copy()
    if fair_only:
        sub = fair_subset(sub, horizon, target_col=itc)
    else:
        sub = sub[sub[itc].notna()].copy()

    if len(sub) == 0:
        return pd.DataFrame(
            columns=[
                "horizon",
                "year",
                "n_imputed",
                "R2_imputed",
                "n_obs_only",
                "R2_obs_only",
            ],
        )

    mode = normalize_target_transform(target_transform)
    sub["_year"] = (sub["period"].astype(np.int64) // 100).astype(int)
    rows: list[dict[str, Any]] = []
    for year, grp in sub.groupby("_year"):
        y = to_level(grp[itc].values, mode)
        p_i = to_level(grp["y_pred_imputed"].values, mode)
        n_i = len(grp)
        r2_i = _r2_safe(y, p_i)

        if otc and otc in grp.columns:
            g_o = grp[grp["y_pred_obs_only"].notna() & grp[otc].notna()]
            yo = to_level(g_o[otc].values, mode)
            po = to_level(g_o["y_pred_obs_only"].values, mode)
            n_o = len(g_o)
            r2_o = _r2_safe(yo, po) if n_o else float("nan")
        else:
            n_o = 0
            r2_o = float("nan")

        rows.append(
            {
                "horizon": horizon,
                "year": int(year),
                "n_imputed": int(n_i),
                "R2_imputed": r2_i,
                "n_obs_only": int(n_o),
                "R2_obs_only": r2_o,
            },
        )

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def plot_yearly_r2_diagnostic(
    rw_df: pd.DataFrame,
    output_root: Path,
    save_figure_fn,
    *,
    horizons: tuple[int, ...] = (1, 12, 24),
    fair_only: bool = True,
    filename: str = "yearly_r2_diagnostic_rolling.png",
    suptitle: str | None = None,
    run_label: str = "Rolling window",
    print_audit: bool = True,
    audit_label: str = "",
    target_transform: str = "none",
    **yearly_table_kw,
) -> pd.DataFrame:
    """
    Line plots of yearly R² (one panel per horizon) + concatenated table for CSV export.

    Extra keyword arguments are forwarded to :func:`yearly_r2_table`
    (e.g. ``imputed_target_col='y_true'`` for static export CSVs).

    When ``print_audit=True``, prints row counts entering yearly R² so you can compare
    rolling vs static horizon-by-horizon (:func:`print_yearly_r2_obs_counts_audit`).
    """
    import matplotlib.pyplot as plt

    if print_audit:
        lab = audit_label or run_label
        print_yearly_r2_obs_counts_audit(
            rw_df,
            horizons=horizons,
            fair_only=fair_only,
            label=lab,
            target_transform=target_transform,
            **yearly_table_kw,
        )

    tables = []
    n_h = len(horizons)
    fig, axes = plt.subplots(n_h, 1, figsize=(10, max(4.0, 3.6 * n_h)), sharex=False)
    if n_h == 1:
        axes = np.array([axes])
    color_imp = "#1976D2"
    color_obs = "#D32F2F"

    for ax, h in zip(axes.flat, horizons):
        tbl = yearly_r2_table(
            rw_df,
            h,
            fair_only=fair_only,
            target_transform=target_transform,
            **yearly_table_kw,
        )
        tables.append(tbl)
        if len(tbl) == 0:
            ax.text(0.5, 0.5, f"h={h}: no rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"h={h}")
            continue
        ax.plot(
            tbl["year"],
            tbl["R2_imputed"],
            "o-",
            label="Imputed (eval target vs pred)",
            color=color_imp,
            linewidth=2,
            markersize=7,
        )
        if tbl["R2_obs_only"].notna().any():
            ax.plot(
                tbl["year"],
                tbl["R2_obs_only"],
                "s-",
                label="ObsOnly (target_obs vs pred)",
                color=color_obs,
                linewidth=2,
                markersize=7,
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
        ax.set_ylabel("R²")
        ax.set_title(
            f"h={h} — yearly R² (origin year = period // 100; "
            f"{'fair rows' if fair_only else 'all rows w/ eval target'})",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xticks(tbl["year"].values)

    axes[-1].set_xlabel("Calendar year (from origin month)")
    if suptitle is None:
        suptitle = f"{run_label} — yearly R² diagnostic"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_figure_fn(filename, output_root)
    plt.show()

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


STATIC_PREDICTIONS_FILENAMES = {
    "rental": "rental_predictions_lgbm_static.csv",
    "sales": "sales_predictions_lgbm_static.csv",
}

SIZE_TIER_LABELS = {
    "SZ1": "very_small",
    "SZ2": "small",
    "SZ3": "medium",
    "SZ4": "large",
    "SZ5": "very_large",
}
SIZE_TIER_ORDER = tuple(SIZE_TIER_LABELS.keys())
OVERALL_BUCKET_KEY = "__overall__"


def resolve_static_predictions_csv(
    predictions_csv: str | Path | None = None,
    *,
    csv_filename: str = "rental_predictions_lgbm_static.csv",
    search_root: str | Path = ".",
) -> Path:
    """
    Resolve a static-run predictions export path.

    If ``predictions_csv`` is set, that path is used (must exist).
    Otherwise picks the newest ``search_root/results_static*/<csv_filename>``.
    """
    if predictions_csv is not None:
        path = Path(predictions_csv)
        if not path.is_file():
            raise FileNotFoundError(f"Predictions CSV not found: {path}")
        return path.resolve()

    root = Path(search_root)
    candidates = sorted(
        root.glob(f"results_static*/{csv_filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {csv_filename!r} under {root.resolve()}/results_static*/ — "
            "set predictions_csv to an explicit path.",
        )
    return candidates[0].resolve()


def _augment_size_tier_from_sidecar(
    df: pd.DataFrame,
    run_dir: Path,
    *,
    size_col: str = "size_tier",
) -> pd.DataFrame:
    """Join ``size_tier`` from the run's feature-matrix export when predictions omit it."""
    if size_col in df.columns:
        return df
    if "period" not in df.columns:
        raise KeyError(
            f"predictions CSV missing {size_col!r} and cannot merge (need period)",
        )

    sidecar_names = (
        "Rental_Data_Size_Bucket_Level_With_Imputation.csv",
        "Sales_Data_Size_Bucket_Level_With_Imputation.csv",
    )
    for name in sidecar_names:
        sidecar = run_dir / name
        if not sidecar.is_file():
            continue
        aux = pd.read_csv(sidecar, low_memory=False)
        if size_col not in aux.columns:
            continue
        period_col = "period" if "period" in aux.columns else "yyyymm"
        merge_keys = ["period"]
        key_cols = [period_col, size_col]
        if "project_idx" in df.columns and "project_idx" in aux.columns:
            merge_keys = ["project_idx", "period"]
            key_cols = ["project_idx", period_col, size_col]
        elif "condo_name" in df.columns and "condo_name" in aux.columns:
            merge_keys = ["condo_name", "period"]
            key_cols = ["condo_name", period_col, size_col]
            if "district" in df.columns and "district" in aux.columns:
                merge_keys = ["condo_name", "district", "period"]
                key_cols = ["condo_name", "district", period_col, size_col]
        else:
            continue

        aux = aux[key_cols].rename(columns={period_col: "period"})
        aux["period"] = aux["period"].astype(int)
        aux = aux.drop_duplicates(merge_keys)
        out = df.merge(aux, on=merge_keys, how="left")
        n_miss = int(out[size_col].isna().sum())
        if n_miss:
            print(f"  Warning: {n_miss:,} rows still missing {size_col} after merge from {name}")
        print(f"  Merged {size_col} from {sidecar.name}")
        return out

    raise KeyError(
        f"predictions CSV missing {size_col!r} and no sidecar "
        f"({', '.join(sidecar_names)}) in {run_dir}",
    )


def load_static_predictions_export(
    predictions_csv: str | Path | None = None,
    *,
    csv_filename: str = "rental_predictions_lgbm_static.csv",
    search_root: str | Path = ".",
    size_col: str = "size_tier",
) -> tuple[pd.DataFrame, Path]:
    """Load ``*_predictions_lgbm_static.csv``; returns ``(dataframe, output_root)``."""
    path = resolve_static_predictions_csv(
        predictions_csv,
        csv_filename=csv_filename,
        search_root=search_root,
    )
    df = pd.read_csv(path)
    for col in ("was_observed", "target_observed", "fair_eval_row"):
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False}).astype("boolean")
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(int)
    if "period" in df.columns:
        df["period"] = df["period"].astype(int)
    df = _augment_size_tier_from_sidecar(df, path.parent, size_col=size_col)
    return df, path.parent


MODEL_PRED_OBS_COLS = (
    ("Imputed", "y_pred_imputed", "y_true"),
    ("ObsOnly", "y_pred_obs_only", "target_obs"),
)


def _period_to_datetime(periods: pd.Series) -> pd.DatetimeIndex:
    p = periods.astype(int)
    y = p // 100
    mo = p % 100
    return pd.to_datetime(
        {"year": y, "month": mo, "day": 1},
        errors="coerce",
    )


def _size_tier_display(tier: str) -> str:
    s = str(tier)
    if s == OVERALL_BUCKET_KEY:
        return "overall"
    return SIZE_TIER_LABELS.get(s, s)


def _ordered_size_tiers(tiers) -> list:
    present = {str(t) for t in tiers}
    ordered = [t for t in SIZE_TIER_ORDER if t in present]
    extras = sorted(present - set(SIZE_TIER_ORDER))
    return ordered + extras


def _aggregate_mean_pred_obs_slice(
    dt: pd.DataFrame,
    pred_col: str,
    obs_col: str,
    mode: str,
    *,
    horizon: int,
    model_name: str,
    size_tier: str,
) -> pd.DataFrame | None:
    m = dt[pred_col].notna() & dt[obs_col].notna()
    if not m.any():
        return None
    obs_lvl = to_level(dt.loc[m, obs_col].values, mode)
    pred_lvl = to_level(dt.loc[m, pred_col].values, mode)
    tmp = pd.DataFrame(
        {
            "period": dt.loc[m, "period"].values,
            "obs_level": obs_lvl,
            "pred_level": pred_lvl,
        }
    )
    g = tmp.groupby("period", as_index=False).agg(
        mean_obs=("obs_level", "mean"),
        mean_pred=("pred_level", "mean"),
        n=("period", "size"),
    )
    g["horizon"] = horizon
    g["size_tier"] = size_tier
    g["size_bucket"] = _size_tier_display(size_tier)
    g["model"] = model_name
    return g


def mean_pred_vs_obs_by_bucket_table(
    df_export: pd.DataFrame,
    *,
    test_cutoff: int = 202001,
    horizons: tuple[int, ...] = (1, 12, 24),
    size_col: str = "size_tier",
    target_transform: str = "none",
) -> pd.DataFrame:
    """
    Mean predicted vs mean observed PSF by origin month, horizon, size tier, and model.

    **Row filter (no imputed-origin rows):** ``was_observed`` origin months only, with an
    **observed forward target** (``target_observed``), in the test window. Observed means
    use ``y_true`` / ``target_obs`` (actual PSF), not VAE-filled targets.

    Includes five size tiers plus an **overall** bucket per horizon × model.
    """
    mode = normalize_target_transform(target_transform)
    if size_col not in df_export.columns:
        raise KeyError(f"mean_pred_vs_obs_by_bucket_table: missing {size_col!r}")

    sub = df_export[
        (df_export["was_observed"] == True)
        & (df_export["target_observed"] == True)
        & (df_export["period"] >= test_cutoff)
    ].copy()

    rows = []
    for h in horizons:
        dh = sub[sub["horizon"] == h]
        tiers = _ordered_size_tiers(dh[size_col].dropna().unique())
        for model_name, pred_col, obs_col in MODEL_PRED_OBS_COLS:
            for tier in tiers:
                dt = dh[dh[size_col] == tier]
                g = _aggregate_mean_pred_obs_slice(
                    dt,
                    pred_col,
                    obs_col,
                    mode,
                    horizon=h,
                    model_name=model_name,
                    size_tier=tier,
                )
                if g is not None:
                    rows.append(g)
            g_all = _aggregate_mean_pred_obs_slice(
                dh,
                pred_col,
                obs_col,
                mode,
                horizon=h,
                model_name=model_name,
                size_tier=OVERALL_BUCKET_KEY,
            )
            if g_all is not None:
                rows.append(g_all)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_mean_pred_obs_panel(ax, sl: pd.DataFrame) -> None:
    sl = sl.sort_values("period")
    x = _period_to_datetime(sl["period"])
    ax.plot(
        x,
        sl["mean_obs"],
        "o-",
        color="#333333",
        linewidth=1.8,
        markersize=4,
        label="Mean observed",
    )
    ax.plot(
        x,
        sl["mean_pred"],
        "s-",
        color="#1976D2",
        linewidth=1.8,
        markersize=4,
        label="Mean predicted",
    )
    ax.set_title(f"{sl['size_bucket'].iloc[0]}  (n={int(sl['n'].sum()):,})")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=35)


def plot_mean_pred_vs_obs_by_size_bucket(
    df_export: pd.DataFrame,
    output_root: Path,
    save_figure_fn,
    *,
    test_cutoff: int = 202001,
    horizons: tuple[int, ...] = (1, 12, 24),
    size_col: str = "size_tier",
    target_transform: str = "none",
    filename_prefix: str = "mean_pred_vs_obs_by_bucket",
    run_label: str = "Static test window",
) -> pd.DataFrame:
    """
    Line plots of mean predicted vs mean observed PSF by origin month.

    One figure per (horizon, model). Each figure has six panels: five size buckets
    (SZ1–SZ5) plus **overall**. Only **observed** origin rows with **observed**
    forward targets enter the means (see :func:`mean_pred_vs_obs_by_bucket_table`).
    """
    import matplotlib.pyplot as plt

    print(
        "Mean pred vs obs: using observed origins only (was_observed) with "
        "observed forward targets (target_observed); "
        f"test period >= {test_cutoff}. "
        "Observed line = actual y_true/target_obs PSF, not imputed targets.",
    )

    tbl = mean_pred_vs_obs_by_bucket_table(
        df_export,
        test_cutoff=test_cutoff,
        horizons=horizons,
        size_col=size_col,
        target_transform=target_transform,
    )
    if len(tbl) == 0:
        print("mean_pred_vs_obs: no rows after filters — skip plots")
        return tbl

    panel_keys = list(SIZE_TIER_ORDER) + [OVERALL_BUCKET_KEY]

    for h in horizons:
        th = tbl[tbl["horizon"] == h]
        if len(th) == 0:
            continue
        for model_name, _, _ in MODEL_PRED_OBS_COLS:
            tm = th[th["model"] == model_name]
            if len(tm) == 0:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
            for ax, tier_key in zip(axes.flat, panel_keys):
                sl = tm[tm["size_tier"] == tier_key]
                if len(sl) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "no data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(_size_tier_display(tier_key))
                    continue
                _plot_mean_pred_obs_panel(ax, sl)

            for ax in axes[:, 0]:
                ax.set_ylabel("Mean PSF (SGD/sqft)")
            fig.suptitle(
                f"{run_label} — h={h}m, {model_name}: mean predicted vs mean observed "
                f"(observed rows only, test ≥ {test_cutoff})",
                fontsize=12,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            model_slug = model_name.lower().replace(" ", "_")
            fname = f"{filename_prefix}_h{h}_{model_slug}.png"
            save_figure_fn(fname, output_root)
            plt.show()

    csv_path = output_root / f"{filename_prefix}_summary.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"Saved mean pred vs obs table → {csv_path}")
    return tbl


def plot_mean_pred_vs_obs_from_csv(
    predictions_csv: str | Path | None = None,
    *,
    csv_filename: str = "rental_predictions_lgbm_static.csv",
    search_root: str | Path = ".",
    output_root: str | Path | None = None,
    test_cutoff: int = 202001,
    horizons: tuple[int, ...] = (1, 12, 24),
    size_col: str = "size_tier",
    target_transform: str = "none",
    filename_prefix: str = "mean_pred_vs_obs_by_bucket",
    run_label: str = "Static test window",
):
    """
    Standalone entry point: load static predictions CSV and plot mean pred vs obs.

    Does not require notebook globals (``df_export``, ``OUTPUT_ROOT``, etc.).
    Figures and summary CSV are written under ``output_root`` (defaults to the
    folder containing the predictions file).
    """
    from functions.forecasting.utils import save_current_figure

    df_export, csv_parent = load_static_predictions_export(
        predictions_csv,
        csv_filename=csv_filename,
        search_root=search_root,
        size_col=size_col,
    )
    out = Path(output_root) if output_root is not None else csv_parent
    out.mkdir(parents=True, exist_ok=True)

    resolved = resolve_static_predictions_csv(
        predictions_csv,
        csv_filename=csv_filename,
        search_root=search_root,
    )
    print(f"Loaded predictions: {resolved}  ({len(df_export):,} rows)")
    print(f"Output folder: {out.resolve()}")

    return plot_mean_pred_vs_obs_by_size_bucket(
        df_export,
        out,
        save_current_figure,
        test_cutoff=test_cutoff,
        horizons=horizons,
        size_col=size_col,
        target_transform=target_transform,
        filename_prefix=filename_prefix,
        run_label=run_label,
    )
